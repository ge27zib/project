# Load packages
using OrdinaryDiffEq
using NODEData
using Plots 
using Flux
using Optimisers 
using SciMLSensitivity 
using Statistics
using Random 
using Printf
using ParameterSchedulers 
using JLD2
using SymbolicRegression

#-----------------------------------------------------------------------------------------------------------------------------------------------
############################################### Development of the Neural Ordinary Differential Equation #######################################
#-----------------------------------------------------------------------------------------------------------------------------------------------

# Model datatype for NODE
abstract type AbstractNDEModel end 

struct NDE{P,R,A,K} <: AbstractNDEModel
    p::P 
    prob::R 
    alg::A
    kwargs::K
end 

function NDE(prob; alg=Tsit5(), kwargs...)
    p = prob.p 
    return NDE{typeof(p),typeof(prob),typeof(alg),typeof(kwargs)}(p, prob, alg, kwargs)
end 

function (m::NDE)(X, p=m.p)
    (t, x) = X 
    Array(solve(remake(m.prob; tspan=(t[1], t[end]), u0=x[:,1], p=p), m.alg; saveat=t, m.kwargs...))
end

Flux.@functor NDE
Optimisers.trainable(m::NDE) = (p = m.p,)

# Generate dataset
function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, γ, δ = p 
    
    du[1] = α*x - β*x*y
    du[2] = -γ*y + δ*x*y
    return [du[1], du[2]]
end

α = 1.3; β = 0.9; γ = 1.8; δ = 0.8
p = Float32.([α, β, γ, δ])
tspan = (0f0, 50f0)
dt = 0.1f0

u0 = Float32.([0.44249296, 4.6280594])
prob = ODEProblem(lotka_volterra!, u0, tspan, p)
sol = solve(prob, Tsit5(), saveat=dt)

# Make dataset sparse
function sparsify_data(sol; fraction=1.)
    t = sol.t
    X = Array(sol)       
    
    mask = rand(length(t)) .< fraction 
    y_sparse = X[2, :] .* mask # By default 0% of the values are missing
    return t, X, y_sparse, mask
end

Random.seed!(1234)
t, X, y_sparse, mask = sparsify_data(sol)

# Pack the full state into one matrix
sol_sparse = vcat(sol[1, :]', y_sparse')

# Create overlapping batches
train, valid = NODEDataloader(sol_sparse, t, 2; valid_set=0.2) # Batch size is 2

# Extract mask slice for a given batch's time vector
function get_mask_for_batch(batch_t, global_mask, t0, dt)
    indices = round.(Int, (batch_t .- t0) ./ dt .+ 1)
    return global_mask[indices]
end

# plot(train.data') # Visualize the data

# Set up the ANN 
U_nn = Chain(
    Dense(2, 32, tanh),
    Dense(32, 32, tanh),
    Dense(32, 2))
θ, U_re = Flux.destructure(U_nn)

p = θ # Learnable parameters

# Define neural ODE
function neural_ode(u, p, t)
    x, y = u
    α_const = Float32(α)
    γ_const = Float32(γ)
    θ = p
    U = U_re(θ)(u)
    
    dx = α_const*x + U[1]   
    dy = -γ_const*y + U[2]       
    return [dx, dy]
end

# Create the NODE problem and model
node_prob = ODEProblem(neural_ode, u0, (Float32(0.), Float32(dt)), p)
model = NDE(node_prob; reltol=1f-5, dt=dt)

# Define the loss function
function loss(m, batch, truth, batch_mask; λ1=0f0, λ2=1f-5)
    pred = m(batch)
    pred_x = pred[1, :]
    pred_y = pred[2, :]
    truth_x = truth[1, :]
    truth_y = truth[2, :]
    
    loss_x = Flux.mse(pred_x, truth_x)
    loss_y = Flux.mse(pred_y .* batch_mask, truth_y .* batch_mask) # Loss is calculated only on the observed points
    
    # Reguralization
    reg1 = λ1 * sum(abs, m.p)
    reg2 = λ2 * sum(abs2, m.p)
    
    return loss_x + loss_y + reg1 + reg2
end

# Set up the optimiser
η = 1f-3
opt = Optimisers.Adam(η)
opt_state = Optimisers.setup(opt, model)

# Training parameters
τ_max = 10 # Maximum batch size
epochs = 20
NN_train = length(train) > 100 ? 100 : length(train)
NN_valid = length(valid) > 100 ? 100 : length(valid)
scheduler = CosAnneal(η, 0f0, epochs, false)

# Training loop
Train = false
if Train
    println("Training started...")
    
    for i_τ = 2:τ_max

        train_τ = NODEDataloader(sol_sparse, t, i_τ)
        @info "Batch size = $(i_τ)"
    
        for (epoch, η) in zip(1:epochs, scheduler)
            Optimisers.adjust!(opt_state, η)

            for batch in train_τ
                batch_t, batch_data = batch
                batch_mask = get_mask_for_batch(batch_t, mask, t[1], dt)
                Flux.train!(model, [(batch_t, batch_data)], opt_state) do m, bt, bd
                    loss(m, (bt, bd), bd, batch_mask)
                end
            end
            
            train_loss = mean([loss(model, train[i], train[i][2], get_mask_for_batch(train[i][1], mask, t[1], dt)) for i=1:NN_train])
            valid_loss = mean([loss(model, valid[i], valid[i][2], get_mask_for_batch(valid[i][1], mask, t[1], dt)) for i=1:NN_valid])
        
            @printf("Epoch %3d | LR: %.2e | Train Loss: %.4f | Valid Loss: %.4f\n", epoch, η, train_loss, valid_loss)

        end
    end
end

# Prediction for the first 70 points

# Trajectories
y_sp = sol_sparse[2, 1:70]
y_spskip = y_sp[y_sp .!= 0] # True nonzero y values

pred = model((t, sol_sparse))[:, 1:70]
plot(pred')
plot!(sol_sparse[1, 1:70], seriestype = :scatter, legend = false)
plot!(y_spskip, seriestype = :scatter, legend = false)

# Interaction terms
Uxy_pred = U_re(model.p.θ)(sol_sparse)[:, 1:70]
plot(Uxy_pred')

Uxy_truth1 = -(β.*sol[1, :].*sol[2, :] )[1:70]
Uxy_truth2 = (δ.*sol[1, :].*sol[2, :])[1:70]
Uxy_truth = vcat(Uxy_truth1', Uxy_truth2')
plot!(Uxy_truth')

# # Save the model
# jldsave("lv_no_sparse_gamma_fixed.jld2"; model=model, model_params=model.p)

# # Load model and parameters
# saved_data = load("lv_no_sparse_gamma_fixed.jld2")
# model = saved_data["model"]
# θ = saved_data["model_params"]

#---------------------------------------------------------------------------------------------------------------------------------------------
#################################################### Finding the Symbolic Expression #########################################################
#---------------------------------------------------------------------------------------------------------------------------------------------

x = Array(sol) # Ground truth
mlp = U_re(θ)
dx = mlp(x)

options = SymbolicRegression.Options(
    binary_operators=[+, *, /, -],
    unary_operators=[],
)

hall_of_fame = EquationSearch(
    x,
    dx,
    niterations=100, 
    options=options,  # What operations are allowed
    parallelism=:multithreading,
)

pareto1 = calculate_pareto_frontier(x, dx[1, :], hall_of_fame[1], options)
pareto1[3].tree

pareto2 = calculate_pareto_frontier(x, dx[2, :], hall_of_fame[2], options)
pareto2[3].tree
####################################################### Recovered the interaction terms ########################################################