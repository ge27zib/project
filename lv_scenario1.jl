# Load packages
using OrdinaryDiffEq
using NODEData
using Plots 
using Flux
using Optimisers 
using SciMLSensitivity 
using Statistics
using Random 
using LinearAlgebra
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

# Generate the model dataset
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

# Create overlapping batches
train, valid = NODEDataloader(sol, 2; dt=dt, valid_set=0.2)

# plot(train.data') # Visualize the data

# Set up the ANN 
U_nn = Chain(
    Dense(2, 32, tanh),
    Dense(32, 32, tanh),
    Dense(32, 2))
θ, U_re = Flux.destructure(U_nn)

p_ln = θ # Learnable parameters

# Define neural ODE
function neural_ode(u, p_ln, t)
    x, y = u
    α_const = Float32(α)
    γ_const = Float32(γ)
    θ = p_ln
    U = U_re(θ)(u)
    
    dx = α_const*x + U[1]   
    dy = -γ_const*y + U[2]       
    return [dx, dy]
end

# Create the NODE problem and model
node_prob = ODEProblem(neural_ode, u0, (Float32(0.), Float32(dt)), p_ln)
model = NDE(node_prob; reltol=1f-5, dt=dt)

# Define the loss function
function loss(m, batch, truth; λ1=0f0, λ2=1f-5)
    loss = Flux.mse(m(batch), truth) # L2 loss
    reg1 = λ1 * sum(abs, m.p) # L1 regularization
    reg2 = λ2 * sum(abs2, m.p) # L2 regularization
    
    return loss + reg1 + reg2
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

        train_τ = NODEDataloader(sol, i_τ)
        @info "Batch size = $(i_τ)"
    
        for (epoch, η) in zip(1:epochs, scheduler)
            Optimisers.adjust!(opt_state, η)

            for batch in train_τ
                batch_t, batch_data = batch
                Flux.train!(model, [(batch_t, batch_data)], opt_state) do m, bt, bd
                    loss(m, (bt, bd), bd)
                end
            end
            
            train_loss = mean([loss(model, train[i], train[i][2]) for i=1:NN_train])
            valid_loss = mean([loss(model, valid[i], valid[i][2]) for i=1:NN_valid])
        
            @printf("Epoch %3d | LR: %.2e | Train Loss: %.4f | Valid Loss: %.4f\n", epoch, η, train_loss, valid_loss)

        end
    end
end

# # Save the model
# jldsave("lv_scenario1.jld2"; model=model, model_params=model.p)

# Load model and parameters
saved_data = load("lv_scenario1.jld2")
model = saved_data["model"]
p_ln = saved_data["model_params"]

#-----------------------------------------------------------------------------------------------------------------------------------------------
############################################ Prediction for the NODE: We visualize for the first 70 points #####################################
# ----------------------------------------------------------------------------------------------------------------------------------------------

# Trajectories
t = sol.t
x = Array(sol)  # Ground truth 
pred = model((t, x))[:, 1:70]

plot(pred[1, 1:70], label="UDE Approximation", lw=2, color=:red, legend=:topleft)
plot!(pred[2, 1:70], label="", lw=2, color=:red)
scatter!(x[1, 1:70], label="Measurements", marker=:circle, color=:black)
scatter!(x[2, 1:70], label="", marker=:circle, color=:black)

xlabel!("t")
ylabel!("x(t), y(t)")

# Interaction Terms
Uxy_pred = U_re(p_ln)(x)[:, 1:70]

plot(Uxy_pred[1, 1:70], label="UDE Approximation", lw=2, color=:red, legend=:topleft)
plot!(Uxy_pred[2, 1:70], label="", lw=2, color=:red)

Uxy_truth1 = -(β*x[1, :].*x[2, :])[1:70]
Uxy_truth2 = (δ*x[1, :].*x[2, :])[1:70]
Uxy_truth = vcat(Uxy_truth1', Uxy_truth2')

plot!(Uxy_truth[1, 1:70], label="True Interaction", lw=2, color=:black)
plot!(Uxy_truth[2, 1:70], label="", lw=2, color=:black)

xlabel!("t")
ylabel!("U(x,y)")

# L2 Loss
train_70 = NODEDataloader(sol, 70)
batch_t, batch_data = train_70[1]  
pred_70 = model((batch_t, batch_data))

l2_error = [norm(pred_70[:, i] - batch_data[:, i]) for i in 1:length(batch_t)] # Compute L2 error at each time step
l2_error = max.(l2_error, eps()) # Prevent log(0)

y_ticks = 10.0 .^ (-3.5:1.:0) # Custom ticks
y_labels = ["10^{$(round(y, digits=1))}" for y in log10.(y_ticks)] # Format labels as 10^x

plot(batch_t, l2_error, xlabel="t", ylabel="L2-Error", xlims=(0, batch_t[end]), yscale=:log10, yticks=(y_ticks, y_labels),  ylims=(minimum(y_ticks), maximum(y_ticks)), lw=2, color=:red, legend=false)

#---------------------------------------------------------------------------------------------------------------------------------------------
#################################################### Finding the Symbolic Expression #########################################################
#---------------------------------------------------------------------------------------------------------------------------------------------

x = Array(sol) # Ground truth
mlp = U_re(p_ln)
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
############################################### Successfully recovered the interaction terms ####################################################