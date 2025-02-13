# Load packages 
using OrdinaryDiffEq
using NODEData
using Plots 
using Flux
using Optimisers 
using SciMLSensitivity
using SciMLStructures
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

Flux.@functor NDE # Make Flux aware of the struct

# Learnable parameters datatype
mutable struct LearnableParams{T,S}
    θ::T
    θ1::S
end

Flux.@functor LearnableParams # Make Flux aware of the struct

# Mark LearnableParams as a SciMLStructure 
SciMLStructures.isscimlstructure(::LearnableParams) = true
ismutablescimlstructure(::LearnableParams) = true

# Declare that it contains a tunable portion
SciMLStructures.hasportion(::SciMLStructures.Tunable, ::LearnableParams) = true

function SciMLStructures.canonicalize(::SciMLStructures.Tunable, p::LearnableParams)
    # Concatenate all tunable values into a single vector
    buffer = vcat(p.θ, p.θ1)

    # Define a repack function to reconstruct LearnableParams from new values
    repack = let p = p
        function repack(newbuffer)
            SciMLStructures.replace(SciMLStructures.Tunable(), p, newbuffer)
        end
    end

    # Return buffer, repack function, and a flag indicating whether aliasing occurs (false here)
    return buffer, repack, false
end

function SciMLStructures.replace(::SciMLStructures.Tunable, p::LearnableParams, newbuffer)
    N = length(p.θ)
    @assert length(newbuffer) == N + 1  # Ensure correct length

    θ = newbuffer[1:N]  # Extract neural network weights
    θ1 = newbuffer[N+1:N+1] # Extract gamma scalar

    return LearnableParams(θ, θ1)
end

function replace!(::SciMLStructures.Tunable, p::LearnableParams, newbuffer)
    N = length(p.θ)
    @assert length(newbuffer) == N + 1  # Ensure correct length

    p.θ .= newbuffer[1:N]  # Update θ in place
    p.θ1 = newbuffer[N+1:N+1]  # Update θ1 scalar

    return p
end

Optimisers.trainable(m::NDE) = (p = m.p,) # Trainable parameters

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
t, X, y_sparse, mask = sparsify_data(sol, fraction=0.9) # 10% random sparsity

# Pack the full state into one matrix
sol_sparse = vcat(sol[1, :]', y_sparse')

# Create overlapping batches
train, valid = NODEDataloader(sol_sparse, t, 2; valid_set=0.2)

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

# Initialize the parameters
θ, U_re = Flux.destructure(U_nn)
θ1 = Float32[1.6] # γ now learnable
p = LearnableParams(θ, θ1)

# Define the neural ODE
function neural_ode(u, p::LearnableParams, t)
    x, y = u
    θ, θ1 = p.θ, p.θ1
    α_const = Float32(α)
    U = U_re(θ)(u)

    dx = α_const*x + U[1]
    dy = -θ1[1]*y + U[2]
    return [dx, dy]
end

# Create the NODE problem and model
node_prob = ODEProblem(neural_ode, u0, (Float32(0.), Float32(dt)), p)
model = NDE(node_prob; reltol=1f-5, dt=dt)

# Define the loss function
function loss(m, batch, truth, batch_mask; λ1=1f-2, λ2=1f-4)
    pred = m(batch)
    pred_x = pred[1, :]
    pred_y = pred[2, :]
    truth_x = truth[1, :]
    truth_y = truth[2, :]
    
    loss_x = Flux.mse(pred_x, truth_x)
    loss_y = Flux.mse(pred_y .* batch_mask, truth_y .* batch_mask)
    
    # Regularization
    params = [m.p.θ; m.p.θ1]
    reg1 = λ1 * sum(abs, params)
    reg2 = λ2 * sum(abs2, params)
    
    return loss_x + loss_y + reg1 + reg2
end

# Set up the optimiser
η = 1f-3
opt = Optimisers.Adam(η)
opt_state = Optimisers.setup(opt, model)

# Training parameters
τ_max = 2
epochs = 20
NN_train = length(train) > 100 ? 100 : length(train)
NN_valid = length(valid) > 100 ? 100 : length(valid)
scheduler = CosAnneal(η, 0f0, epochs, false)

# Training loop
Train = true
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
