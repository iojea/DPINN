
# FUNCIÓN DE PÉRDIDA
# La definimos en dos partes. Primero definimos una función `loss_function` que es comprensible y hace las cuentas. Esta recibe las tres redes que conforman una `SolNN`, los datos del problema agrupados en una estructura `ProblemData` y un conjunto de puntos. La segunda es la función `loss` que sigue la sintaxis necesaria para `Lux`, traduce las cosas al formato adecuado y llama a la `loss_function`.

@views function loss_function(
        fixednets,
        data::ProblemData,
        pts::AbstractArray
    )
    (;f,dist,A,b,exact) = data
    v       = first(fixednets)
    φs      = fixednets[2:end]
    vp      = v(pts)
    ∇vp     = Enzyme.gradient(Reverse, v, pts)[1]
    φp      = vcat([φ(pts) for φ in φs]...)
    divφp     = sum(Enzyme.gradient(Reverse,φ,pts)[1][i:i,:] for (i,φ) in enumerate(φs))
    dp,∇dp = dist(pts)
    fp      = f(pts)
    ∇up     = @. vp*∇dp  + dp*∇vp
    up      = dp.*vp
    ℓ = mean(abs2,sum(A*∇up-φp,dims=1)) + mean(abs2,-divφp + b*up - fp)
    e =  isnothing(exact) ? 0.0f0 : mean(abs2,up - exact(pts)) 
    return ℓ,e
end

# La función que ejecuta internamente `Lux` a través de `Training.single_train_step!`
function loss(model, ps, st, (pts,data))
    nom = propertynames(model)
    fixednets = [StatefulLuxLayer{true}(getfield(model,k),ps[k],st[k]) for k in nom]
    ℓ,error = loss_function(fixednets, data, pts)
    #error  = mse_loss_function(u_net, target_data, xyt)
    return (
        ℓ,
        NamedTuple{nom}((fun.st for fun in fixednets)),
        (; ℓ, error)
    )
end


# Entrenamiento:
# La principal diferencia con nuestro viejo entrenamiento es que aquí hago uso de `DataLoader` (viene en `Reactant`) para manipular los datos y pasárselos a `xdev`. Esencialmente `DataLoader` particiona los datos para hacer el entrenamiento usando batches aleatorios. Esto tiene sentido cuando los datos están dados (e.g.: en un problema de ajuste o de clasificación). Nosotros queremos samplear datos en un dominio que contiene infinito puntos. Como no queremos gastar memoria alojando varios millones de puntos, lo que hice fue mantener nuestra filosofía de tener un array pre-alocado de `n_points` puntos y resamplear los puntos en cada iteración. Pero además, esos `n_points` puntos se dividen en batches de tamaño `bs` (batch size). Por defecto, `bs` es un décimo de `n_points` de modo que en una iteración en realidad se hacen 10 (pseudo)pasos, con `n_points/10` datos. 
function train_model(nets::M,n_points,data; seed=0, bs::Int=n_points÷10, maxiters::Int=5000) where M<: AbstractLuxContainerLayer
    pts = zeros(Float32,data.dim,n_points) # reservo espacio para los puntos.
    #nets = create_nets(dim,structure)
    rng = Xoshiro(0)
    Random.seed!(rng,seed)
     
    ps, st = Lux.setup(rng, nets) |> xdev 
    train_state = Training.TrainState(nets, ps, st, Adam(0.05f0))
    step(i) =  0.05f0 / 2^(i÷1000)
    loss_list = []
    error_list= []
    iter = 1
    while iter < maxiters
        batch_loss = []
        batch_error = []
        data.gendata!(pts)
        dataloader = DataLoader(pts; batchsize=bs, shuffle=true, partial=false) |> xdev
        Optimisers.adjust!(train_state, step(iter))
        for pts_batch in dataloader
            _, ℓ, stats, train_state = Training.single_train_step!(
                AutoEnzyme(), loss,(pts_batch,data),train_state
            )

            #isnan(ℓ) && throw(ArgumentError("NaN Loss Detected"))
            push!(batch_loss,Float32(ℓ))
            push!(batch_error,Float32(stats.error))
        end
        push!(loss_list,mean(batch_loss))
        push!(error_list,mean(batch_error))
        if iter % 1000 == 1 || iter == maxiters
            println("Iteration:",iter,"/",maxiters)
            println("     Loss: ",last(loss_list))
            println("    Error: ",last(error_list))
        end
        iter += 1
    end
    return StatefulLuxLayer{true}(
        nets, cdev(train_state.parameters), cdev(train_state.states)
    ), loss_list,error_list
end

