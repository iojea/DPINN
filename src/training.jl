
# FUNCIÓN DE PÉRDIDA
# La definimos en dos partes. Primero definimos una función `loss_function` que es comprensible y hace las cuentas. Esta recibe las tres redes que conforman una `SolNN`, los datos del problema agrupados en una estructura `ProblemData` y un conjunto de puntos. La segunda es la función `loss` que sigue la sintaxis necesaria para `Lux`, traduce las cosas al formato adecuado y llama a la `loss_function`.

@views function loss_function(
        v::StatefulLuxLayer,
        φ₁::StatefulLuxLayer,
        φ₂::StatefulLuxLayer,
        data::ProblemData,
        pts::AbstractArray
    )
    (;f,dist,A,b,exact) = data
    e       = 0.0
    vp      = v(pts)
    ∇vp     = Enzyme.gradient(Enzyme.Reverse, v, pts)[1]
    vxp,vyp = ∇vp[1:1, :], ∇vp[2:2, :]
    φ₁p     = φ₁(pts)
    φ₁xp    = Enzyme.gradient(Enzyme.Reverse, φ₁, pts)[1][1:1, :]
    φ₂p     = φ₂(pts)
    φ₂yp    = Enzyme.gradient(Enzyme.Reverse, φ₂, pts)[1][2:2, :]
    dp,dxp,dyp = dist(pts)
    fp      = f(pts)
    uxp = @. dp*vxp + dxp*vp
    uyp = @. dp*vyp + dyp*vp
    ℓ =  mean(abs2,@. A[1,1]*uxp+A[1,2]*uyp-φ₁p) + mean(abs2,@. A[2,1]*uxp+A[2,2]*uyp-φ₂p) + mean(abs2,@. -φ₁xp - φ₂yp + b*dp*vp - fp)
    if !isnothing(exact)
        Up      = exact(pts)
        e =  mean(abs2,@. dp*vp - Up)
    end
    return ℓ,e
end

# La función que ejecuta internamente `Lux` a través de `Training.single_train_step!`
function loss(model, ps, st, (pts,data))
    v_net  = StatefulLuxLayer{true}(model.v, ps.v, st.v)
    φ₁_net = StatefulLuxLayer{true}(model.φ₁, ps.φ₁, st.φ₁)
    φ₂_net = StatefulLuxLayer{true}(model.φ₂, ps.φ₂, st.φ₂)
    ℓ,error = loss_function(v_net, φ₁_net, φ₂_net, data, pts)
    #error  = mse_loss_function(u_net, target_data, xyt)
    return (
        ℓ,
        (; v=v_net.st, φ₁=φ₁_net.st, φ₂=φ₂_net.st),
        (; ℓ, error)
    )
end


# Entrenamiento:
# La principal diferencia con nuestro viejo entrenamiento es que aquí hago uso de `DataLoader` (viene en `Reactant`) para manipular los datos y pasárselos a `xdev`. Esencialmente `DataLoader` particiona los datos para hacer el entrenamiento usando batches aleatorios. Esto tiene sentido cuando los datos están dados (e.g.: en un problema de ajuste o de clasificación). Nosotros queremos samplear datos en un dominio que contiene infinito puntos. Como no queremos gastar memoria alojando varios millones de puntos, lo que hice fue mantener nuestra filosofía de tener un array pre-alocado de `n_points` puntos y resamplear los puntos en cada iteración. Pero además, esos `n_points` puntos se dividen en batches de tamaño `bs` (batch size). Por defecto, `bs` es un décimo de `n_points` de modo que en una iteración en realidad se hacen 10 (pseudo)pasos, con `n_points/10` datos. 
function train_model(n_points,structure,data; seed=0, bs::Int=n_points÷10, maxiters::Int=5000)
    pts = zeros(Float32,2,n_points) # reservo espacio para los puntos.
    nets = SolNN(structure) # defino las redes
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
        data.gendata(pts)
        dataloader = DataLoader(pts; batchsize=bs, shuffle=true, partial=false) |> xdev
        Optimisers.adjust!(train_state, step(iter))
        for pts_batch in dataloader
            _, ℓ, stats, train_state = Training.single_train_step!(
                AutoEnzyme(), loss,(pts_batch,data),train_state
            )

            isnan(ℓ) && throw(ArgumentError("NaN Loss Detected"))
            push!(batch_loss,ℓ)
            push!(batch_error,stats.error)
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

