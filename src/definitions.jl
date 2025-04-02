# ## DEFINICIÓN DEL PROBLEMA

"""
    create_nn(N,activation,hidden,depth)

Crea una red de Rᴺ->R, con `depth` capas totales, con ancho `hidden` en las capas ocultas y activación `activation` en todas las capas. 
"""
function create_nn(N,activation, hidden, depth)
    return Chain(
        Dense(N => hidden, activation),
        (Dense(hidden=>hidden,activation) for _ in 1:depth-2)...,
        Dense(hidden => 1)
    )
end

"""
    create_nets(N,activation,hidde,depth)

Crea un vector de `N+1` redes iguales, construidas con `create_nn`.

Alternativamente, se pueden englobar los parámetros en una `NamedTuple`.
"""
function create_nets(N,activation,hidden,depth)
    [create_nn(N,activation,hidden,depth) for _ in 1:N+1]
end

function create_nets(structure::NamedTuple)
    (;N,activation,hidden,depth) = structure
    create_nets(N,activation,hidden,depth)
end


#### CLASE N-DIMENSIONAL ###
# # Esto es un poco rebuscado, pero fue el único camino con el que logré que funcione.
# Para que Enzyme.gradient nos funcione bien necesitamos redes de Rⁿ->R. Pero si tenemos un problema en Rⁿ, entonces necesitamos n+1 funciones (v y todas las derivadas parciales). Y eso lo tenemos que empaquetar en un tipo de dato que sea subtipo de `Lux.AbstractLuxContainerLayer`. Esto no es sencillo, porque uno no puede (de manera natural) crear estructuras con un número indefinido de campos. Para lograr esto, implementé el macro `@crear_clase` que nos permite definir una estructura nueva, dándode un nombre e indicando la dimensión del problema a tratar. Esto elimina la vieja clase SolNN.

"""
    crear_nombres(N)

devuelve una tupla de símbolos con los nombres de las funciones que formarán las redes, para un problema en dimensión `N`.
```julia
julia> crear_nombres(3)
  (:v,:φ1,:φ2,:φ3)
```
"""
crear_nombres(N) = Tuple([:v,[Symbol.("φ$n") for n in 1:N]...])


"""
   @crear_clase

Macro para definir una nueva estructura de redes. Recibe el nombre de la estructura a crear y la dimensión del problema.

```julia
julia> @crear_clase Ejemplo3D 3
julia> redes = create_nets(3,Lux.σ,10,4) #dim 3, activación σ, ancho 10, profundidad 4
julia> nets = Ejemplo3D(redes...)
```

Además se define un constructor que recibe directamente una una `NamedTuple` con los datos del problema. 
"""
macro crear_clase(nombre,N)
    quote
    struct $nombre <: Lux.AbstractLuxContainerLayer{$(crear_nombres(N))}
        $(crear_nombres(N)...)
        function $nombre(t)
            new(create_nets(t)...)
        end
    end
    end
end


"""
    Distance(d,∇d)
Crea una variable de tipo `Distance` que almacena una función distancia y su gradiente. Notar que `∇d` debe ser un campo vectorial: para un vector `x`, debe ocurrir `length(∇d(x))==length(x)`.
"""

struct Distance{D,N<:Union{Nothing,Function}}
    dD::D
    dN::N
end
Distance(dD) = Distance(dD,nothing)

#Definimos cómo evaluar la función distancia (y sus derivadas en una)
function (dist::Distance)(x::Union{AbstractVector,AbstractMatrix})
    (;dD,dN) = dist
    return (dD(x),dN(x))
end

struct BoundaryData{D<:Union{Nothing,Function},N<:Union{Nothing,Function},nN<:Union{Nothing,Function}}
    gD::D
    gN::N
    η::nN
end



# Definimos una estructura para almacenar la información general del problema 
# struct ProblemData{F<:Function,G<:Function,D::Distance,M<:AbstractMatrix,B<:AbstractFloat}
#     gendata::F
#     f::G
#     dist::D
#     A::M
#     b::B
# end


struct ProblemData{I<:Integer,G<:Function,F<:Function,D<:Distance,BD<:BoundaryData,T<:AbstractMatrix,B<:AbstractFloat,E<:Union{Nothing,Function}}
    dim::I
    gendata!::G
    f::F
    dist::D
    bd::BD
    A::T
    b::B
    exact::E
    function ProblemData(dim,gendata!,f,dist,bd,A,b,exact)
        size(A,1)==size(A,2) || throw("`A` debe ser cuadrada y no de tamaño $(size(A))")
        size(A,1)==dim || throw("La matriz `A` no coincide con la dimensión del problema. Dimensión $dim, pero size(A) = $(size(A))")
        A = A |> xdev
        return new{typeof(dim),typeof(gendata!),typeof(f),typeof(dist),typeof(bd),typeof(A),typeof(b),typeof(exact)}(dim,gendata!,f,dist,bd,A,b,exact)
    end
end

#Un constructor para que se asigne A=I y b = 0 si no son especificados. 

function ProblemData(dim,gendata!,f,d::Distance,bd::BoundaryData)
    A = Matrix{Float32}(I(dim)) |> xdev
    return ProblemData(dim,gendata!,f,d,bd,A,0.0f0,nothing)
end 


function ProblemData(dim,gendata!,f,d::Distance,bd::BoundaryData,exact::E) where {E<:Function}
    A = Matrix{Float32}(I(dim))|>xdev
    return ProblemData(dim,gendata!,f,d,bd,A,0.0f0,exact)
end 
