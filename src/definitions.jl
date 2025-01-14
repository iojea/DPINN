# ## DEFINICIÓN DEL PROBLEMA

# Una función para generar redes (la estructura `SolNN` nos da flexibilidad para usar una arquitectura distinta para cada parte).
function create_nn(activation, hidden, depth)
    return Chain(
        Dense(2 => hidden, activation),
        (Dense(hidden=>hidden,activation) for _ in 1:depth-2)...,
        Dense(hidden => 1)
    )
end

# Definimos una estructura para almacenar las 3 redes: una para u y una para cada componente de φ.
struct SolNN{V,F1,F2} <: Lux.AbstractLuxContainerLayer{(:v, :φ₁, :φ₂)}
    v::V
    φ₁::F1
    φ₂::F2
end

# Definimos un constructor genérico que admite una arquitectura totalmente distinta para cada componente.
"""
    SolNN(strv::NamedTuple,strφ₁::NamedTuple,strφ₂::NamedTuple)
Permite construir un conjunto de redes a partir de `NamedTuple`s indicando los parámetros.
```julia
julia> strv=(;activation=σ,hidden=15,depth=5)
julia> strφ₁=(;activation=tanh,hidden=10,depth=4)
julia> strφ₂=(;activation=tanh,hidden=10,depth=4))
julia> sol = SolNN(stru,strφ₁,strφ₂)
```
Se admite una versión simplificada cuando φ₁ y φ₂ tengan la misma estructura y otra cuando todas las funciones tengan la misma estructura.

El ejemplo anterior es equivalente a:
```julia
julia> sol = SolNN(strv,strφ₁)
```

En cambio:
```julia
julia> sol = SolNN(strv)
```
Crea una `SolNN` con la misma estructura para `v`, `φ₁` y `φ₂`.
"""
function SolNN(strv::NamedTuple,strφ₁::NamedTuple,strφ₂::NamedTuple)
    return SolNN(
        create_nn(strv.activation, strv.hidden, strv.depth),
        create_nn(strφ₁.activation,strφ₁.hidden,strφ₁.depth),
        create_nn(strφ₂.activation,strφ₂.hidden,strφ₂.depth)
    )
end
SolNN(strv::NamedTuple,strφ::NamedTuple) = SolNN(strv,strφ,strφ)
SolNN(str::NamedTuple) = SolNN(str,str,str)

## Definimos una estructura para almacenar la distancia y sus derivadas
struct Distance{D,Dx,Dy}
    d::D
    dx::Dx
    dy::Dy
end

#Definimos cómo evaluar la función distancia (y sus derivadas en una)
function (::Distance)(x::AbstractVector)
    return (d(x),dx(x),dy(y))
end

function (dist::Distance)(x::AbstractMatrix)
    (;d,dx,dy) = dist
    return (d(x),dx(x),dy(x))
end

# Definimos una estructura para almacenar la información general del problema 
# struct ProblemData{F<:Function,G<:Function,D::Distance,M<:AbstractMatrix,B<:AbstractFloat}
#     gendata::F
#     f::G
#     dist::D
#     A::M
#     b::B
# end

struct ProblemData{G<:Function,F<:Function,D<:Distance,T<:AbstractMatrix,B<:AbstractFloat,E<:Function}
    gendata::G
    f::F
    dist::D
    A::T
    b::B
    exact::Union{E,Nothing}
end

#Un constructor para que se asigne A=I y b = [0,0] si no son especificados. 
function ProblemData(gendata,f,d::Distance)
    return ProblemData(gendata,f,d,I(2),0.0f0,nothing)
end
function ProblemData(gendata,f,d::Distance,exact::E) where E<:Function
    return ProblemData(gendata,f,d,I(2),0.0f0,exact)
end
