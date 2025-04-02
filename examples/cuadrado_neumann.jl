using CairoMakie, DPINN, Lux

### Resolvemos el problema -Δu = -4 con condiciones de borde Dirichlet homogéneas en el disco unitario. Definimos paso a paso todos los elementos para hacer uso de nuestro módulo general.

# # DEFINICIÓN DEL PROBLEMA
# SAMPLEO de puntos (acá entra el dominio)

# esta función asume que la matriz de puntos ya fue generada. 
function gen_data!(puntos)
    for v in eachcol(puntos)
        v .= rand(eltype(puntos),2)
    end    
end

# DISTANCIA
# La distancia y sus derivadas por separado. La sintaxis peculiar es para:
#  1. garantizar estabilidad de tipos (Lux suele trabajar por defecto con Float32, mientras que Julia en general usa Float64, es importante evitar mezclarlos para evitar un deterioro de la performance)
#  2. evitar el indexado con escalares (por eso el slicing: x[1:1,:]), que da un error al operar sobre xdev.
# Con estos elementos creamos un objeto de tipo Distance. 
dD(x)  = x[1:1,:] .*(x[2:2,:] .- x[2:2,:].^2)
dN(x)  = 1.0f0 .- x[1:1,:]
dist = Distance(dD,dN)

# multiplos:
const Π = Float32(π)
A = Float32.([1 -0.5;-0.5 1])
b = 0.0f0

# DATO
# De nuevo, la sintaxis está para estabilidad de tipos y manejo de arrays de Reactant.
f(x) = Π*cos.(Π*x[2:2,:]) + Π^2*x[1:1,:].*sin.(Π*x[2:2,:])

#Boundary Data
gD(x) = zero.(x[1:1,:])
η(x) = reduce(vcat,(one.(x[1:1,:]),zero.(x[2:2,:])))
gN(x) = sin.(Π*x[2:2,:])
bd = BoundaryData(gD,gN,η)
# SOLUCIÓN EXACTA
U(x) = x[1:1,:].*sin.(Π*x[2:2,:])

# PROBLEMA
problem_data = ProblemData(2,gen_data!,f,dist,bd,A,b,U)
# alternativa sin solución exacta (no computa errores)
#problem_data = ProblemData(gen_data!,f,dist)


# # ENTRENAMIENTO DE LA RED
# Estructura de la red:
structure = (;N=2,activation=sigmoid_fast,hidden=12,depth=4)
@crear_clase SolCuadrado 2
model = SolCuadrado(structure)

# Entrenamos el modelo:
n_points = 10000

trained_model,losses,errors = train_model(model,n_points,problem_data;bs=n_points÷2, maxiters=10000)

# recuperamos la componente v.
trained_v = Lux.testmode(
    StatefulLuxLayer{true}(trained_model.model.v, trained_model.ps.v, trained_model.st.v)
)

# Construimos la solución u a partir de v y d:
u(x) = gD(x)[1] + dD(x)[1]*trained_v(x)[1]

# # GRAFICOS
# Graficamos en una misma figura 6 cosas: la solución aproximada y la exacta en una especie de heatmap, con su correspondiente Colorbar, y a la derecha la evolución de la loss y del error respecto de la solución exacta.
#  Notar que el error se calcula en base a los mismos puntos sampleados que se usan para la loss. No es un cálculo muuuy preciso, pero da una idea. 
fig = Figure()
ax11 = Axis(fig[1, 1], title = "Solución")
xx = 0:0.05f0:1
yy = 0:0.05f0:1
cs = [u([xxx,yyy]) for xxx in xx, yyy in yy]
p1 = surface!(ax11, 0..1, 0..1, zeros(size(cs)), color = cs, shading = NoShading, colormap = :coolwarm)
#ax.gridz[] = 100
tightlimits!(ax11) # surface plots include padding by default
hidedecorations!(ax11)
hidespines!(ax11)
Colorbar(fig[1,2], p1, flipaxis = false)
ax21 = Axis(fig[2,1],title = "Exacta")
ce = [U([xxx,yyy])[1] for xxx in xx, yyy in yy]
p2 = surface!(ax21, 0..1, 0..1, zeros(size(cs)), color = ce, shading = NoShading, colormap = :coolwarm)
Colorbar(fig[2,2], p2, flipaxis = false)
tightlimits!(ax21) # surface plots include padding by default
hidedecorations!(ax21)
hidespines!(ax21)

ax13 = Axis(fig[1,3],title= "Loss")
lines!(ax13,500:length(losses),losses[500:end])

ax23 = Axis(fig[2,3],title = "Error (vs Sol Exacta)")
lines!(ax23,500:length(errors),errors[500:end])
fig
