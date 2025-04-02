using CairoMakie, DPINN, Lux

### Resolvemos el problema -Δu = -4 con condiciones de borde Dirichlet homogéneas en el disco unitario. Definimos paso a paso todos los elementos para hacer uso de nuestro módulo general.

# # DEFINICIÓN DEL PROBLEMA
# SAMPLEO de puntos (acá entra el dominio)

# esta función asume que la matriz de puntos ya fue generada. 
function cuadr() 
    x = rand(Float32,2)   
    while all(x.>0.5f0)
        x .= rand(Float32,2)
    end
    return x   
end
function gen_data!(puntos)
    for v in eachcol(puntos)
        v .= cuadr()
    end    
end

# DISTANCIA
# La distancia y sus derivadas por separado. La sintaxis peculiar es para:
#  1. garantizar estabilidad de tipos (Lux suele trabajar por defecto con Float32, mientras que Julia en general usa Float64, es importante evitar mezclarlos para evitar un deterioro de la performance)
#  2. evitar el indexado con escalares (por eso el slicing: x[1:1,:]), que da un error al operar sobre xdev.
# Con estos elementos creamos un objeto de tipo Distance. 


RectInf(x) = (x[1:1,:] .> 0.5f0).*(x[2:2,:].<0.5f0)
RectSup(x) = (x[2:2,:] .> 0.5f0).*(x[1:1,:].<0.5f0)
Quad(x) = (x[1:1,:] .< 0.5f0).*(x[2:2,:] .< 0.5f0)
	
function d(x)
    RectInf(x).*(x[2:2,:].*(0.5f0one(eltype(x)).-x[2:2,:]).*(one(eltype(x)).-x[1:1,:])) + RectSup(x).*(x[1:1,:].*(0.5f0one(eltype(x)).-x[1:1,:]).*(one(eltype(x)).-x[2:2,:])) + Quad(x).*(x[1:1,:].*x[2:2,:].*(one(eltype(x)).-x[1:1,:].-x[2:2,:]))
end

#∇d(x) = [(one(eltype(x)) .- 2x[1:1,:]).*(x[2:2,:] .- x[2:2,:].^2);
#         (one(eltype(x)) .- 2x[2:2,:]).*(x[1:1,:] .- x[1:1,:].^2)]


dist = Distance(d,nothing)


# multiplos:
const Π = Float32(π)
A = Float32.([1 2;0 1])
b = -8Π^2

# DATO
# De nuevo, la sintaxis está para estabilidad de tipos y manejo de arrays de Reactant.
f(x) = -8Π^2*cos.(2*Π*x[1:1,:]).*cos.(2*Π*x[2:2,:]) 

#Boundary Data
bd = BoundaryData(nothing,nothing,nothing)
# SOLUCIÓN EXACTA
U(x) = sin.(2*Π*x[1:1,:]).*sin.(2*Π*x[2:2,:]).*(d(x).>=0.0f0)

# PROBLEMA

problem_data = ProblemData(2,gen_data!,f,dist,bd,A,b,U)



# # ENTRENAMIENTO DE LA RED
# Estructura de la red:
structure = (;N=2,activation=sigmoid_fast,hidden=15,depth=5)

@crear_clase SolL 2
model = SolL(structure)

# Entrenamos el modelo:
n_points = 5000
trained_model,losses,errors = train_model(model,n_points,problem_data;maxiters=10000)

# recuperamos la componente v.
trained_v = Lux.testmode(
    StatefulLuxLayer{true}(trained_model.model.v, trained_model.ps.v, trained_model.st.v)
)

# Construimos la solución u a partir de v y d:
u(x) = all(x.>0.5) ? 0.0f0 : d(x)[1]*trained_v(x)[1]

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
#save("Plot_Nuevo_cuadrado2.png",fig)

#print("el loss es " ,losses[end])
#print("el error es ", errors[end])

#Primera tirada del nuevo: el loss es 8.214351e-5 y el error es 7.9356305e-6, 1134.271594 seconds (750.79 M allocations: 34.870 GiB, 1.05% gc time, 18.45% compilation time: 21% of which was recompilation)
# Con 2500 puntos 10000 iters 15 ocultas y 5 de profundidad.

#Segunda tirada el loss es 0.00015872286el error es 7.5548996e-6. Tardó 774.893957 seconds.
# Con 1000 puntos 10k iters y 15 ocultas y 5 prof.