using CairoMakie, DPINN, Lux

### Resolvemos el problema -Δu = -4 con condiciones de borde Dirichlet homogéneas en el disco unitario. Definimos paso a paso todos los elementos para hacer uso de nuestro módulo general.

# # DEFINICIÓN DEL PROBLEMA
# SAMPLEO de puntos (acá entra el dominio)
function circ_point() 
    x = [2rand(Float32)-1,2rand(Float32)-1]    
    while sum(x.^2)>1
        x = [2rand(Float32)-1,2rand(Float32)-1] 
    end
    return x   
end

# esta función asume que la matriz de puntos ya fue generada. 
function gen_data!(puntos)
    for v in eachcol(puntos)
        v .= circ_point()
    end    
end

# DISTANCIA
# La distancia y sus derivadas por separado. La sintaxis peculiar es para:
#  1. garantizar estabilidad de tipos (Lux suele trabajar por defecto con Float32, mientras que Julia en general usa Float64, es importante evitar mezclarlos para evitar un deterioro de la performance)
#  2. evitar el indexado con escalares (por eso el slicing: x[1:1,:]), que da un error al operar sobre xdev.
# Con estos elementos creamos un objeto de tipo Distance. 
d(x)  = one(eltype(x)) .- sum(x.^2,dims=1)
dx(x) = -2x[1:1,:]
dy(x) = -2x[2:2,:]
dist = Distance(d,dx,dy)

# DATO
# De nuevo, la sintaxis está para estabilidad de tipos y manejo de arrays de Reactant. 
f(x) = -4.0f0*one.(x[1:1,:]) 

# SOLUCIÓN EXACTA
U(x) = one(eltype(x)) .- sum(x.^2,dims=1)

# PROBLEMA
problem_data = ProblemData(gen_data!,f,dist,U)

# # ENTRENAMIENTO DE LA RED
# Estructura de la red:
structure = (;activation=σ,hidden=15,depth=5)

# Entrenamos el modelo:
trained_model,losses,errors = train_model(1000,structure,problem_data)

# recuperamos la componente v.
trained_v = Lux.testmode(
    StatefulLuxLayer{true}(trained_model.model.v, trained_model.ps.v, trained_model.st.v)
)

# Construimos la solución u a partir de v y d:
u(x) = d(x)[1]*trained_v(x)[1]

# # GRAFICOS
# Graficamos en una misma figura 6 cosas: la solución aproximada y la exacta en una especie de heatmap, con su correspondiente Colorbar, y a la derecha la evolución de la loss y del error respecto de la solución exacta.
#  Notar que el error se calcula en base a los mismos puntos sampleados que se usan para la loss. No es un cálculo muuuy preciso, pero da una idea. 
fig = Figure()
ax11 = PolarAxis(fig[1, 1], title = "Solución")
rs = 0:0.05f0:1
Θ = range(0.0f0, 2Float32(pi), 67)
cs = [u([r*cos(θ),r*sin(θ)]) for θ in Θ, r in rs]
p1 = surface!(ax11, 0..2pi, 0..1, zeros(size(cs)), color = cs, shading = NoShading, colormap = :coolwarm)
#ax.gridz[] = 100
tightlimits!(ax11) # surface plots include padding by default
hidedecorations!(ax11)
hidespines!(ax11)
Colorbar(fig[1,2], p1, flipaxis = false)
ax21 = PolarAxis(fig[2,1],title = "Exacta")
ce = [u([r*cos(θ),r*sin(θ)]) for θ in Θ, r in rs]
p2 = surface!(ax21, 0..2pi, 0..1, zeros(size(cs)), color = ce, shading = NoShading, colormap = :coolwarm)
Colorbar(fig[2,2], p2, flipaxis = false)
tightlimits!(ax21) # surface plots include padding by default
hidedecorations!(ax21)
hidespines!(ax21)

ax13 = Axis(fig[1,3],title= "Loss")
lines!(ax13,500:length(losses),losses[500:end])

ax23 = Axis(fig[2,3],title = "Error (vs Sol Exacta)")
lines!(ax23,500:length(errors),errors[500:end])
fig
