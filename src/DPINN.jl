module DPINN

# # Pequeño módulo para resolver problemas según nuestro modelo general usando Enzyme (para autodiferenciación) + Reactant (para compilación). El enfoque sigue el siguiente ejemplo: https://github.com/LuxDL/Lux.jl/blob/main/examples/PINN2DPDE/main.jl, que viene de la documentación de Lux. 
#
# El ejemplo resuelve la ecuación del calor. La filosofía es similar a la nuestra: se descompone la ecuación como un sistema de orden 1, tomando V = ∂U_∂x, W = ∂U_∂y. La principal diferencia es que en el ejemplo los datos se contorno se ajustan y por lo tanto participan en la función de pérdida. 
#
# Otra diferencia importante es que no define una red de R²→R³, sino que crea una estructura que contiene tres redes R²→R, (U,V y W). Eso está piola, así que lo tomo. Usé la misma filosofía para encapsular la función distancia y sus derivadas. 

# ## Imports

using Lux, Optimisers, Random, Statistics, MLUtils,LinearAlgebra, Reactant, Enzyme

# devices
const xdev = reactant_device(; force=true)
const cdev = cpu_device()

# código
include("definitions.jl")
include("training.jl")

# exports

export Distance, ProblemData, @crear_clase
export train_model

end # module DPINN
