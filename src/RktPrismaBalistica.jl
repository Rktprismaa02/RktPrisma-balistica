"""
    RktPrismaBalistica

Núcleo de balística interna do RktPrisma usado nos Capítulos 5 e 6 do PFC:
geometria do grão (regressão por dilatação de Minkowski), modelo 0D de
parâmetros concentrados, modelo quasi-unidimensional (1D) com queima erosiva
e erosão da garganta, e correção de duas fases.

```julia
using RktPrismaBalistica
res = simular_caso(inp; cfg = ConfigModelo(modo_simulacao = :zero_d))
```
"""
module RktPrismaBalistica

using StaticArrays
using Printf
using DelimitedFiles
using Statistics

include("types.jl")
include("ModuloGeometria.jl")    # sub-módulo: geometria do grão (LibGEOS)
include("geometry.jl")           # malha 1D e geometria local câmara/tubeira
include("Nozzle.jl")             # garganta: erosão, η_bl, C_F
include("physics.jl")            # termos-fonte, regressão, queima erosiva
include("Solver1D.jl")           # fluxos HLLC, MUSCL, termo de área bem balanceado
include("Metrics.jl")
include("ChamberProfile.jl")     # perfis axiais e regressão do grão (pós-processamento)
include("SimulationCore.jl")     # laço temporal do modelo 1D
include("Termoquimica.jl")       # Γ(γ), c*, Pr e μ do gás
include("DuasFases.jl")          # η_2ph: Hermsen + Stokes
include("GeometriaTubeira.jl")   # contorno da tubeira p/ o modelo de duas fases acoplado
include("TwoPhaseNozzle.jl")     # duas fases acoplado (opcional: usar_2fases_acoplado)
include("ErosiveAnalysis.jl")    # gráficos da queima erosiva (pós-processamento)
include("Solver0D.jl")           # modelo 0D (quasi-estático e com enchimento)
include("CaseRunner.jl")         # CaseInput → geometria → simulação → correções

export CaseInput, GrainSpec, ConfigModelo, SimulationResult
export ModuloGeometria
export simular_caso, simular_0d, diagnostico_geometria
export build_case_geometry, build_propellant_from_case, validar_case_input
export analisar_erosao_espacial, plotar_analise_erosiva
export calcular_d43_hermsen, calcular_d43_hermsen_corrigido

end # module
