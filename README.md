# RktPrisma — balística interna

Código de balística interna usado no Projeto de Fim de Curso **"Projeto de um motor-foguete a
propelente sólido"** (IME), Capítulos 5 e 6. Este repositório contém apenas o necessário para
reproduzir as análises balísticas do trabalho:

- geometria do grão, com regressão da superfície por dilatação de Minkowski (LibGEOS);
- modelo 0D de parâmetros concentrados (Seção 5.1.2 e Apêndice B);
- modelo quasi-unidimensional (1D), com queima erosiva de Mukunda e Paul e erosão da garganta
  (Seção 6.1 e Apêndice B);
- correção de duas fases para as partículas de Al₂O₃;
- verificação contra o OpenMotor e varredura da geometria definitiva.

As análises térmica e estrutural do trabalho e a interface gráfica do RktPrisma não fazem parte
deste repositório.

## Instalação

Requer [Julia](https://julialang.org/downloads/) 1.10 ou mais recente (testado na 1.12.5).
Na pasta do repositório:

```
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

## O que cada script reproduz

| Script | Seção do PFC | Tempo aproximado |
|---|---|---|
| `scripts/verificacao_openmotor.jl` | 5.1.4 — verificação contra o OpenMotor (Figuras 9 e 10, Tabela 9) | 1 min |
| `scripts/varredura.jl` | 6.1.5 — geometria finocyl definitiva (Equação 6.7) | 1 h |
| `scripts/projeto_final.jl` | 6.1.6 — desempenho do projeto final, 0D e 1D a −10, 25 e +50 °C | 1,5 h (0D: segundos) |

Exemplo:

```
julia --project=. scripts/verificacao_openmotor.jl
julia --project=. scripts/projeto_final.jl 0d
```

Os resultados (CSV e gráficos) são gravados em `resultados/`. As curvas de referência do
OpenMotor estão em `dados/`.

## Organização do código (`src/`)

| Arquivo | Conteúdo |
|---|---|
| `types.jl` | estruturas de entrada (`CaseInput`), configuração (`ConfigModelo`) e resultado |
| `ModuloGeometria.jl`, `geometry.jl` | seção do grão, regressão e geometria ao longo do eixo |
| `Solver0D.jl` | modelo 0D quasi-estático e com enchimento de câmara |
| `physics.jl` | termos-fonte, taxa de regressão local e queima erosiva |
| `Solver1D.jl` | fluxos HLLC, reconstrução MUSCL e termo de área bem balanceado |
| `SimulationCore.jl` | laço temporal do modelo 1D |
| `Nozzle.jl` | erosão da garganta e perda de camada-limite |
| `Termoquimica.jl`, `DuasFases.jl`, `GeometriaTubeira.jl`, `TwoPhaseNozzle.jl` | c\*, correção de duas fases (escalar e acoplada à tubeira) |
| `ChamberProfile.jl`, `ErosiveAnalysis.jl`, `Metrics.jl` | perfis axiais, gráficos da queima erosiva e métricas |
| `CaseRunner.jl` | monta o caso a partir de `CaseInput`, executa o modelo e aplica as correções |

## Uso direto

```julia
using RktPrismaBalistica

inp = CaseInput(name = "exemplo", geometry_type = :bates,
                D_ext = 0.100, D_core = 0.040, L_grao = 0.200, N_graos = 3, inhibited_ends = 0,
                rho_p = 1700.0, a = 9.0e-6, n = 0.412, Tc = 2977.0, gamma = 1.198, R = 332.82,
                x_garganta = 0.65, D_garganta_ini = 0.025, D_saida = 0.050, L_total = 0.75,
                N_malha = 100, t_maximo = 20.0)

r0 = simular_caso(inp; cfg = ConfigModelo(modo_simulacao = :zero_d))   # modelo 0D
r1 = simular_caso(inp; cfg = ConfigModelo(modo_simulacao = :um_d))     # modelo 1D
println(r0.P_max, " MPa | ", r1.P_max, " MPa")
```

Unidades SI nas entradas (m, kg, Pa, K). O coeficiente `a` da lei de Saint-Robert é dado em
m/(s·Paⁿ).

## Limitações

O código foi **verificado** (contra o OpenMotor, soluções analíticas e uma solução
quasi-unidimensional independente), mas não **validado** contra ensaios de queima: o par
balístico (a, n) do propelente vem da literatura. Os resultados são adequados ao projeto
preliminar e não substituem ensaios.
