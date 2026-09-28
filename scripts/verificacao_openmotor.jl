# ==============================================================================
# verificacao_openmotor.jl — verificação do modelo 0D contra o OpenMotor
# ==============================================================================
# Reproduz a Seção 5.1.4 do PFC (Figuras 9 e 10, Tabela 9).
#   Caso 1: três grãos BATES Ø100 × 200 mm, canal 40 mm, sem faces inibidas.
#   Caso 2: Finocyl Ø100 × 600 mm, canal 40 mm, 7 aletas de 10 × 20 mm nos
#           180 mm junto à tubeira, as duas faces inibidas.
# Ambos: propelente da Tabela 8, garganta 25 mm, ε = 4 e η_c* = 1. Eficiência de
# tubeira do RktPrisma: 0,95 no Caso 1 e 1,0 no Caso 2, como nas rodadas da
# Tabela 9. Nenhum parâmetro ajustado.
#
# Uso:   julia --project=. scripts/verificacao_openmotor.jl
# Referências do OpenMotor em dados/caso1_openmotor.csv e dados/caso2_openmotor.csv.
# ==============================================================================
using RktPrismaBalistica, Printf, DelimitedFiles
include(joinpath(@__DIR__, "CompararCSV.jl"))

const DADOS = joinpath(@__DIR__, "..", "dados")
cd(mkpath(joinpath(@__DIR__, "..", "resultados")))

prop = (rho_p = 1700.0, a = 9.0e-6, n = 0.412, Tc = 2977.0, gamma = 1.198, R = 332.82,
        eta_cstar = 1.0, frac_alumina = 0.12)
tubeira = (D_garganta_ini = 0.025, D_saida = 0.050, alpha_divergencia = 15.0)
cfg = ConfigModelo(modo_simulacao = :zero_d, modo_silencioso = true)

casos = [
    ("caso1_bates", "caso1_openmotor.csv",
     CaseInput(; name = "caso1_bates", geometry_type = :bates,
               D_ext = 0.100, D_core = 0.040, L_grao = 0.200, N_graos = 3, inhibited_ends = 0,
               x_garganta = 0.65, L_total = 0.75, N_malha = 100, t_maximo = 20.0,
               eta_tubeira = 0.95, tubeira..., prop...)),
    ("caso2_finocyl", "caso2_openmotor.csv",
     CaseInput(; name = "caso2_finocyl", geometry_type = :finocyl, slot_shape = :rectangular,
               D_ext = 0.100, D_core = 0.040, L_grao = 0.600, N_graos = 1, inhibited_ends = 2,
               n_fins = 7, fin_width = 0.010, fin_length = 0.020, fin_fraction = 0.30,
               finocyl_no_bocal = true,
               x_garganta = 0.65, L_total = 0.75, N_malha = 100, t_maximo = 20.0,
               eta_tubeira = 1.0, tubeira..., prop...)),
]

for (nome, ref, inp) in casos
    res = simular_caso(inp; cfg = cfg, salvar_csv = true)   # grava resultados_0d_<nome>.csv
    kn_om = readdlm(joinpath(DADOS, ref), ',', Float64; skipstart = 1)[1, 2]
    @printf("\n%s — Kn(0): OpenMotor %.2f | RktPrisma %.2f (%+.2f %%)\n",
            nome, kn_om, res.Kn_ini, 100 * (res.Kn_ini / kn_om - 1))
    comparar_curvas(joinpath(DADOS, ref), "resultados_0d_$(nome).csv";
                    rotulo_ref = "OpenMotor", rotulo_sim = "RktPrisma",
                    salvar = "verificacao_$(nome).png",
                    titulo = nome == "caso1_bates" ? "Caso 1 — BATES" : "Caso 2 — Finocyl")
end
