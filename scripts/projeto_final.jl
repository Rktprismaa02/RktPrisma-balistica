# ==============================================================================
# projeto_final.jl — balística do projeto definitivo (Capítulo 6)
# ==============================================================================
# Simula o grão Finocyl selecionado (Tabela 16 do PFC) pelos modelos 0D e 1D
# (quasi-unidimensional, com queima erosiva de Mukunda e erosão da garganta)
# nas temperaturas do grão de −10, 25 e +50 °C.
#
# Uso:   julia --project=. scripts/projeto_final.jl          (0D e 1D)
#        julia --project=. scripts/projeto_final.jl 0d       (só 0D, segundos)
#
# O 1D leva da ordem de 20 a 30 min por temperatura (N = 100). Saídas em
# resultados/: CSVs de histórico e de perfis axiais, gráficos e o resumo impresso.
# ==============================================================================
using RktPrismaBalistica, Printf

const SO_0D = !isempty(ARGS) && lowercase(ARGS[1]) == "0d"
cd(mkpath(joinpath(@__DIR__, "..", "resultados")))

# Duas fases: false = η_2ph escalar (Hermsen + Stokes); true = acoplado à tubeira.
# Os resultados do Capítulo 6 foram obtidos com o modelo acoplado.
const DUAS_FASES_ACOPLADO = true

grao_final(T_C) = CaseInput(; name = @sprintf("final_T%+d", T_C),
    geometry_type = :finocyl, slot_shape = :rectangular,
    D_ext = 0.540, L_grao = 2.600, N_graos = 1, D_core = 0.150,
    n_fins = 7, fin_width = 0.035, fin_length = 0.105,
    fin_fraction = 0.850 / 2.600, finocyl_no_bocal = true,     # trecho aletado junto à tubeira
    inhibited_ends = 1,                                          # face traseira inibida
    fin_taper_zone_length = 0.240, fin_taper_n_segs = 40, fin_taper_start_frac = 0.0,
    rho_p = 1700.0, a = 9.0e-6, n = 0.412, Tc = 2977.0, gamma = 1.198, R = 332.82,
    eta_cstar = 0.98, frac_alumina = 0.12,
    sigma_p = 0.002, T_ref = 298.15, T_grain = 273.15 + T_C,
    x_garganta = 2.873, D_garganta_ini = 0.094, D_saida = 0.310, alpha_divergencia = 15.0,
    L_total = 3.2761, N_malha = 100, t_maximo = 45.0,
    erosao_ativa = true, erosao_r_dot_ref = 0.09, erosao_P_ref_MPa = 5.0, erosao_n_exp = 0.8)

cfg(modo) = ConfigModelo(modo_simulacao = modo, modo_silencioso = true,
                         usar_erosiva = true, modelo_erosivo = :mukunda,
                         usar_erosao_garganta = true,
                         usar_2fases_acoplado = DUAS_FASES_ACOPLADO)

linha(rot, r) = @printf("%-10s %8.3f %8.3f %8.1f %9.3f %8.1f %8.3f %8.1f\n", rot,
    r.P_max, r.P_avg, r.t_burn, r.I_total_corrigido / 1e6, r.Isp_corrigido,
    r.CF_avg, r.D_garganta_final)

resultados = Tuple{String, Any}[]
for T_C in (-10, 25, 50)
    push!(resultados, (@sprintf("0D %+d°C", T_C), simular_caso(grao_final(T_C); cfg = cfg(:zero_d), salvar_csv = true)))
    if !SO_0D
        t = @elapsed r1 = simular_caso(grao_final(T_C); cfg = cfg(:um_d), salvar_csv = true)
        @printf("1D %+d °C concluído em %.1f min\n", T_C, t / 60); flush(stdout)
        push!(resultados, (@sprintf("1D %+d°C", T_C), r1))
    end
end

println("\nP_max: 0D = pressão de equilíbrio; 1D = pressão estática na cabeça do motor")
@printf("%-10s %8s %8s %8s %9s %8s %8s %8s\n", "caso", "Pmax", "Pmed", "tq[s]",
        "I[MN·s]", "Isp[s]", "CF", "Dt[mm]")
for (rot, r) in resultados
    linha(rot, r)
end
