# ==============================================================================
# varredura.jl — varredura exaustiva 0D da geometria do grão Finocyl definitivo
# ==============================================================================
# Reproduz a varredura da Seção 6.1.5 do PFC: chanfro das aletas, erosão da
# garganta, η_c* = 0,98, restrições de viabilidade e escore da Equação 6.7.
#
# Uso:   julia --project=. scripts/varredura.jl
#
# Cada combinação da grade passa pelo módulo geométrico e pelo modelo 0D; as que
# violam alguma restrição são descartadas e as viáveis são ordenadas pelo escore.
# Saídas em resultados/: CSV com todas as configurações válidas (separador ; e
# vírgula decimal) e o top-N impresso no terminal.
# ==============================================================================
using RktPrismaBalistica, Printf, Dates

# ── Grade: (mínimo, máximo, passo) em mm; passo 0 = valor único ─────────────
L_total   = (2600.0, 2600.0,   0.0)
L_finocyl = ( 850.0,  900.0,  25.0)
D_core    = ( 140.0,  160.0,   5.0)
n_fins    = (   7.0,    9.0,   1.0)
fin_width = (  35.0,   50.0,   5.0)
fin_len   = ( 105.0,  115.0,   5.0)
throat    = (  94.0,   96.0,   1.0)
fin_taper = ( 240.0,  240.0,   0.0)              # zona de chanfro das aletas [mm]

# ── Parâmetros fixos e propelente (Tabela 8) ─────────────────────────────────
const D_ext   = 540.0                            # [mm]
const D_saida = 310.0                            # [mm]
const alpha   = 15.0                             # meia-abertura do divergente [°]
const inh     = 1                                # face traseira (junto à tubeira) inibida
const prop = (rho_p = 1700.0, a = 9.0e-6, n = 0.412, Tc = 2977.0, gamma = 1.198,
              R = 332.82, frac_alumina = 0.12, eta_cstar = 0.98)

# ── Erosão da garganta: ṙ = ṙ_ref·(P₀/P_ref)^n_t ─────────────────────────────
const EROSAO_R_DOT_REF = 0.09                    # [mm/s]
const EROSAO_P_REF_MPA = 5.0                     # [MPa]
const EROSAO_N_EXP     = 0.8

# ── Restrições ───────────────────────────────────────────────────────────────
const I_ALVO        = 2_000_000.0                # [N·s]
const TOL_ALVO      = 0.05                       # ±5 %
const P_MAX_LIMITE  = 7.0                        # [MPa] MEOP
const P_MIN_LIMITE  = 3.5                        # [MPa] vale entre as fases
const P_SUSTAIN_MAX = 4.6                        # [MPa] máximo da sustentação
const M_PROP_MAX    = 1000.0                     # [kg]
# e P_boost > P_sustain (pico global na fase aletada)

# ── Escore (Eq. 6.7) ─────────────────────────────────────────────────────────
# escore = M_ref/m_p − λ_subida·f_subida − λ_queda·max(0, f_queda − f_alvo)
#          − λ_steep·|dP/dt|_queda/P_boost
const M_REF      = 1000.0                        # [kg]
const W_SUBIDA   = 0.50
const W_QUEDA    = 0.40
const QUEDA_ALVO = 0.40
const W_STEEP    = 0.15                          # [s]

const TOP_N   = 20
const CSV_OUT = "varredura_definitiva.csv"
# ──────────────────────────────────────────────────────────────────────────────

const CFG0 = ConfigModelo(modo_simulacao = :zero_d, modo_silencioso = true,
                          usar_erosao_garganta = true)

function _metricas(Lt, Lf, Dc, nf, fw, fl, th, ftap)
    ff    = clamp(Lf / Lt, 0.01, 1.0)            # fração aletada (junto à tubeira)
    Ltm   = Lt / 1e3
    tzone = ftap / 1e3
    inp = CaseInput(; name = "v", geometry_type = :finocyl, slot_shape = :rectangular,
        D_ext = D_ext/1e3, D_core = Dc/1e3, L_grao = Ltm, N_graos = 1, inhibited_ends = inh,
        n_fins = nf, fin_width = fw/1e3, fin_length = fl/1e3, fin_fraction = ff, L_trans_frac = 0.0,
        fin_taper_zone_length = tzone,
        fin_taper_start_frac  = 0.0,
        fin_taper_n_segs      = (tzone > 1e-4 ? 8 : 1),
        x_garganta = Ltm + 0.2, D_garganta_ini = th/1e3, D_saida = D_saida/1e3,
        alpha_divergencia = alpha, L_total = Ltm + 0.5, N_malha = 80, t_maximo = 150.0,
        erosao_ativa = true, erosao_r_dot_ref = EROSAO_R_DOT_REF,
        erosao_P_ref_MPa = EROSAO_P_REF_MPA, erosao_n_exp = EROSAO_N_EXP,
        prop...)
    res = redirect_stdout(devnull) do
        simular_caso(inp; cfg = CFG0, salvar_csv = false)
    end
    res === nothing && return nothing
    # ── pressões características na janela de queima ativa (F > 10 % F_max) ──
    t = res.tempos; P = res.pressoes; F = res.empuxos
    idx = findall(f -> f > 0.10 * maximum(F), F)
    isempty(idx) && return nothing
    i1, i2 = first(idx), last(idx)
    ib = i1 - 1 + argmax(@view P[i1:i2])                 # pico do boost (máx. global)
    ir = i2
    while ir > ib + 1 && P[ir-1] > P[ir]; ir -= 1; end    # pico da sustentação
    iv = ib - 1 + argmin(@view P[ib:ir])                 # vale entre os dois
    P_boost, P_sust, P_min = P[ib], P[ir], P[iv]
    queda_frac  = max(P_boost - P_min, 0.0) / P_boost    # f_queda
    subida_frac = max(P_sust  - P_min, 0.0) / P_min      # f_subida
    dPdt_queda  = (P_min - P_boost) / max(t[iv] - t[ib], 1e-6)
    return (P_max = res.P_max, P_boost = P_boost, P_sust = P_sust, P_min = P_min,
            t_pico = t[ib], queda_frac = queda_frac, subida_frac = subida_frac,
            dPdt_queda = dPdt_queda, D_gt_fin = res.D_garganta_final,
            Isp = res.Isp_corrigido, I_total = res.I_total_corrigido,
            t_burn = res.t_burn, m_prop = res.m_consumida)
end

_score(r) = M_REF / r.m_prop - W_SUBIDA * r.subida_frac -
            W_QUEDA * max(0.0, r.queda_frac - QUEDA_ALVO) -
            W_STEEP * abs(r.dPdt_queda) / max(r.P_boost, 0.1)

_viavel(r) = r.P_max <= P_MAX_LIMITE && r.P_min >= P_MIN_LIMITE &&
             abs(r.I_total - I_ALVO) <= TOL_ALVO * I_ALVO &&
             r.P_boost > r.P_sust && r.P_sust <= P_SUSTAIN_MAX && r.m_prop <= M_PROP_MAX

_faixa(v) = (v[3] <= 0 || v[1] >= v[2]) ? [v[1]] : collect(v[1]:v[3]:v[2])
combos = [(Lt, Lf, Dc, nf, fw, fl, th, ft)
          for Lt in _faixa(L_total) for Lf in _faixa(L_finocyl) for Dc in _faixa(D_core)
          for nf in _faixa(n_fins)  for fw in _faixa(fin_width) for fl in _faixa(fin_len)
          for th in _faixa(throat)  for ft in _faixa(fin_taper)]
@printf("Varredura: %d combinações — início %s\n", length(combos), Dates.format(now(), "HH:MM:SS"))
flush(stdout)

linhas = NamedTuple[]
n_inv = 0; t0 = time()
for (k, c) in enumerate(combos)
    m = try _metricas(c...) catch; nothing end            # geometria inválida → descarta
    m === nothing ? (global n_inv += 1) :
        push!(linhas, (Lt=c[1], Lf=c[2], Dc=c[3], nf=c[4], fw=c[5], fl=c[6], th=c[7], ft=c[8], m...))
    k % 100 == 0 && (@printf("  %d/%d (%.0f s)\n", k, length(combos), time() - t0); flush(stdout))
end
viaveis = sort!(filter(_viavel, linhas); by = _score, rev = true)
@printf("Concluído em %.0f s: %d válidas, %d inválidas, %d viáveis\n",
        time() - t0, length(linhas), n_inv, length(viaveis))

cd(mkpath(joinpath(@__DIR__, "..", "resultados")))
open(CSV_OUT, "w") do io
    println(io, "L_mm;Lf_mm;Dc_mm;n_fins;wf_mm;lf_mm;Dt_mm;chanfro_mm;P_max_MPa;P_boost_MPa;",
                "P_sustain_MPa;P_min_MPa;t_pico_s;f_queda;f_subida;dPdt_queda_MPa_s;",
                "Dt_final_mm;Isp_s;I_total_Ns;t_queima_s;m_prop_kg;escore;viavel")
    for r in linhas
        s = @sprintf("%.0f;%.0f;%.0f;%d;%.0f;%.0f;%.0f;%.0f;%.4f;%.4f;%.4f;%.4f;%.2f;%.4f;%.4f;%.4f;%.2f;%.2f;%.1f;%.3f;%.2f;%.5f;%d",
                     r.Lt, r.Lf, r.Dc, r.nf, r.fw, r.fl, r.th, r.ft, r.P_max, r.P_boost,
                     r.P_sust, r.P_min, r.t_pico, r.queda_frac, r.subida_frac, r.dPdt_queda,
                     r.D_gt_fin, r.Isp, r.I_total, r.t_burn, r.m_prop, _score(r), _viavel(r))
        println(io, replace(s, '.' => ','))
    end
end
@printf("CSV: resultados/%s\n\nTOP %d (maior escore):\n", CSV_OUT, TOP_N)
@printf("%-5s %-5s %-4s %-3s %-3s %-4s %-4s | %-6s %-6s %-6s %-6s %-8s %-6s\n",
        "L", "Lf", "Dc", "Nf", "wf", "lf", "Dt", "Pbst", "Psust", "Pmin", "tpico", "I[kNs]", "mp[kg]")
for r in first(viaveis, min(TOP_N, length(viaveis)))
    @printf("%-5.0f %-5.0f %-4.0f %-3d %-3.0f %-4.0f %-4.0f | %-6.2f %-6.2f %-6.2f %-6.1f %-8.1f %-6.1f\n",
            r.Lt, r.Lf, r.Dc, r.nf, r.fw, r.fl, r.th, r.P_boost, r.P_sust, r.P_min,
            r.t_pico, r.I_total / 1e3, r.m_prop)
end
