# ==============================================================================
# Solver0D.jl — Modelo quasi-estático 0D para SRMs (parâmetros concentrados)
# ==============================================================================
#
# Modelo quasi-steady: câmara tratada como volume uniforme (sem gradientes axiais).
# A equação de equilíbrio balístico P_eq(y) = (ρ_p·a·c*·Kn)^(1/(1−n))
# define a pressão em cada instante da regressão y ∈ [0, y_max].
#
# Integração temporal em espaço-y:
#   dt = dy / r(P_eq)   com  r = a · P^n
#   ts[i] = ts[i-1] + 0.5·(1/r[i-1] + 1/r[i])·Δy   (regra do trapézio)
#
# Empuxo ao nível do mar:
#   Fs[i] = Cf_vac · λ_div · Peqs[i] · At  −  Pa · Ae
#
# onde Cf_vac é o Cf isentrópico a vácuo calculado via Newton-Raphson.
#
# Nota sobre eficiências (mesma convenção do SimulationCore.jl):
#   λ_div  já está EMBUTIDO nas empuxos retornados (em Cf, igual ao solver 1D).
#   η_2ph  é aplicado pelo CaseRunner no pós-processamento (igual ao modo 1D).
#   Não há η_bl em 0D (efeito exclusivamente 1D / CFD axial).
#
# Erosão da garganta: suportada nos dois modos 0D (ver §5). Quando activa, A_t
# cresce ao longo da queima, o que baixa o Kn e a pressão; ε = Ae/At cai e o Cf
# também. Desligada, o caminho de cálculo é preservado bit-a-bit.
#
# Velocidade: ~1 000× mais rápido que o solver 1D MUSCL-HLLC.
# ==============================================================================

using Printf

"""
    simular_0d(inp::CaseInput, cfg::ConfigModelo) -> Union{SimulationResult, Nothing}

Simulação 0D quasi-estática de motor de foguete sólido.

## Modelo físico
- Câmara como volume uniforme (parâmetros concentrados)
- Pressão de equilíbrio: P_eq = (ρ_p · a · c* · Kn)^{1/(1−n)}
- Integração em espaço-y com regra trapezoidal: dt = dy / r(P_eq)
- Empuxo: F = Cf_vac · λ_div · Peq · At − Pa · Ae

## Eficiências
- `λ_div = (1 + cos α) / 2` é embutido no Cf (empuxos do resultado já corrigidos).
- `η_2ph` (duas fases) é aplicado pelo `CaseRunner` como pós-processamento.
- Sem `η_bl` (camada limite): efeito exclusivamente axial/CFD.

## Quando usar
- Pré-dimensionamento rápido
- Verificação de ordem de grandeza antes da simulação 1D
- Monte Carlo de alta velocidade

## Limitações
- Sem gradientes axiais de pressão / Mach
- Sem queima erosiva (depende do fluxo de massa axial G no porto — efeito exclusivamente 1D)
- Sem efeitos de ignição transiente detalhados
- Sem análise térmica da parede

## Retorna
`SimulationResult` com os mesmos campos do solver 1D, ou `nothing` em caso de falha.
"""
function simular_0d(inp::CaseInput, cfg::ConfigModelo) ::Union{SimulationResult, Nothing}

    # ── Despacho especial: end-burner ─────────────────────────────────────────
    # O end-burner queima AXIALMENTE (face plana) — P_burn = 0 em toda a LUT
    # radial do diagnostico_geometria, tornando Kn = 0 e rs = 1e-12 para todos
    # os pontos. Tratamos analiticamente: câmara uniforme, queima constante.
    if inp.geometry_type == :end_burner
        return _simular_0d_end_burner(inp, cfg)
    end

    cfg.modo_silencioso || begin
        println("------------------------------------------------------")
        @printf("Caso (0D): %s\n", inp.name)
        println("Modo: quasi-estático (parâmetros concentrados)")
        println("------------------------------------------------------")
    end

    # ── 1. LUT de Kn(y) e P_eq(y) via diagnostico_geometria ──────────────────
    N_pts = 500
    local ys_lr, Kns_raw, Peqs_raw, _cstar
    try
        ys_lr, Kns_raw, Peqs_raw, _cstar = diagnostico_geometria(inp;
            N_amostra      = N_pts,
            plotar         = false,
            mostrar_tabela = !cfg.modo_silencioso
        )
    catch e
        @warn "[Solver0D] Falha em diagnostico_geometria: $e"
        return nothing
    end

    ys   = collect(Float64, ys_lr)    # LinRange → Vector para indexação
    Kns  = Vector{Float64}(Kns_raw)
    Peqs = Vector{Float64}(Peqs_raw)
    N    = length(ys)

    N < 2 && (@warn "[Solver0D] LUT com menos de 2 pontos — abortando."; return nothing)

    # ── Sensibilidade térmica de queima (mesma lei do solver 1D, physics.jl) ──
    # r = a·exp(σ_p·(T_grão−T_ref))·Pⁿ. O diagnostico_geometria devolve P_eq com
    # `a` cru; como P_eq ∝ a^(1/(1−n)), escalamos Peqs aqui. T_grão = T_ref (default)
    # → f_therm = 1.0 exatamente → baseline bit-idêntico (não altera o validado).
    f_therm = exp(inp.sigma_p * (inp.T_grain - inp.T_ref))
    a_eff   = inp.a * f_therm
    if f_therm != 1.0
        Peqs .*= f_therm ^ (1.0 / (1.0 - inp.n))
    end

    # ── 2. Taxa de queima em cada ponto ──────────────────────────────────────
    # r = a_eff · P^n  [m/s]
    rs = a_eff .* (Peqs .^ inp.n)

    # ── 2b. Truncar ao fim da queima ──────────────────────────────────────────
    # Para geometrias como estrela e moon-burner, P_burn(y_max) = 0 → Peqs[end] = 0
    # → rs[end] = 0. Sem truncagem, o último passo de integração explode:
    #   dt = (1/rs[N]) * Δy → infinito.
    # Truncamos na última posição onde P_eq > P_tailoff (critério de extinção
    # idêntico ao do solver 1D). Isso equivale fisicamente a parar quando a
    # câmara não sustenta mais a combustão.
    P_tailoff_Pa = max(cfg.P_tailoff, 1_000.0)  # mínimo 1 kPa
    idx_end_raw  = findlast(P -> P > P_tailoff_Pa, Peqs)

    if isnothing(idx_end_raw)
        @warn "[Solver0D] Nenhum ponto com P_eq > P_tailoff — motor provavelmente inválido."
        return nothing
    end

    # Garante ao menos 2 pontos para integração
    idx_end = max(idx_end_raw, 2)

    if idx_end < N
        cfg.modo_silencioso || @printf(
            "  [Solver0D] Queima truncada em y=%.4f m (%.1f%% do y_max; P_eq > %.0f Pa)\n",
            ys[idx_end], 100.0 * ys[idx_end] / ys[end], P_tailoff_Pa)
        ys   = ys[1:idx_end]
        Kns  = Kns[1:idx_end]
        Peqs = Peqs[1:idx_end]
        rs   = rs[1:idx_end]
        N    = idx_end
    end

    # Clamp para evitar divisão por zero nos pontos remanescentes
    rs = max.(rs, 1e-12)

    # ── 3. Geometria da tubeira ───────────────────────────────────────────────
    At = π / 4.0 * inp.D_garganta_ini^2    # área da garganta [m²]
    Ae = π / 4.0 * inp.D_saida^2           # área de saída    [m²]
    ε  = Ae / At                            # razão de expansão [-]
    Pa = 101_325.0                          # pressão ambiente  [Pa]

    # ── 4. Coeficiente de empuxo vácuo (isentrópico) ─────────────────────────
    Cf_vac = _cf_isentropico_0d(ε, inp.gamma)

    # Eficiência de divergência cônica: λ = (1 + cos α) / 2
    # Embutida em Cf_eff (mesma convenção do SimulationCore → sem dupla contagem)
    λ_div  = (1.0 + cosd(inp.alpha_divergencia)) / 2.0
    Cf_eff = Cf_vac * λ_div

    # ── 5. Integração temporal (espaço-y → tempo) ─────────────────────────────
    #
    # EROSÃO DA GARGANTA
    # ------------------
    # Sem erosão, `Peqs` já vem pronto da LUT (calculado com A_t inicial) e a
    # integração é um trapézio simples. Com erosão, A_t deixa de ser constante e
    # o problema fica ACOPLADO: A_t depende de Δt, que depende de r, que depende
    # de P_eq, que depende de A_t. Resolve-se por ponto fixo em cada passo — a
    # erosão por passo é uma perturbação pequena, portanto converge em 2–3
    # iterações.
    #
    # A área de queima A_b(y) é uma propriedade do GRÃO e não muda com a
    # garganta; é ela que se preserva (não o Kn, que é A_b/A_t).
    At_ini    = At
    Ab        = Kns .* At_ini                    # área de queima [m²] por nó
    # Mesma chave que o 1D testa. `_cfg_com_erosao` (CaseRunner) já traduziu
    # `inp.erosao_ativa` para cá — testar os dois faria o 0D divergir do 1D quando
    # a erosão fosse pedida directamente pelo cfg num script.
    erosao_on = cfg.usar_erosao_garganta
    r_dot_ref = inp.erosao_r_dot_ref * 1e-3      # mm/s → m/s (raio)
    P_ref_ero = inp.erosao_P_ref_MPa * 1e6       # MPa → Pa
    n_exp_ero = inp.erosao_n_exp

    ts      = Vector{Float64}(undef, N); ts[1] = 0.0
    Dt_hist = fill(inp.D_garganta_ini, N)        # diâmetro da garganta [m]
    At_hist = fill(At_ini, N)

    if !erosao_on
        # Caminho original — preservado bit-a-bit quando a erosão está desligada.
        for i in 2:N
            Δy    = ys[i] - ys[i-1]
            ts[i] = ts[i-1] + 0.5 * (1.0 / rs[i-1] + 1.0 / rs[i]) * Δy
        end
    else
        expo  = 1.0 / (1.0 - inp.n)
        K_peq = inp.rho_p * a_eff * _cstar        # P_eq = (K·Kn)^{1/(1−n)}  (a_eff: sens. térmica)
        for i in 2:N
            Δy   = ys[i] - ys[i-1]
            Dt_i = Dt_hist[i-1]                   # chute inicial: garganta do passo anterior
            P_i  = Peqs[i]; r_i = rs[i]; dt_i = 0.0
            rdot_ant = calcular_taxa_erosao(Peqs[i-1], Dt_hist[i-1], cfg;
                           r_dot_ref = r_dot_ref, P_ref = P_ref_ero, n_exp = n_exp_ero)
            for _ in 1:3                          # ponto fixo
                At_i = π / 4.0 * Dt_i^2
                P_i  = (K_peq * Ab[i] / At_i)^expo
                r_i  = max(a_eff * P_i^inp.n, 1e-12)
                dt_i = 0.5 * (1.0 / rs[i-1] + 1.0 / r_i) * Δy
                rdot = 0.5 * (rdot_ant +
                       calcular_taxa_erosao(P_i, Dt_i, cfg;
                           r_dot_ref = r_dot_ref, P_ref = P_ref_ero, n_exp = n_exp_ero))
                Dt_i = Dt_hist[i-1] + 2.0 * rdot * dt_i    # taxa é RADIAL → ×2
            end
            Dt_hist[i] = Dt_i
            At_hist[i] = π / 4.0 * Dt_i^2
            Peqs[i]    = P_i
            rs[i]      = r_i
            Kns[i]     = Ab[i] / At_hist[i]       # Kn real (com a garganta aberta)
            ts[i]      = ts[i-1] + dt_i
        end

        # A truncagem de tail-off (§2b) usou a pressão SEM erosão. Como a erosão
        # baixa a pressão, o motor pode extinguir-se antes — re-verifica.
        idx_ero = findlast(P -> P > P_tailoff_Pa, Peqs)
        if !isnothing(idx_ero) && idx_ero < N
            idx_ero = max(idx_ero, 2)
            cfg.modo_silencioso || @printf(
                "  [Solver0D] Erosão antecipou o tail-off: y=%.4f m (era %.4f m)\n",
                ys[idx_ero], ys[N])
            ys = ys[1:idx_ero];   Kns = Kns[1:idx_ero]; Peqs = Peqs[1:idx_ero]
            rs = rs[1:idx_ero];   ts  = ts[1:idx_ero];  Ab   = Ab[1:idx_ero]
            Dt_hist = Dt_hist[1:idx_ero]; At_hist = At_hist[1:idx_ero]
            N = idx_ero
        end
    end
    t_burn = ts[end]

    # ── 6. Empuxo ao nível do mar [N] ─────────────────────────────────────────
    # F = η_tub · (Cf_eff · P · At − Pa · Ae)   (Cf já inclui λ_div)
    # η_tub multiplica o empuxo INTEIRO (inclusive o termo de pressão), que é a
    # convenção do campo "Efficiency" do OpenMotor. Default 1.0 = sem efeito.
    #
    # Com erosão, ε = Ae/At CAI ao longo da queima, portanto Cf também cai: o
    # empuxo perde por dois caminhos (menos pressão e pior expansão). Recalcular
    # Cf por passo custa um Newton por ponto — desprezável nesta escala.
    Fs = if !erosao_on
        inp.eta_tubeira .* max.(Cf_eff .* Peqs .* At .- Pa .* Ae, 0.0)
    else
        F_tmp = Vector{Float64}(undef, N)
        @inbounds for i in 1:N
            Cf_i     = _cf_isentropico_0d(Ae / At_hist[i], inp.gamma) * λ_div
            F_tmp[i] = inp.eta_tubeira *
                       max(Cf_i * Peqs[i] * At_hist[i] - Pa * Ae, 0.0)
        end
        F_tmp
    end

    # Pressão em MPa (convenção SimulationResult.pressoes)
    Ps_MPa = Peqs ./ 1e6

    # ── 7. Massa consumida (integração em y-espaço) ───────────────────────────
    # dm = ρ_p · A_b · dy. Usa A_b directamente (e não Kn·A_t), porque com
    # erosão o A_t do denominador do Kn já não é o mesmo em todos os pontos —
    # multiplicar por um A_t único daria massa errada.
    m_consumida = inp.rho_p * _trapz_0d(ys, Ab)

    # ── 8. Impulso total (integração trapezoidal em tempo) ────────────────────
    It = _trapz_0d(ts, Fs)

    # ── 9. Isp ────────────────────────────────────────────────────────────────
    g0  = 9.80665
    Isp = It / max(m_consumida * g0, 1e-9)

    # ── 10. Métricas filtradas (exclui ignição transitória) ───────────────────
    # Filtro idêntico ao de SimulationCore: F > 50 N  e  P > 0.5 MPa
    idx_q   = findall(k -> Fs[k] > 50.0 && Ps_MPa[k] > 0.5, 1:N)
    f_medio = isempty(idx_q) ? 0.0 : sum(Fs[idx_q]) / length(idx_q)
    p_medio = isempty(idx_q) ? 0.0 : sum(Ps_MPa[idx_q]) / length(idx_q)  # [MPa]

    P_max = isempty(Ps_MPa) ? 0.0 : maximum(Ps_MPa)   # [MPa]
    F_max = isempty(Fs)     ? 0.0 : maximum(Fs)        # [N]

    # CF médio (usa p_medio em Pa: *1e6). Com erosão, a referência é a garganta
    # MÉDIA na janela de queima — usar a inicial daria um Cf artificialmente alto.
    At_ref = isempty(idx_q) ? At_ini : sum(@view At_hist[idx_q]) / length(idx_q)
    CF_avg = (p_medio > 0.01) ? f_medio / (p_medio * 1e6 * At_ref) : 0.0

    Kn_ini = Kns[1]
    Kn_max = maximum(Kns)

    # ── 11. Histórico da garganta ─────────────────────────────────────────────
    hist_D_gar     = Dt_hist .* 1000.0             # [mm]
    Dt_mm          = hist_D_gar[end]               # diâmetro FINAL [mm]
    erosao_radial_mm = 0.5 * (Dt_mm - inp.D_garganta_ini * 1000.0)

    # ── 12. Relatório terminal ────────────────────────────────────────────────
    if !cfg.modo_silencioso
        println("\n=========== RELATÓRIO 0D (quasi-estático) ===========")
        @printf("Caso:           %s\n",       inp.name)
        @printf("t_burn:         %.3f s\n",   t_burn)
        @printf("I_total:        %.1f N·s\n", It)
        @printf("Isp:            %.1f s\n",   Isp)
        @printf("P_max:          %.3f MPa\n", P_max)
        @printf("F_max:          %.1f N\n",   F_max)
        @printf("m_consumida:    %.4f kg\n",  m_consumida)
        @printf("Kn_max:         %.1f\n",     Kn_max)
        @printf("Cf_vac:         %.4f  (ε=%.2f, γ=%.3f)\n", Cf_vac, ε, inp.gamma)
        @printf("λ_div (α=%.1f°): %.4f  → Cf_eff=%.4f\n",
                inp.alpha_divergencia, λ_div, Cf_eff)
        println("======================================================")
    end

    # ── 13. Montar SimulationResult ───────────────────────────────────────────
    # η_div = λ_div (já embutido em Cf_eff; registrado para rastreabilidade)
    # η_2ph = 1.0  (CaseRunner preencherá com o valor calculado)
    # η_total = 1.0 (idem)
    return SimulationResult(
        inp.name,
        ts, Ps_MPa, Fs,
        t_burn, It, Isp, m_consumida,
        P_max, p_medio, F_max, f_medio, CF_avg,
        Kn_ini, Kn_max,
        0, "",            # n_fallbacks=0, csv_path=""
        λ_div, 1.0, 1.0, Isp, It,   # eta_div, eta_2ph, eta_total, Isp_c, It_c
        0.0,              # d43_um (será preenchido pelo CaseRunner se frac_alumina>0)
        Dt_mm, erosao_radial_mm,   # D_garganta_final [mm], erosão radial acumulada [mm]
        hist_D_gar,       # hist_D_garganta (constante se a erosão estiver desligada)
        Kns,              # kn_hist = Kn(y), mesmo eixo de tempos
        0.0, Inf          # MEOP: placeholder (CaseRunner preenche)
    )
end

# ==============================================================================
# 0D TRANSIENTE (unsteady) — EDO de balanço de massa da câmara
# ==============================================================================
"""
    simular_0d_unsteady(inp, cfg) -> Union{SimulationResult, Nothing}

Modelo 0D **transiente** (unsteady): resolve a EDO de balanço de massa da câmara
(gás perfeito, Tc congelado):

    d(Pc·Vc)/dt = R·Tc·(ṁ_ger − ṁ_out)

com ṁ_ger = ρp·Ab(y)·r, ṁ_out = Pc·At/c*, r = a·Pc^n, dy/dt = r e o volume livre
Vc(y) crescendo conforme o grão regride (dVc/dt = Ab·r). Integração RK4 com passo
adaptativo à constante de tempo da câmara. Captura a **rampa de ignição**, o
**transiente de câmara** e o **tail-off** — é o mesmo modelo do OpenMotor.

Contrasta com `simular_0d` (quasi-estático: P_eq = (ρp·a·c*·Kn)^(1/(1−n))
algébrico e instantâneo). No regime estabelecido os dois convergem; divergem na
ignição, no tail-off e em picos de Kn muito agudos.
"""
function simular_0d_unsteady(inp::CaseInput, cfg::ConfigModelo)::Union{SimulationResult, Nothing}
    # End-burner: solução analítica dedicada (Ab da face plana, sem porto radial)
    inp.geometry_type == :end_burner && return _simular_0d_end_burner(inp, cfg)

    # ── 1. LUT Kn(y) ─────────────────────────────────────────────────────────
    local ys_lr, Kns_raw
    try
        ys_lr, Kns_raw, _, _ = diagnostico_geometria(inp;
            N_amostra = 500, plotar = false, mostrar_tabela = !cfg.modo_silencioso)
    catch e
        @warn "[Solver0D-unsteady] Falha em diagnostico_geometria: $e"
        return nothing
    end
    ys  = collect(Float64, ys_lr)
    Kns = Vector{Float64}(Kns_raw)
    N   = length(ys)
    N < 2 && (@warn "[Solver0D-unsteady] LUT < 2 pontos"; return nothing)
    y_max = ys[end]

    # ── 2. Tubeira, c*, Cf ───────────────────────────────────────────────────
    At     = π / 4.0 * inp.D_garganta_ini^2
    Ae     = π / 4.0 * inp.D_saida^2
    ε      = Ae / At
    Pa     = 101_325.0
    cstar  = calcular_cstar_teorico(inp.R, inp.Tc, inp.gamma, inp.eta_cstar)
    Cf_vac = _cf_isentropico_0d(ε, inp.gamma)
    λ_div  = (1.0 + cosd(inp.alpha_divergencia)) / 2.0
    Cf_eff = Cf_vac * λ_div
    RTc    = inp.R * inp.Tc

    # ── 3. Ab(y) + volume queimado acumulado + volume livre inicial ──────────
    Ab_nodes = Kns .* At                                   # área de queima [m²] por nó
    Vburn    = zeros(Float64, N)                           # ∫₀^y Ab dy' [m³]
    @inbounds for i in 2:N
        Vburn[i] = Vburn[i-1] + 0.5 * (Ab_nodes[i] + Ab_nodes[i-1]) * (ys[i] - ys[i-1])
    end
    V_prop_0 = Vburn[end]                                  # volume total de propelente
    L_cam    = inp.L_grao * Float64(inp.N_graos)
    V_bore   = π / 4.0 * inp.D_ext^2 * L_cam
    V_aft    = π / 4.0 * inp.D_ext^2 * max(inp.x_garganta - L_cam, 0.0)  # câmara aft
    V_free0  = max(V_bore - V_prop_0, 1e-6) + V_aft        # porto inicial + câmara aft

    # Interpolador linear (Ab, Vburn) em y; Ab=0 e Vburn=total após o esgotamento.
    @inline function interp_y(yq::Float64)
        yq <= ys[1] && return (Ab_nodes[1], Vburn[1])
        yq >= y_max && return (0.0, Vburn[N])              # grão esgotado → sem superfície
        lo = 1; hi = N
        while hi - lo > 1
            mid = (lo + hi) >> 1
            ys[mid] <= yq ? (lo = mid) : (hi = mid)
        end
        f = (yq - ys[lo]) / (ys[hi] - ys[lo])
        return (Ab_nodes[lo] + f * (Ab_nodes[hi] - Ab_nodes[lo]),
                Vburn[lo]    + f * (Vburn[hi]    - Vburn[lo]))
    end

    # EROSÃO DA GARGANTA: aqui a marcha já é no tempo, portanto basta avançar o
    # diâmetro junto com o estado. `A_t` deixa de ser fechado na `rhs` e passa a
    # ser argumento — dentro de um passo RK4 é congelado (a erosão é lenta face
    # à dinâmica da câmara: τ_câmara ~ ms, erosão ~ mm/s).
    # Mesma chave que o 1D testa. `_cfg_com_erosao` (CaseRunner) já traduziu
    # `inp.erosao_ativa` para cá — testar os dois faria o 0D divergir do 1D quando
    # a erosão fosse pedida directamente pelo cfg num script.
    erosao_on = cfg.usar_erosao_garganta
    r_dot_ref = inp.erosao_r_dot_ref * 1e-3      # mm/s → m/s (raio)
    P_ref_ero = inp.erosao_P_ref_MPa * 1e6       # MPa → Pa
    n_exp_ero = inp.erosao_n_exp

    # Sensibilidade térmica (mesma lei do 1D): a_eff = a·exp(σ_p·(T_grão−T_ref)).
    # T_grão = T_ref → a_eff = a (baseline inalterado).
    a_eff = inp.a * exp(inp.sigma_p * (inp.T_grain - inp.T_ref))

    # RHS da EDO: estado (P, y) e garganta A_t → (dP/dt, dy/dt)
    @inline function rhs(P::Float64, y::Float64, At_c::Float64)
        Abi, Vbi = interp_y(y)
        Vc   = V_free0 + Vbi
        Pc   = max(P, 0.1 * Pa)
        r    = a_eff * Pc^inp.n
        mgen = inp.rho_p * Abi * r
        mout = Pc * At_c / cstar
        dPdt = (RTc * (mgen - mout) - Pc * (Abi * r)) / Vc   # inclui −Pc·dVc/dt
        return (dPdt, r)
    end

    # ── 4. Integração RK4 adaptativa ─────────────────────────────────────────
    ts = Float64[]; Ps = Float64[]; Kns_t = Float64[]; Dts = Float64[]
    P = Pa; y = 0.0; t = 0.0
    Dt_c = inp.D_garganta_ini            # diâmetro corrente da garganta [m]
    At_c = At                            # área corrente [m²]
    push!(ts, t); push!(Ps, P); push!(Kns_t, Ab_nodes[1] / At_c); push!(Dts, Dt_c)
    t_max = 120.0
    P_stop = max(cfg.P_tailoff, 1_000.0)
    while t < t_max
        _, Vb = interp_y(y)
        τ  = cstar * (V_free0 + Vb) / (RTc * At_c)           # constante de tempo da câmara
        dt = clamp(0.05 * τ, 1e-6, 5e-3)
        k1P, k1y = rhs(P, y, At_c)
        k2P, k2y = rhs(P + 0.5dt * k1P, y + 0.5dt * k1y, At_c)
        k3P, k3y = rhs(P + 0.5dt * k2P, y + 0.5dt * k2y, At_c)
        k4P, k4y = rhs(P + dt * k3P,     y + dt * k3y,     At_c)
        P += dt / 6.0 * (k1P + 2k2P + 2k3P + k4P)
        y += dt / 6.0 * (k1y + 2k2y + 2k3y + k4y)
        t += dt
        P = max(P, 0.05 * Pa)
        y = min(y, y_max)
        if erosao_on
            rdot  = calcular_taxa_erosao(P, Dt_c, cfg;
                        r_dot_ref = r_dot_ref, P_ref = P_ref_ero, n_exp = n_exp_ero)
            Dt_c += 2.0 * rdot * dt      # taxa é RADIAL → ×2 no diâmetro
            At_c  = π / 4.0 * Dt_c^2
        end
        Abi, _ = interp_y(y)
        push!(ts, t); push!(Ps, P); push!(Kns_t, Abi / At_c); push!(Dts, Dt_c)
        # tail-off concluído: grão esgotado E pressão colapsou
        (y >= y_max && P < P_stop) && break
        # guarda de não-ignição (motor inválido)
        (t > 1.0 && P < 1.2 * Pa && y < 0.01 * y_max) &&
            (@warn "[Solver0D-unsteady] não pressurizou — motor inválido"; return nothing)
    end
    Nt = length(ts)

    # ── 5. Empuxo, métricas ──────────────────────────────────────────────────
    Ps_Pa  = Ps
    Ps_MPa = Ps ./ 1e6
    # Com erosão, ε = Ae/At cai ao longo da queima → Cf também cai. O empuxo
    # perde por dois caminhos: menos pressão e pior expansão.
    Ats    = (π / 4.0) .* (Dts .^ 2)
    Fs = if !erosao_on
        inp.eta_tubeira .* max.(Cf_eff .* Ps_Pa .* At .- Pa .* Ae, 0.0)
    else
        F_tmp = Vector{Float64}(undef, Nt)
        @inbounds for i in 1:Nt
            Cf_i     = _cf_isentropico_0d(Ae / Ats[i], inp.gamma) * λ_div
            F_tmp[i] = inp.eta_tubeira *
                       max(Cf_i * Ps_Pa[i] * Ats[i] - Pa * Ae, 0.0)
        end
        F_tmp
    end
    t_burn = ts[Nt]
    It     = _trapz_0d(ts, Fs)
    # massa consumida = propelente realmente queimado até o y final (se o grão
    # esgotou, interp_y(y_max) devolve Vburn[end] = todo o propelente).
    _, Vb_fim = interp_y(y)
    m_consumida = inp.rho_p * Vb_fim
    g0  = 9.80665
    Isp = It / max(m_consumida * g0, 1e-9)

    # Médias PONDERADAS NO TEMPO (o passo é adaptativo — média simples dos pontos
    # sobre-pesaria a fase inicial de baixa pressão, densa em pontos). Janela de
    # queima = trecho contíguo com F>50 N e P>0.5 MPa (exclui ignição e tail-off).
    idx_q = findall(k -> Fs[k] > 50.0 && Ps_MPa[k] > 0.5, 1:Nt)
    if isempty(idx_q)
        f_medio = 0.0; p_medio = 0.0
    else
        i1 = idx_q[1]; i2 = idx_q[end]
        Tw = max(ts[i2] - ts[i1], 1e-9)
        p_medio = _trapz_0d(ts[i1:i2], Ps_MPa[i1:i2]) / Tw
        f_medio = _trapz_0d(ts[i1:i2], Fs[i1:i2])     / Tw
    end
    P_max   = maximum(Ps_MPa)
    F_max   = maximum(Fs)
    # Cf médio referido à garganta MÉDIA da janela de queima (com erosão, usar a
    # inicial daria um Cf artificialmente alto).
    At_ref  = isempty(idx_q) ? At : sum(@view Ats[idx_q]) / length(idx_q)
    CF_avg  = (p_medio > 0.01) ? f_medio / (p_medio * 1e6 * At_ref) : 0.0
    Kn_ini  = Kns_t[1]
    Kn_max  = maximum(Kns_t)
    hist_D_gar = Dts .* 1000.0                       # [mm]
    Dt_mm      = hist_D_gar[end]                     # diâmetro FINAL [mm]
    erosao_radial_mm = 0.5 * (Dt_mm - inp.D_garganta_ini * 1000.0)

    if !cfg.modo_silencioso
        println("\n=========== RELATÓRIO 0D (unsteady / transiente) ===========")
        @printf("Caso:           %s\n",       inp.name)
        @printf("t_burn:         %.3f s\n",   t_burn)
        @printf("I_total:        %.1f N·s\n", It)
        @printf("Isp:            %.1f s\n",   Isp)
        @printf("P_max:          %.3f MPa   P_avg: %.3f MPa\n", P_max, p_medio)
        @printf("F_max:          %.1f N     F_avg: %.1f N\n",   F_max, f_medio)
        @printf("m_consumida:    %.4f kg    Kn_max: %.1f\n",    m_consumida, Kn_max)
        @printf("Vc0=%.4f L  (porto %.4f + aft %.4f L)  τ_câmara≈%.2f ms\n",
                V_free0*1e3, (V_bore-V_prop_0)*1e3, V_aft*1e3,
                1e3*cstar*V_free0/(RTc*At))
        println("============================================================")
    end

    return SimulationResult(
        inp.name,
        ts, Ps_MPa, Fs,
        t_burn, It, Isp, m_consumida,
        P_max, p_medio, F_max, f_medio, CF_avg,
        Kn_ini, Kn_max,
        0, "",
        λ_div, 1.0, 1.0, Isp, It,
        0.0,
        Dt_mm, erosao_radial_mm,      # D_garganta_final [mm], erosão radial [mm]
        hist_D_gar,                   # hist_D_garganta(t) — constante se erosão off
        Kns_t,                        # kn_hist = Kn(t), mesmo eixo de `ts`
        0.0, Inf
    )
end

# ==============================================================================
# Helpers privados
# ==============================================================================

"""
    _cf_isentropico_0d(ε, γ) -> Float64

Coeficiente de empuxo isentrópico a vácuo para razão de expansão `ε = Ae/At`
e razão de calores específicos `γ`.

Usa Newton-Raphson para encontrar o número de Mach de saída `Me` supersônico
a partir de `A/A*(Me) = ε`, depois calcula:

    Cf_vac = √[ 2γ²/(γ−1) · (2/(γ+1))^{(γ+1)/(γ−1)} · (1 − (Pe/P₀)^{(γ−1)/γ}) ]
             + ε · (Pe/P₀)

Referência: Sutton & Biblarz, *Rocket Propulsion Elements*, 8ª ed., Eq. 3-30.
"""
function _cf_isentropico_0d(ε::Float64, γ::Float64) ::Float64

    # ── A/A*(Me) para raiz supersônica ────────────────────────────────────────
    function area_ratio(Me::Float64)
        t = 1.0 + (γ - 1.0) / 2.0 * Me^2
        return (1.0 / Me) * (2.0 / (γ + 1.0) * t) ^ ((γ + 1.0) / (2.0 * (γ - 1.0)))
    end

    # ── Newton-Raphson para Me > 1 ────────────────────────────────────────────
    Me = 3.0   # chute inicial (supersônico)
    for _ in 1:200
        f  = area_ratio(Me) - ε
        dh = 1e-7 * max(Me, 1.0)
        fp = (area_ratio(Me + dh) - area_ratio(Me - dh)) / (2.0 * dh)
        abs(fp) < 1e-14 && break
        Me -= f / fp
        Me  = max(Me, 1.001)        # mantém raiz supersônica
        abs(f) < 1e-10 && break
    end

    # Fallback robusto: se Newton não convergiu (ε extremo), bisseção.
    # area_ratio(Me) é monótona crescente para Me>1 → converge sempre.
    if abs(area_ratio(Me) - ε) > 1e-6 * max(ε, 1.0)
        lo, hi = 1.0001, 100.0
        for _ in 1:100
            mid = 0.5 * (lo + hi)
            area_ratio(mid) < ε ? (lo = mid) : (hi = mid)
        end
        Me = 0.5 * (lo + hi)
    end

    # ── Razão de pressão isentrópica Pe/P₀ ────────────────────────────────────
    Pe_P0 = (1.0 + (γ - 1.0) / 2.0 * Me^2) ^ (-γ / (γ - 1.0))

    # ── Cf vácuo (Sutton & Biblarz, Eq. 3-30) ─────────────────────────────────
    Cf_vac = sqrt(
        2.0 * γ^2 / (γ - 1.0) *
        (2.0 / (γ + 1.0)) ^ ((γ + 1.0) / (γ - 1.0)) *
        (1.0 - Pe_P0 ^ ((γ - 1.0) / γ))
    ) + ε * Pe_P0

    return Cf_vac
end

"""
    _trapz_0d(x, y) -> Float64

Integração numérica trapezoidal de `y` sobre `x`.
Retorna 0.0 se `length(x) < 2`.
"""
function _trapz_0d(x::AbstractVector{Float64}, y::AbstractVector{Float64}) ::Float64
    n = length(x)
    n < 2 && return 0.0
    s = 0.0
    @inbounds for i in 2:n
        s += 0.5 * (y[i] + y[i-1]) * (x[i] - x[i-1])
    end
    return s
end

# ==============================================================================
# End-burner analítico (despacho especial)
# ==============================================================================

"""
    _simular_0d_end_burner(inp, cfg) -> Union{SimulationResult, Nothing}

Solução analítica 0D para geometria end-burner (queima axial pura).

O end-burner queima em face plana — a área de queima é constante e igual à
seção transversal do grão (`A_ext`). Portanto:
- Kn = A_ext / At  (constante)
- P_eq  = (ρ_p · a · c* · Kn)^{1/(1−n)}  (constante)
- r     = a · P_eq^n  (constante)
- t_burn = L_grao_total / r  (comprimento axial total / taxa de queima)
- F = (Cf_eff · P_eq · At) − Pa · Ae  (constante)

Não chama `diagnostico_geometria` — evita o caminho que retorna P_burn = 0
para todas as posições radiais (correto para LUT radial, mas inapropriado
para a queima axial do end-burner).
"""
function _simular_0d_end_burner(inp::CaseInput, cfg::ConfigModelo) ::Union{SimulationResult, Nothing}

    cfg.modo_silencioso || begin
        println("------------------------------------------------------")
        @printf("Caso (0D end-burner): %s\n", inp.name)
        println("Modo: analítico — queima axial (face plana constante)")
        println("------------------------------------------------------")
    end

    # ── Geometria da tubeira ──────────────────────────────────────────────────
    At = π / 4.0 * inp.D_garganta_ini^2    # área da garganta [m²]
    Ae = π / 4.0 * inp.D_saida^2           # área de saída    [m²]
    ε  = Ae / At                            # razão de expansão [-]
    Pa = 101_325.0                          # pressão ambiente  [Pa]

    # Área de queima = seção frontal do grão (face plana)
    A_ext = π / 4.0 * inp.D_ext^2          # [m²]

    # ── Kn constante ─────────────────────────────────────────────────────────
    Kn = A_ext / At

    # ── c* efetivo (fonte única: calcular_cstar_teorico) ─────────────────────
    cstar  = calcular_cstar_teorico(inp.R, inp.Tc, inp.gamma, inp.eta_cstar)

    # ── Sensibilidade térmica: a_eff = a·exp(σ_p·(T_grão−T_ref)) (mesma lei do 1D) ──
    a_eff = inp.a * exp(inp.sigma_p * (inp.T_grain - inp.T_ref))

    # ── Pressão de equilíbrio (constante) ────────────────────────────────────
    exp_p = 1.0 / (1.0 - inp.n)
    P_eq  = (inp.rho_p * a_eff * cstar * Kn) ^ exp_p   # [Pa]

    if P_eq < 1_000.0
        @warn "[Solver0D end-burner] P_eq calculada = $(round(P_eq/1e3; digits=1)) kPa — motor provavelmente inválido (Kn muito baixo)."
        return nothing
    end

    # ── Taxa de queima axial (constante) ─────────────────────────────────────
    r = a_eff * P_eq ^ inp.n   # [m/s]

    # ── Tempo de queima = comprimento total do propelente / r ─────────────────
    L_total_grao = inp.L_grao * Float64(inp.N_graos)   # [m]
    t_burn = L_total_grao / max(r, 1e-12)              # [s]

    # ── Coeficiente de empuxo ─────────────────────────────────────────────────
    Cf_vac = _cf_isentropico_0d(ε, inp.gamma)
    λ_div  = (1.0 + cosd(inp.alpha_divergencia)) / 2.0
    Cf_eff = Cf_vac * λ_div

    # ── Empuxo constante ao nível do mar ──────────────────────────────────────
    F = inp.eta_tubeira * max(Cf_eff * P_eq * At - Pa * Ae, 0.0)   # [N]

    # ── Série temporal uniforme ───────────────────────────────────────────────
    N_pts  = 200
    ts     = collect(LinRange(0.0, t_burn, N_pts))
    Ps_MPa = fill(P_eq / 1e6, N_pts)
    Fs     = fill(F, N_pts)
    Kns    = fill(Kn, N_pts)

    # ── Massa consumida ───────────────────────────────────────────────────────
    # m = ρ_p · A_ext · L_total  (todo o propelente)
    m_consumida = inp.rho_p * A_ext * L_total_grao

    # ── Impulso total e Isp ───────────────────────────────────────────────────
    g0  = 9.80665
    It  = F * t_burn
    Isp = It / max(m_consumida * g0, 1e-9)

    # ── Métricas ──────────────────────────────────────────────────────────────
    P_max_MPa = P_eq / 1e6
    CF_avg    = (P_eq > 0.0) ? F / (P_eq * At) : 0.0
    Dt_mm     = inp.D_garganta_ini * 1000.0
    hist_D_gar = fill(Dt_mm, N_pts)

    # ── Relatório terminal ────────────────────────────────────────────────────
    if !cfg.modo_silencioso
        println("\n=========== RELATÓRIO 0D — end-burner (analítico) ===========")
        @printf("Caso:           %s\n",       inp.name)
        @printf("t_burn:         %.3f s\n",   t_burn)
        @printf("P_eq:           %.3f MPa  (constante)\n", P_max_MPa)
        @printf("F:              %.1f N   (constante)\n",  F)
        @printf("I_total:        %.1f N·s\n", It)
        @printf("Isp:            %.1f s\n",   Isp)
        @printf("m_consumida:    %.4f kg\n",  m_consumida)
        @printf("Kn:             %.1f  (A_ext/At)\n", Kn)
        @printf("c*:             %.1f m/s\n", cstar)
        @printf("r:              %.4f mm/s\n", r * 1000.0)
        @printf("Cf_vac:         %.4f  (ε=%.2f, γ=%.3f)\n", Cf_vac, ε, inp.gamma)
        @printf("λ_div (α=%.1f°): %.4f  → Cf_eff=%.4f\n",
                inp.alpha_divergencia, λ_div, Cf_eff)
        println("=============================================================")
    end

    # ── Montar SimulationResult ───────────────────────────────────────────────
    return SimulationResult(
        inp.name,
        ts, Ps_MPa, Fs,
        t_burn, It, Isp, m_consumida,
        P_max_MPa, P_max_MPa, F, F, CF_avg,
        Kn, Kn,
        0, "",                            # n_fallbacks, csv_path
        λ_div, 1.0, 1.0, Isp, It,        # eta_div, eta_2ph, eta_total, Isp_c, It_c
        0.0,                              # d43_um
        Dt_mm, 0.0, hist_D_gar,          # D_garganta_final, erosao_radial_mm, hist_D_gar
        Kns,                              # kn_hist
        0.0, Inf                          # MEOP: placeholder (CaseRunner preenche)
    )
end
