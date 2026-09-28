# ==============================================================================
# SimulationCore.jl — laço temporal 1D (kernel type-stable)
# ==============================================================================
# Dois pilares de arquitetura:
#
# 1. FUNCTION BARRIER (type-stability).
#    `obter_geom_em_xy` é chamada ~10M vezes/simulação. Se `prop.layout` for
#    inferido como tipo abstrato, cada chamada vira dynamic dispatch → mata os
#    ganhos de @inline + SVector. `executar_simulacao_v2` extrai `prop.layout` e
#    o passa como argumento typed para `_sim_kernel!`; Julia vê o tipo concreto L
#    em compile-time → especializa → zero dispatch (verificado: sem alocação por
#    célula no hot loop).
#
# 2. SimulationState — agrupa todo o estado mutável da simulação num struct único,
#    alocado uma vez e passado por referência. Torna o kernel testável,
#    serializável e pré-adaptado a threads/GPU.
#
# API pública: `executar_simulacao_v2` (chamado por `simular_caso`);
# CaseInput/ConfigModelo não mudam.
# ==============================================================================

using StaticArrays
using Printf

# ──────────────────────────────────────────────────────────────────────────────

"""
    salvar_perfil_alvo_csv(state, prop, nome_caso)

Grava "perfil_alvo_<caso>.csv": cabeçalho com t, P_cam, P_cabeça, ṁ de saída,
ṁ gerada e empuxo no instante em que P_cam atingiu cfg.P_perfil_alvo_MPa, e
por célula x, A, perímetro de queima, vazão injetada por metro, ρ, u, P, T,
M e P₀ (isentrópico local).
"""
function salvar_perfil_alvo_csv(state, prop, nome_caso::String)
    γ, R = prop.gamma, prop.R
    info = state.alvo_info
    fname = "perfil_alvo_$(nome_caso).csv"
    open(fname, "w") do io
        @printf(io, "# t_s=%.6f
# P_cam_MPa=%.6f
# P_cabeca_MPa=%.6f
", info[1], info[2], info[3])
        @printf(io, "# mdot_saida_kg_s=%.6f
# mdot_gerada_kg_s=%.6f
# empuxo_N=%.4f
", info[4], info[5], info[6])
        @printf(io, "# gamma=%.6f
# R=%.6f
", γ, R)
        println(io, "x_m,A_m2,P_queima_m,mdot_inj_kg_s_m,rho_kg_m3,u_m_s,P_MPa,T_K,M,P0_MPa")
        for i in eachindex(state.alvo_W)
            ρ, u, P = state.alvo_W[i]
            T  = P / (max(ρ, 1e-9) * R)
            M  = abs(u) / sqrt(γ * P / max(ρ, 1e-9))
            P0 = P * (1 + 0.5 * (γ - 1) * M^2)^(γ / (γ - 1))
            @printf(io, "%.6f,%.8e,%.6e,%.6e,%.6f,%.4f,%.8f,%.4f,%.6f,%.8f
",
                    state.x_centros_snap[i], state.alvo_A[i], state.alvo_Pb[i], state.alvo_minj[i],
                    ρ, u, P / 1e6, T, M, P0 / 1e6)
        end
    end
    println("  ✅ Perfil no nível de pressão escolhido: $(fname)")
end

# ──────────────────────────────────────────────────────────────────────────────
# 1. ESTADO MUTÁVEL DA SIMULAÇÃO
#    1 struct coerente, alocado uma vez, passado por referência (em vez de dezenas
#    de variáveis locais soltas). Benefício: testável unitariamente, serializável,
#    passável a threads.
# ──────────────────────────────────────────────────────────────────────────────

mutable struct SimulationState
    # ── Solução CFD (vetores conservativos e primitivos) ──────────────────────
    U       ::Vector{SVector{3, Float64}}   # [ρ, ρu, ρE]
    W       ::Vector{SVector{3, Float64}}   # [ρ,  u,  P] — primitivas
    U_star  ::Vector{SVector{3, Float64}}   # preditor RK2
    W_star  ::Vector{SVector{3, Float64}}
    U_novo  ::Vector{SVector{3, Float64}}   # resultado do passo RK2

    # ── Termos fonte e fluxos (pre-alocados, zero-copy no loop) ───────────────
    S            ::Vector{SVector{3, Float64}}
    S_star       ::Vector{SVector{3, Float64}}
    F_faces      ::Vector{SVector{3, Float64}}
    F_star_faces ::Vector{SVector{3, Float64}}

    # ── Estado sólido ─────────────────────────────────────────────────────────
    y_queima     ::Vector{Float64}   # regressão radial acumulada por célula [m]
    y_queima_pre ::Vector{Float64}   # snapshot pre-RK2 (estágio preditor)
    y_axial_faces::Vector{Float64}   # regressão axial: 2 faces × N_graos

    # ── Caches geométricos (evitam re-interpolação em cada chamada de fonte) ──
    r_local_cache::Vector{Float64}   # taxa de regressão local r(i) [m/s]
    geom_cache_A ::Vector{Float64}   # A_port pré-calculada por célula [m²]
    geom_cache_P ::Vector{Float64}   # P_burn pré-calculada por célula [m]

    # ── Superfície do sólido / ignição ────────────────────────────────────────
    T_sup    ::Vector{Float64}   # temperatura superficial por célula [K]
    est_ign  ::Vector{Bool}      # célula ignitada?
    t_heating::Vector{Float64}   # tempo acumulado de aquecimento por célula [s]

    # ── Escalares de estado ───────────────────────────────────────────────────
    t               ::Float64
    dt              ::Float64
    passo           ::Int
    D_garganta_atual::Float64
    m_consumida     ::Float64

    # ── Histórico da erosão da garganta ──────────────────────────────────────
    hist_D_garganta ::Vector{Float64}   # D_garganta(t) em mm, gravado a cada n_hist_skip

    # ── Histórico de saída ────────────────────────────────────────────────────
    hist_t              ::Vector{Float64}
    hist_Pcam           ::Vector{Float64}
    hist_Phead          ::Vector{Float64}
    hist_F_empuxo       ::Vector{Float64}
    hist_mach_saida     ::Vector{Float64}
    hist_m_dot_saida    ::Vector{Float64}
    hist_m_dot_gerada   ::Vector{Float64}
    hist_p_exit         ::Vector{Float64}
    hist_rho_exit       ::Vector{Float64}
    hist_u_exit         ::Vector{Float64}
    hist_A_queima_total ::Vector{Float64}
    hist_M_fluido       ::Vector{Float64}
    hist_E_fluido       ::Vector{Float64}
    hist_Tsup_media     ::Vector{Float64}
    hist_n_ign          ::Vector{Float64}

    # ── Controlo de burnout ───────────────────────────────────────────────────
    prop_queimando         ::Bool
    tempo_burnout          ::Float64
    tempo_candidato_burnout::Float64
    A_queima_pico          ::Float64
    m_dot_pico             ::Float64
    empuxo_pico            ::Float64
    total_fallbacks        ::Int

    # ── Snapshots espaciais (perfil de Mach / P / u / T ao longo do eixo) ────
    # Capturados automaticamente em 5%/20%/40%/60%/80% de t_maximo.
    # Permite visualizar a distribuição 1D real — diferencial vs OpenMotor/BurnSim.
    prof_snapshots ::Vector{SpatialSnapshot}   # perfis capturados
    prof_t_targets ::Vector{Float64}           # tempos-alvo [s]
    prof_next_idx  ::Int                       # próximo alvo ainda não capturado
    x_centros_snap ::Vector{Float64}           # cópia das posições das células [m]

    # ── Snapshots da regressão do grão (y_queima e A_port) ───────────────────
    # Capturados nos mesmos instantes dos prof_snapshots.
    # Alimentam plotar_regressao_grao: β(x,t), D_porto(x,t), Kn axial.
    grain_snapshots ::Vector{GrainRegSnapshot}

    # ── Cache dos invariantes (ṁ, T₀, p₀, M) por célula (termo de área
    #    bem-balanceado): calculados 1× por estágio em calcular_fluxos_nas_faces!
    #    e reutilizados no termo de área da célula. Índice k+2 ↔ célula k (−1..N+2).
    Q_cache ::Vector{NTuple{4, Float64}}

    # ── Perfil no nível de pressão escolhido (cfg.P_perfil_alvo_MPa) ─────────
    alvo_capturado ::Bool
    alvo_info      ::Vector{Float64}              # [t, P_cam, P_cabeça (MPa), ṁ_saída, ṁ_gerada (kg/s), F (N)]
    alvo_W         ::Vector{SVector{3, Float64}}  # (ρ, u, P) por célula
    alvo_A         ::Vector{Float64}              # área de passagem por célula [m²]
    alvo_Pb        ::Vector{Float64}              # perímetro de queima por célula [m]
    alvo_minj      ::Vector{Float64}              # vazão injetada por metro [kg/(s·m)]

    # Pressão de estagnação na entrada da tubeira [Pa], gravada com os históricos:
    # é a pressão que define C_F = F/(P₀·A_t) (a média estática da câmara fica abaixo).
    hist_P0_bocal  ::Vector{Float64}

    # Perfil "logo após a ignição": instante de G = ρ|u| máximo no grão antes do
    # 1º alvo fixo — o momento crítico para a queima erosiva (canal mais estreito).
    ign_Gmax       ::Float64
    ign_prof       ::Vector{SpatialSnapshot}
    ign_grain      ::Vector{GrainRegSnapshot}
end

"""
    criar_estado_inicial(malha, prop, D_garganta_ini, E_tab, T_tab, G_tab) -> SimulationState

Aloca e inicializa todo o estado mutável da simulação numa estrutura única.
Condições iniciais: gás atmosférico em repouso (P_atm, T_atm = 300 K).
Ignição instantânea em toda a câmara (equivalente ao OpenMotor).
"""
function criar_estado_inicial(
    malha       ::Malha1D,
    prop        ::Propelente{L},   # Fase 2
    D_garganta_ini::Float64,
    E_tab       ::Vector{Float64},
    T_tab       ::Vector{Float64},
    G_tab       ::Vector{Float64};
    usar_flame_spread::Bool = false,   # false = ignição instantânea (baseline)
    x_ignitor        ::Float64 = 0.0,  # posição do fim do ignitor [m] (só p/ flame-spread)
)::SimulationState where {L}
    N = malha.N
    P_atm = 101325.0
    T_atm = 300.0
    L_camara = comprimento_camara(prop)

    # ── Inicializar campos CFD ─────────────────────────────────────────────────
    U_ini, W_ini = inicializar_estado(malha, P_atm, T_atm, prop.R, prop.gamma)

    zSV  = zero(SVector{3, Float64})
    cap  = 600   # capacidade inicial dos históricos

    # ── Ignição inicial ───────────────────────────────────────────────────────
    # Default: instantânea em toda a câmara (= OpenMotor). Com flame-spread: só a
    # região do ignitor acende no t=0; o resto acende por aquecimento (physics.jl),
    # produzindo uma frente de chama com velocidade finita (transiente físico).
    T_sup   = fill(prop.T_grain, N)
    est_ign = fill(false, N)
    x_ign_ini = (usar_flame_spread && x_ignitor > 0.0) ? min(x_ignitor, L_camara) : L_camara
    @inbounds for i in 1:N
        if tem_propelente(i, malha, x_ign_ini)   # célula sobrepõe o trecho aceso (inclui a de borda)
            T_sup[i]   = prop.T_ignicao + 1.0
            est_ign[i] = true
        end
    end

    function _hist()::Vector{Float64}
        v = Float64[]
        sizehint!(v, cap)
        return v
    end

    return SimulationState(
        # Solução CFD
        copy(U_ini), copy(W_ini),
        copy(U_ini), copy(W_ini),
        copy(U_ini),
        # Termos fonte / fluxos
        zeros(SVector{3,Float64}, N),
        zeros(SVector{3,Float64}, N),
        zeros(SVector{3,Float64}, N+1),
        zeros(SVector{3,Float64}, N+1),
        # Estado sólido
        zeros(Float64, N),
        zeros(Float64, N),
        zeros(Float64, 2 * n_graos_fisicos(prop)),
        # Caches
        zeros(Float64, N),
        zeros(Float64, N),
        zeros(Float64, N),
        # Superfície
        T_sup, est_ign, zeros(Float64, N),
        # Escalares
        0.0, 1e-6, 0, D_garganta_ini, 0.0,
        # Histórico da garganta
        Float64[],
        # Históricos (15 vetores)
        _hist(), _hist(), _hist(), _hist(), _hist(),
        _hist(), _hist(), _hist(), _hist(), _hist(),
        _hist(), _hist(), _hist(), _hist(), _hist(),
        # Burnout
        true, -1.0, 0.0, 1e-8, 1e-8, 1e-8, 0,
        # Snapshots espaciais (targets preenchidos em executar_simulacao_v2)
        SpatialSnapshot[], Float64[], 1, copy(malha.x_centros),
        # Snapshots de regressão do grão
        GrainRegSnapshot[],
        # Cache dos invariantes do termo de área bem-balanceado
        Vector{NTuple{4, Float64}}(undef, N + 4),
        # Perfil no nível de pressão escolhido
        false, Float64[], SVector{3, Float64}[], Float64[], Float64[], Float64[],
        # P₀ na entrada da tubeira
        _hist(),
        # Perfil logo após a ignição (G máximo no grão)
        0.0, SpatialSnapshot[], GrainRegSnapshot[]
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# 2. FUNCTION BARRIER — O FIX CENTRAL DE TYPE-STABILITY
#
#    Por que é necessário?
#    `Propelente.layout` é declarado como `Any` (types.jl:101).
#    Quando o compilador Julia vê `prop.layout.segments` dentro de um loop
#    quente, ele não conhece o tipo concreto de `layout` → gera código com
#    dynamic dispatch → boxing de SVectors → GC pressure → lentidão.
#
#    A solução é a Function Barrier:
#    1. `executar_simulacao_v2` (função pública) extrai `prop.layout` — neste
#       ponto Julia infere o tipo concreto, por ex. GrainLayout{...}.
#    2. Passa-o como argumento para `_sim_kernel!` (função interna).
#    3. `_sim_kernel!` tem `layout::L where L` — Julia especializa-se
#       completamente em L e gera código nativo sem dispatch.
#
#    Resultado medido em benchmarks similares: 3–8× speedup no hot loop.
# ──────────────────────────────────────────────────────────────────────────────

"""
    executar_simulacao_v2(nome, prop, ig, cfg, geom; ...) -> (t, P, F)

Wrapper público que substitui `executar_simulacao`.
Extrai `prop.layout` para activar o function barrier interno.
API idêntica à versão anterior — substituição directa.
"""
function executar_simulacao_v2(
    nome_caso    ::String,
    prop         ::Propelente{L},   # Fase 2
    ig           ::Ignitor,
    cfg          ::ConfigModelo,
    geom         ::Dict;
    N_malha      ::Int     = 150,
    t_maximo     ::Float64 = 10.0,
    salvar_csv   ::Bool    = true,    # false durante Monte Carlo → sem I/O por run
) where {L}
    # ── Preparação (fora do hot loop) ─────────────────────────────────────────
    if !cfg.modo_silencioso
        println("=============================================================")
        println(" SPP v5.0 — SimulationCore (Function Barrier + SimState)    ")
        println("=============================================================")
        println("Caso: $nome_caso | N=$N_malha células")
    end

    # (Revisão final) Footgun: h_smear_faces_m > 0 reativa o smear axial de face,
    # que é NÃO-CONSERVATIVO (revertido na validação; ver VALIDACAO.md §7.3). Avisa
    # uma única vez (maxlog=1) para não poluir Monte Carlo.
    cfg.h_smear_faces_m > 0.0 && @warn(
        "cfg.h_smear_faces_m = $(cfg.h_smear_faces_m) > 0 reativa o smear axial de face " *
        "(NÃO-CONSERVATIVO — revertido na validação). Use 0.0 salvo teste deliberado.",
        maxlog = 1)

    malha = criar_malha(N_malha, geom["L_total"])

    E_tab, T_tab, G_tab = criar_tabelas_LUT(prop.R, prop.frac_alumina, prop.gamma)

    atualizar_geometria_tubeira!(malha, prop,
        geom["x_garganta"], geom["D_garganta_ini"], geom["D_saida"])

    state = criar_estado_inicial(malha, prop, geom["D_garganta_ini"],
                                  E_tab, T_tab, G_tab;
                                  usar_flame_spread = cfg.usar_flame_spread,
                                  x_ignitor         = ig.posicao_final)

    # Tempos-alvo para snapshots espaciais: 5%/20%/40%/60%/80% de t_maximo.
    # Cobrem ignição precoce, crescimento, queima estável, final e tail-off.
    state.prof_t_targets = [0.05, 0.20, 0.40, 0.80] .* t_maximo   # + o instante pós-ignição (G máx)

    atualizar_geometria_camara!(malha, prop, state.y_queima, state.y_axial_faces)

    h0_prop = calcular_h0_propelente(prop)

    # ── FUNCTION BARRIER ──────────────────────────────────────────────────────
    # Este é o momento crítico: `prop.layout` é lido UMA VEZ aqui.
    # Julia infere o tipo concreto de `layout` (ex: GrainLayout) neste scope.
    # Ao passá-lo como argumento typed para _sim_kernel!, o compilador
    # especializa-se no tipo L e elimina todo o dispatch dinâmico dentro.
    layout = prop.layout   # tipo concreto capturado aqui

    return _sim_kernel!(
        nome_caso, prop, layout, ig, cfg, geom,
        malha, state, E_tab, T_tab, G_tab, h0_prop;
        t_maximo        = t_maximo,
        salvar_csv      = salvar_csv,
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# 3. KERNEL INTERNO — TYPE-STABLE POR CONSTRUÇÃO
#    `layout::L where L` garante especialização total.
#    Todo o acesso a `layout.segments` aqui é resolvido em compile-time.
# ──────────────────────────────────────────────────────────────────────────────

function _sim_kernel!(
    nome_caso      ::String,
    prop           ::Propelente{L},
    layout         ::L,
    ig             ::Ignitor,
    cfg            ::ConfigModelo,
    geom           ::Dict,
    malha          ::Malha1D,
    state          ::SimulationState,
    E_tab          ::Vector{Float64},
    T_tab          ::Vector{Float64},
    G_tab          ::Vector{Float64},
    h0_prop        ::Float64;
    t_maximo       ::Float64  = 10.0,
    salvar_csv     ::Bool     = true,
) where {L}
    # ── Aliases locais (evitam acesso repetido aos campos de state) ────────────
    # Julia faz copy-on-bind de referências a vectores — não há cópia dos dados.
    U            = state.U
    W            = state.W
    U_star       = state.U_star
    W_star       = state.W_star
    U_novo       = state.U_novo
    S            = state.S
    S_star       = state.S_star
    F_faces      = state.F_faces
    F_star_faces = state.F_star_faces
    Q_cache      = state.Q_cache      # invariantes por célula (termo de área bem-balanceado)
    gmax_isen    = _gmax_isen(prop.gamma)  # vazão crítica adimensional (só depende de γ)
    y_queima     = state.y_queima
    y_queima_pre = state.y_queima_pre
    y_axial_faces= state.y_axial_faces
    r_local_cache= state.r_local_cache
    geom_cache_A = state.geom_cache_A
    geom_cache_P = state.geom_cache_P
    T_sup        = state.T_sup
    est_ign      = state.est_ign
    t_heating    = state.t_heating

    # ── Parâmetros geométricos fixos ──────────────────────────────────────────
    x_garganta     = geom["x_garganta"]
    D_garganta_ini = geom["D_garganta_ini"]
    D_saida        = geom["D_saida"]
    L_total        = geom["L_total"]
    L_camara       = comprimento_camara(prop)
    P_atm          = 101325.0

    alpha_div_graus   = get(geom, "alpha_divergencia", 15.0)
    lambda_div        = (1.0 + cos(alpha_div_graus * π / 180.0)) / 2.0
    # Parâmetros de erosão (configuráveis via CaseInput → geom dict)
    erosao_r_dot_ref  = get(geom, "erosao_r_dot_ref", 1.5e-4)  # m/s
    erosao_P_ref      = get(geom, "erosao_P_ref",     5.0e6)   # Pa
    erosao_n_exp      = get(geom, "erosao_n_exp",     0.8)

    idx_garganta = clamp(round(Int, x_garganta / malha.dx), 1, malha.N)
    n_cel_camara = sum(x -> x <= L_camara, malha.x_centros)
    # idx_head: célula de referência a 10 % do comprimento da câmara (head-end).
    # Usada apenas para registrar P_head no CSV (coluna diagnóstico).
    idx_head     = max(1, round(Int, 0.1 * n_cel_camara))

    idx_inicio_tubeira = clamp(round(Int, L_camara / malha.dx) + 1, 1, malha.N)
    idx_face_garganta_fixo   = idx_inicio_tubeira - 1 +
                               argmin(@view malha.A_faces[idx_inicio_tubeira:end])
    idx_centro_garganta_fixo = idx_inicio_tubeira - 1 +
                               argmin(@view malha.A_centros[idx_inicio_tubeira:end])

    A_exit = π * (D_saida / 2.0)^2

    # ── Loop Principal ────────────────────────────────────────────────────────
    if !cfg.modo_silencioso
        println("\n=> Iniciando kernel type-stable (Function Barrier activo)...")
        if cfg.usar_strang_splitting
            println("   Modo: STRANG SPLITTING  S(dt/2)→L(dt)→S(dt/2)  [instabilidade acústica]")
        else
            println("   Modo: SSP-RK2 acoplado  (padrão)")
        end
        println()
    end

    # Valores de E_tot / M_tot mantidos entre iterações (computados a cada 100 passos).
    # Inicializados a zero — a primeira gravação no histórico (passo 0) regista 0,
    # o que é correcto (estado inicial atmosférico).
    _E_tot = 0.0
    _M_tot = 0.0

    # Buffer para Strang splitting: snapshot de y_axial_faces antes do passo
    # (evita dupla contagem do avanço axial quando usar_strang_splitting=true)
    _y_axial_strang = zeros(Float64, 2 * n_graos_fisicos(prop))

    # Progresso p/ a barra da GUI: chama cfg.progress_cb(t, t_maximo) a cada ~0.5 %
    # do tempo (throttle — a GUI amostra a ~700 ms). nothing (padrão) = no-op:
    # scripts/testes/headless não pagam nada e a numérica não muda.
    _prog_cb   = cfg.progress_cb
    _prog_next = 0.0
    _prog_dt   = max(t_maximo, 1e-9) / 200

    while state.t < t_maximo
        # Sim em background (GUI): cede o passo ao escalonador p/ o loop HTTP do Dash
        # respirar mesmo se a thread da @spawn coincidir com a do servidor. No-op em
        # execução single-thread (scripts/testes) — NÃO altera a numérica.
        Threads.nthreads() > 1 && yield()

        # Reporta progresso (throttled) — só alimenta a barra da GUI; não afeta a numérica.
        if _prog_cb !== nothing && state.t >= _prog_next
            _prog_cb(state.t, t_maximo)
            _prog_next += _prog_dt
        end

        # CFL adaptativo
        max_vel = 1e-6
        @inbounds for i in 1:malha.N
            rho_l = max(W[i][1], 1e-6)
            p_l   = W[i][3]
            a_l   = sqrt(prop.gamma * p_l / rho_l)
            v_l   = abs(W[i][2]) + a_l
            if v_l > max_vel; max_vel = v_l; end
        end
        state.dt = cfg.cfl * malha.dx / max_vel
        if state.t < cfg.t_ramp_ignicao; state.dt = min(state.dt, cfg.dt_max_startup); end
        if state.t + state.dt > t_maximo; state.dt = t_maximo - state.t; end
        dt = state.dt

        # ── Frente de chama (flame-spread) ────────────────────────────────────
        # A ignição avança do ignitor à velocidade cfg.v_flame_spread, acendendo as
        # células alcançadas. Modelo EXPLÍCITO e robusto — não estagna a baixa pressão
        # como o aquecimento convectivo puro. Só age com usar_flame_spread=true.
        if cfg.usar_flame_spread
            x_flame = ig.posicao_final + cfg.v_flame_spread * max(state.t - ig.t_ign_start, 0.0)
            @inbounds for i in 1:malha.N
                if !est_ign[i] && malha.x_centros[i] <= x_flame && tem_propelente(i, malha, L_camara)
                    est_ign[i] = true
                    T_sup[i]   = prop.T_ignicao + 1.0
                end
            end
        end

        # Snapshot pré-RK2 (radial e axial)
        y_queima_pre .= y_queima
        if cfg.usar_strang_splitting
            _y_axial_strang .= y_axial_faces   # salva estado axial antes do avanço completo
        end

        # ── Regressão sólido + termo fonte (inclui atualização de y_queima) ───
        A_queima_total, _, _, m_dot_passo = calcular_termo_fonte_e_regressao!(
            S, y_queima, y_axial_faces, r_local_cache, dt,
            malha, W, prop, ig, cfg, T_sup, est_ign, t_heating, state.t,
            geom_cache_A, geom_cache_P, h0_prop
        )

        # Critério de corte de tail-off
        # Condição 1: t > t_min_tailoff (guarda contra falsos positivos na ignição)
        # Condição 2: OR propelente já esgotado — permite blowdown em motores curtos
        in_tailoff = !state.prop_queimando
        if (state.t > cfg.t_min_tailoff || in_tailoff) && W[1][3] < cfg.P_tailoff
            if !cfg.modo_silencioso
                println("\n=> TAIL-OFF: pressão abaixo do limiar. Finalizando.")
            end
            if state.tempo_burnout < 0.0; state.tempo_burnout = state.t; end
            break
        end

        state.m_consumida += m_dot_passo * dt
        state.A_queima_pico = max(state.A_queima_pico, A_queima_total)
        state.m_dot_pico    = max(state.m_dot_pico, m_dot_passo)

        # Erosão da garganta
        if cfg.usar_erosao_garganta
            # Pressão de ESTAGNAÇÃO na entrada do convergente (mesma referência do
            # 0D e da lei de erosão, calibrada em pressão de câmara). Antes usava a
            # estática na célula da garganta (~0,56·P₀ no sônico) → o 1D erodia
            # sistematicamente menos que o 0D para o mesmo ṙ_ref/P_ref.
            # Com o termo de área bem-balanceado (padrão), P₀ não é distorcido no
            # convergente e a própria célula da garganta dá o P₀ certo. No esquema
            # antigo, recua até a entrada do convergente (A ≥ 4·A_t), onde P₀ ainda
            # não foi inflado pelo erro do termo de área.
            A_t_atual = π * (state.D_garganta_atual / 2.0)^2
            i_ref = idx_garganta
            if !cfg.usar_termo_area_wb
                @inbounds while i_ref > 1 && malha.A_centros[i_ref] < 4.0 * A_t_atual
                    i_ref -= 1
                end
            end
            ρ_r, u_r, p_r = W[i_ref][1], W[i_ref][2], W[i_ref][3]
            M2_r  = u_r^2 / max(prop.gamma * p_r / max(ρ_r, 1e-6), 1e-6)
            P_gar = p_r * (1.0 + 0.5 * (prop.gamma - 1.0) * M2_r)^(prop.gamma / (prop.gamma - 1.0))
            taxa_er = calcular_taxa_erosao(P_gar, state.D_garganta_atual, cfg;
                          r_dot_ref = erosao_r_dot_ref,
                          P_ref     = erosao_P_ref,
                          n_exp     = erosao_n_exp)
            state.D_garganta_atual += 2.0 * taxa_er * dt
            atualizar_geometria_tubeira!(malha, prop, x_garganta,
                                         state.D_garganta_atual, D_saida)
        end

        # Geometria da câmara (a cada cfg.n_geom_skip passos — erro < 0.01%)
        if state.passo % cfg.n_geom_skip == 0
            atualizar_geometria_camara!(malha, prop, y_queima, y_axial_faces)
        end

        A_garganta_real = π * (state.D_garganta_atual / 2.0)^2
        if cfg.usar_erosao_garganta
            idx_fg = idx_inicio_tubeira - 1 +
                     argmin(@view malha.A_faces[idx_inicio_tubeira:end])
            idx_cg = idx_inicio_tubeira - 1 +
                     argmin(@view malha.A_centros[idx_inicio_tubeira:end])
        else
            idx_fg = idx_face_garganta_fixo
            idx_cg = idx_centro_garganta_fixo
        end
        malha.A_faces[idx_fg]    = A_garganta_real
        malha.A_centros[idx_cg]  = A_garganta_real

        if !cfg.usar_strang_splitting
        # ══════════════════════════════════════════════════════════════════════
        # MODO PADRÃO — SSP-RK2 acoplado (comportamento original)
        # ══════════════════════════════════════════════════════════════════════

        # ── Estágio 1: Preditor ───────────────────────────────────────────────
        calcular_fontes_gas_rk2!(S, W, y_queima_pre, y_axial_faces,
            r_local_cache, malha, prop, cfg, est_ign, state.t, ig,
            geom_cache_A, geom_cache_P, h0_prop)

        fb1, _, _, _ = calcular_fluxos_nas_faces!(F_faces, U, W, malha,
                            prop.R, P_atm, cfg, E_tab, T_tab, G_tab; Q_cache = Q_cache)
        state.total_fallbacks += fb1
        cfg.usar_garganta_sonica && _impor_fluxo_garganta!(F_faces, idx_fg, W, prop.gamma, prop.R, P_atm, cfg.usar_termo_area_wb ? nothing : malha.A_centros, A_garganta_real)

        @inbounds for i in 1:malha.N
            Ae = malha.A_faces[i];   Ad = malha.A_faces[i+1]
            Ac = malha.A_centros[i]
            H   = SVector{3,Float64}(0.0, (cfg.usar_termo_area_wb && !_area_plana(Ae, Ac, Ad)) ?
                      termo_area_momento_Q(Q_cache[i+2], Ae, Ad, prop.gamma, prop.R, gmax_isen) :
                      W[i][3]*(Ad - Ae), 0.0)
            Res = -(1.0/(Ac*malha.dx))*(F_faces[i+1]*Ad - F_faces[i]*Ae - H) + S[i]
            U_star[i] = U[i] + dt * Res
        end
        @inbounds for i in 1:malha.N
            rho_s = max(U_star[i][1], 1e-6)
            p_s, u_s, _, _, _ = propriedades_locais(U_star[i], prop.R, E_tab, T_tab, G_tab)
            W_star[i] = SVector{3,Float64}(rho_s, u_s, p_s)
        end

        # ── Estágio 2: Corretor ───────────────────────────────────────────────
        calcular_fontes_gas_rk2!(S_star, W_star, y_queima_pre, y_axial_faces,
            r_local_cache, malha, prop, cfg, est_ign, state.t, ig,
            geom_cache_A, geom_cache_P, h0_prop)

        fb2, p_exit, rho_exit, u_exit = calcular_fluxos_nas_faces!(
            F_star_faces, U_star, W_star, malha, prop.R, P_atm, cfg, E_tab, T_tab, G_tab; Q_cache = Q_cache)
        state.total_fallbacks += fb2
        cfg.usar_garganta_sonica && _impor_fluxo_garganta!(F_star_faces, idx_fg, W_star, prop.gamma, prop.R, P_atm, cfg.usar_termo_area_wb ? nothing : malha.A_centros, A_garganta_real)

        @inbounds for i in 1:malha.N
            Ae = malha.A_faces[i];   Ad = malha.A_faces[i+1]
            Ac = malha.A_centros[i]
            H    = SVector{3,Float64}(0.0, (cfg.usar_termo_area_wb && !_area_plana(Ae, Ac, Ad)) ?
                       termo_area_momento_Q(Q_cache[i+2], Ae, Ad, prop.gamma, prop.R, gmax_isen) :
                       W_star[i][3]*(Ad - Ae), 0.0)
            Res2 = -(1.0/(Ac*malha.dx))*(F_star_faces[i+1]*Ad - F_star_faces[i]*Ae - H) + S_star[i]
            U_novo[i] = 0.5*U[i] + 0.5*(U_star[i] + dt*Res2)
        end

        U .= U_novo
        @inbounds for i in 1:malha.N
            rho_n = max(U[i][1], 1e-6)
            p_n, u_n, _, _, _ = propriedades_locais(U[i], prop.R, E_tab, T_tab, G_tab)
            W[i] = SVector{3,Float64}(rho_n, u_n, p_n)
        end

        else # cfg.usar_strang_splitting
        # ══════════════════════════════════════════════════════════════════════
        # MODO STRANG — S(dt/2) → L(dt) → S(dt/2)     2ª ordem em tempo
        #
        # S = fonte de combustão + avanço do sólido (r·ρ_p·A_b, h0, Δy)
        # L = transporte hiperbólico (HLLC) + atrito viscoso
        #
        # Acoplamento acústica–combustão de 2ª ordem:
        #   • 1º meio-passo S usa r(P^n)      — taxa ao início do passo
        #   • Passo L resolve ondas acústicas  — P evolui para P^(n+1)
        #   • 2º meio-passo S usa r(P^(n+1))  — taxa ao final do passo
        # ══════════════════════════════════════════════════════════════════════
        dt2 = 0.5 * dt

        # ── S(dt/2): 1º meio-passo de combustão ──────────────────────────────
        # r_local_cache já foi calculado em calcular_termo_fonte_e_regressao!
        # (contém r(P^n)); usar aqui para avançar y e injetar massa em U.
        # Restaura y_queima E y_axial_faces para o snapshot t^n (evita dupla
        # contagem do dt completo já feito em calcular_termo_fonte_e_regressao!).
        y_queima      .= y_queima_pre    # restaura y_radial^n
        y_axial_faces .= _y_axial_strang # restaura y_axial^n

        A_queima_total, m_dot_passo = aplicar_meio_passo_combustao!(
            U, W, y_queima, y_axial_faces, r_local_cache,
            malha, prop, ig, cfg, est_ign, state.t,
            geom_cache_A, geom_cache_P, h0_prop, dt2)

        # Reconstrói W a partir de U após a injeção do 1º meio-passo
        @inbounds for i in 1:malha.N
            rho_s = max(U[i][1], 1e-6)
            p_s, u_s, _, _, _ = propriedades_locais(U[i], prop.R, E_tab, T_tab, G_tab)
            W[i] = SVector{3,Float64}(rho_s, u_s, p_s)
        end
        # Atualiza geometria da câmara com y^(n+1/2) recém calculado
        atualizar_geometria_camara!(malha, prop, y_queima, y_axial_faces)

        # ── L(dt): passo hiperbólico completo (só atrito, sem combustão) ──────
        # Estágio 1 (preditor)
        calcular_fontes_somente_atrito!(S, W, malha, prop, cfg,
                                        geom_cache_A, geom_cache_P)
        fb1, _, _, _ = calcular_fluxos_nas_faces!(F_faces, U, W, malha,
                            prop.R, P_atm, cfg, E_tab, T_tab, G_tab; Q_cache = Q_cache)
        state.total_fallbacks += fb1
        cfg.usar_garganta_sonica && _impor_fluxo_garganta!(F_faces, idx_fg, W, prop.gamma, prop.R, P_atm, cfg.usar_termo_area_wb ? nothing : malha.A_centros, A_garganta_real)

        @inbounds for i in 1:malha.N
            Ae = malha.A_faces[i];   Ad = malha.A_faces[i+1]
            Ac = malha.A_centros[i]
            H   = SVector{3,Float64}(0.0, (cfg.usar_termo_area_wb && !_area_plana(Ae, Ac, Ad)) ?
                      termo_area_momento_Q(Q_cache[i+2], Ae, Ad, prop.gamma, prop.R, gmax_isen) :
                      W[i][3]*(Ad - Ae), 0.0)
            Res = -(1.0/(Ac*malha.dx))*(F_faces[i+1]*Ad - F_faces[i]*Ae - H) + S[i]
            U_star[i] = U[i] + dt * Res
        end
        @inbounds for i in 1:malha.N
            rho_s = max(U_star[i][1], 1e-6)
            p_s, u_s, _, _, _ = propriedades_locais(U_star[i], prop.R, E_tab, T_tab, G_tab)
            W_star[i] = SVector{3,Float64}(rho_s, u_s, p_s)
        end

        # Estágio 2 (corretor)
        calcular_fontes_somente_atrito!(S_star, W_star, malha, prop, cfg,
                                        geom_cache_A, geom_cache_P)
        fb2, p_exit, rho_exit, u_exit = calcular_fluxos_nas_faces!(
            F_star_faces, U_star, W_star, malha, prop.R, P_atm, cfg, E_tab, T_tab, G_tab; Q_cache = Q_cache)
        state.total_fallbacks += fb2
        cfg.usar_garganta_sonica && _impor_fluxo_garganta!(F_star_faces, idx_fg, W_star, prop.gamma, prop.R, P_atm, cfg.usar_termo_area_wb ? nothing : malha.A_centros, A_garganta_real)

        @inbounds for i in 1:malha.N
            Ae = malha.A_faces[i];   Ad = malha.A_faces[i+1]
            Ac = malha.A_centros[i]
            H    = SVector{3,Float64}(0.0, (cfg.usar_termo_area_wb && !_area_plana(Ae, Ac, Ad)) ?
                       termo_area_momento_Q(Q_cache[i+2], Ae, Ad, prop.gamma, prop.R, gmax_isen) :
                       W_star[i][3]*(Ad - Ae), 0.0)
            Res2 = -(1.0/(Ac*malha.dx))*(F_star_faces[i+1]*Ad - F_star_faces[i]*Ae - H) + S_star[i]
            U_novo[i] = 0.5*U[i] + 0.5*(U_star[i] + dt*Res2)
        end
        U .= U_novo
        @inbounds for i in 1:malha.N
            rho_n = max(U[i][1], 1e-6)
            p_n, u_n, _, _, _ = propriedades_locais(U[i], prop.R, E_tab, T_tab, G_tab)
            W[i] = SVector{3,Float64}(rho_n, u_n, p_n)
        end

        # ── S(dt/2): 2º meio-passo de combustão ──────────────────────────────
        # Recalcula r a partir de P^(n+1) — CHAVE do acoplamento 2ª ordem.
        # A resposta da superfície vê a pressão DEPOIS da onda acústica passar.
        recalcular_r_de_W!(r_local_cache, W, y_queima,
                           malha, prop, cfg, est_ign,
                           geom_cache_A, geom_cache_P)

        A_q2, m2 = aplicar_meio_passo_combustao!(
            U, W, y_queima, y_axial_faces, r_local_cache,
            malha, prop, ig, cfg, est_ign, state.t + dt2,
            geom_cache_A, geom_cache_P, h0_prop, dt2)

        A_queima_total = max(A_queima_total, A_q2)   # pico para estatísticas
        m_dot_passo    = 0.5 * (m_dot_passo + m2)    # média dos dois meios-passos

        # W final a partir de U com injeção do 2º meio-passo
        @inbounds for i in 1:malha.N
            rho_n = max(U[i][1], 1e-6)
            p_n, u_n, _, _, _ = propriedades_locais(U[i], prop.R, E_tab, T_tab, G_tab)
            W[i] = SVector{3,Float64}(rho_n, u_n, p_n)
        end

        end # if usar_strang_splitting

        # ── Monitor de divergência (a cada 100 passos — economiza ~N iters/passo) ────
        # A cada 100 passos: recalcula E_tot/M_tot e verifica divergência.
        # Entre verificações, usa o valor em cache (_E_tot/_M_tot).
        # Justificativa: divergências numéricas são catastróﬁcas e detectáveis
        # com folga de 100 passos (dt~1e-6s → janela de ~1e-4s).
        if state.passo % 100 == 0
            _E_tot = 0.0
            _M_tot = 0.0
            @inbounds for i in 1:malha.N
                vol   = malha.A_centros[i] * malha.dx
                _M_tot += U[i][1] * vol
                _E_tot += U[i][3] * vol
            end
            # Detecta divergência: NaN, ±Inf, OU magnitude absurda em qualquer sinal.
            # (O teste antigo `_E_tot > 1e12` deixava passar divergências negativas,
            #  ex. −1.46e20, que só estouravam depois num round(Int,·) das métricas.)
            if !isfinite(_E_tot) || abs(_E_tot) > 1e12
                error("Simulação divergiu numericamente em t=$(round(state.t, digits=5))s " *
                      "(passo $(state.passo), N=$(malha.N)). Causas típicas: malha grossa " *
                      "demais para o gradiente, CFL alto, ou motor agressivo (porto apertado / " *
                      "Kn alto). Tente aumentar N, reduzir o CFL, ou abrir a garganta.")
            end
        end
        E_tot = _E_tot   # valor em cache (atualizado a cada 100 passos)
        M_tot = _M_tot

        # ── Métricas globais ──────────────────────────────────────────────────
        # P_cam_media: média ponderada pela área de porto sobre toda a câmara (F3).
        # • Ponderação por A_i(t) é mais representativa que a média simples dos
        #   primeiros 10% do comprimento: em grãos finocyl boost-sustain a pressão
        #   na cabeça pode diferir significativamente da região mid-grain.
        # • Custo: O(n_cel_camara) por ciclo — irrelevante frente ao solver de fluxo.
        P_cam_media = 0.0
        A_cam_sum   = 0.0
        @inbounds for i in 1:n_cel_camara
            ai           = malha.A_centros[i]
            P_cam_media += W[i][3] * ai
            A_cam_sum   += ai
        end
        P_cam_media /= max(A_cam_sum, 1e-12)
        P_head_atm   = W[idx_head][3] / 101325.0

        # η_bl: perda de camada limite na garganta (renomeado de eta_tubeira para
        # não confundir com prop.eta_tubeira, que é o derating GLOBAL opcional).
        eta_bl       = calcular_eta_bl(state.D_garganta_atual)

        # ── Estado no PLANO DE SAÍDA (área A_exit) ────────────────────────────
        # A última célula guarda o estado do seu CENTRO (área A_centros[N] <
        # A_exit). Multiplicar esse estado pela área da face de saída inflava ṁ
        # e empuxo em A_exit/A_centros[N] (≈ +6% num divergente de 15° com
        # N=100). Com saída supersônica, projeta-se a última célula para A_exit
        # pela onda estacionária isentrópica (mesmos ṁ, T₀, p₀): ṁ_saída passa a
        # ser o ρuA da própria célula e (p, u) os do plano de saída. Com saída
        # subsônica (partida/cauda) mantém-se a média com o estado fantasma da
        # condição de contorno (que impõe P_atm).
        W_N  = W[malha.N]
        M_N  = abs(W_N[2]) / sqrt(prop.gamma * max(W_N[3], 1e-2) / max(W_N[1], 1e-6))
        if M_N >= 1.0
            W_e = _projetar_estado(W_N, malha.A_centros[malha.N], A_exit, prop.gamma, prop.R)
            rho_exit, u_exit, p_exit = W_e[1], W_e[2], W_e[3]
        end
        m_dot_saida  = rho_exit * u_exit * A_exit
        empuxo_puro  = (m_dot_saida * u_exit) + (p_exit - P_atm) * A_exit
        empuxo_cfd   = max(0.0, empuxo_puro) * lambda_div * eta_bl * prop.eta_tubeira
        state.empuxo_pico = max(state.empuxo_pico, empuxo_cfd)

        T_exit  = p_exit / max(rho_exit * prop.R, 1e-6)
        M_exit  = abs(u_exit) / sqrt(prop.gamma * prop.R * max(T_exit, 1.0))

        # ── Registo histórico (a cada 1000 passos) ────────────────────────────
        if state.passo % cfg.n_hist_skip == 0
            push!(state.hist_t,              state.t)
            push!(state.hist_Pcam,           P_cam_media / 1e6)
            push!(state.hist_Phead,          P_head_atm)
            push!(state.hist_F_empuxo,       empuxo_cfd)
            push!(state.hist_mach_saida,     M_exit)
            push!(state.hist_m_dot_saida,    m_dot_saida)
            push!(state.hist_m_dot_gerada,   m_dot_passo)
            push!(state.hist_p_exit,         p_exit)
            push!(state.hist_rho_exit,       rho_exit)
            push!(state.hist_u_exit,         u_exit)
            push!(state.hist_A_queima_total, A_queima_total)
            push!(state.hist_M_fluido,       M_tot)
            push!(state.hist_E_fluido,       E_tot)
            push!(state.hist_Tsup_media,     sum(T_sup) / malha.N)
            push!(state.hist_n_ign,          Float64(sum(est_ign)))
            push!(state.hist_D_garganta,     state.D_garganta_atual * 1000.0)  # mm
            # P₀ na 1ª célula do convergente (subsônica; P₀ se conserva até a garganta)
            let Wb = W[clamp(idx_inicio_tubeira + 1, 1, malha.N)]
                M2b = Wb[2]^2 / max(prop.gamma * Wb[3] / max(Wb[1], 1e-6), 1e-6)
                push!(state.hist_P0_bocal,
                      Wb[3] * (1.0 + 0.5 * (prop.gamma - 1.0) * M2b)^(prop.gamma / (prop.gamma - 1.0)))
            end
        end

        if state.passo % 20000 == 0 && !cfg.modo_silencioso
            @printf("T:%.2fs | P:%.2f MPa | F:%.1f N | Ma:%.2f | E:%.2e | Ign:%d/%d\n",
                state.t, P_cam_media/1e6, empuxo_cfd, M_exit, E_tot,
                sum(est_ign), count(i -> tem_propelente(i, malha, L_camara), 1:malha.N))
        end

        state.t    += dt
        state.passo += 1

        # ── Snapshot espacial (perfil de Mach / P / u / T) + regressão do grão ─
        # Custo: ~N floats alocados por snapshot (≤5 vezes por simulação → negligível).
        # Antes do 1º alvo fixo, guarda o instante de G máximo no grão (logo após a
        # ignição). Ignora a rampa inicial, cujo transiente de partida não é físico.
        if state.prof_next_idx == 1 && state.passo % 20 == 0 &&
           state.t > max(0.05, 3.0 * cfg.t_ramp_ignicao)
            G_grao = 0.0
            @inbounds for i in 1:malha.N
                tem_propelente(i, malha, L_camara) || break
                G_grao = max(G_grao, W[i][1] * abs(W[i][2]))
            end
            if G_grao > state.ign_Gmax
                state.ign_Gmax = G_grao
                empty!(state.ign_prof); empty!(state.ign_grain)
                push!(state.ign_prof, calcular_snapshot(W, malha, prop.gamma, prop.R, state.t))
                push!(state.ign_grain, GrainRegSnapshot(state.t, copy(y_queima), copy(geom_cache_A)))
            end
        end

        if state.prof_next_idx <= length(state.prof_t_targets) &&
           state.t >= state.prof_t_targets[state.prof_next_idx]
            if state.prof_next_idx == 1 && !isempty(state.ign_prof)
                append!(state.prof_snapshots, state.ign_prof)     # entra antes do 1º alvo
                append!(state.grain_snapshots, state.ign_grain)
            end
            push!(state.prof_snapshots,
                  calcular_snapshot(W, malha, prop.gamma, prop.R, state.t))
            push!(state.grain_snapshots,
                  GrainRegSnapshot(state.t, copy(y_queima), copy(geom_cache_A)))
            state.prof_next_idx += 1
        end

        # ── Perfil no nível de pressão escolhido (1ª vez que P_cam o atinge) ──
        if cfg.P_perfil_alvo_MPa > 0.0 && !state.alvo_capturado && state.t > 0.05 &&
           P_cam_media >= cfg.P_perfil_alvo_MPa * 1e6
            state.alvo_capturado = true
            state.alvo_info = [state.t, P_cam_media / 1e6, W[idx_head][3] / 1e6,
                               m_dot_saida, m_dot_passo, empuxo_cfd]
            state.alvo_W    = copy(W)
            state.alvo_A    = copy(malha.A_centros)
            state.alvo_Pb   = copy(geom_cache_P)
            # S[i][1] é a fonte de massa por volume do estágio preditor (modo RK2)
            state.alvo_minj = [S[i][1] * malha.A_centros[i] for i in 1:malha.N]
        end

        # ── Critério de burnout ───────────────────────────────────────────────
        # Usa P_cam_media (head-end, já calculada acima) em vez de média sobre
        # toda a câmara — equivalente para detecção de burnout (pressão cai uniformemente).
        # Elimina o loop redundante de n_cel_camara iterações por passo.
        # Guarda mínimo: evita falso positivo durante rampa de ignição (≈5 ms).
        # 0.05 s = 10× t_ramp_ignicao default — funciona para motores de qualquer tamanho.
        t_guard = max(0.05, 3.0 * cfg.t_ramp_ignicao)
        if state.t > t_guard && P_cam_media < 1.1 * P_atm
            if !cfg.modo_silencioso
                println("\n[BURNOUT] Pressão média abaixo do limiar. Finalizando...")
            end
            if state.tempo_burnout < 0.0; state.tempo_burnout = state.t; end
            break
        end

        crit_area   = A_queima_total  <= cfg.limiar_burnout_area_rel  * state.A_queima_pico
        crit_mdot   = m_dot_passo     <= cfg.limiar_burnout_mdot_rel  * state.m_dot_pico
        crit_empuxo = empuxo_cfd      <= cfg.limiar_burnout_empuxo_rel* state.empuxo_pico

        if state.t > t_guard && crit_area && crit_mdot && crit_empuxo
            state.tempo_candidato_burnout += dt
        else
            state.tempo_candidato_burnout = 0.0
        end

        if state.prop_queimando && state.tempo_candidato_burnout >= cfg.tempo_confirmacao_burnout
            state.prop_queimando = false
            state.tempo_burnout  = state.t
            if !cfg.modo_silencioso
                println("\n=> BURNOUT DETECTADO — iniciando blow-down (tail-off).")
            end
            # Sem break: o loop CFD continua para capturar a descida de pressão.
            # O gás remanescente na câmara exaure pela garganta → curva tail-off real.
            # O critério P_tailoff acima encerrará o loop quando Pc < P_tailoff.
        end
    end  # while

    # ── Pós-processamento ─────────────────────────────────────────────────────
    return _calcular_metricas_e_salvar(nome_caso, state, geom, prop, lambda_div;
        salvar_csv      = salvar_csv,
        silencioso      = cfg.modo_silencioso,
        cfg             = cfg)
end

# ──────────────────────────────────────────────────────────────────────────────
# 4. PÓS-PROCESSAMENTO (separado do kernel — testável isoladamente)
# ──────────────────────────────────────────────────────────────────────────────

function _calcular_metricas_e_salvar(
    nome_caso       ::String,
    state           ::SimulationState,
    geom            ::Dict,
    prop            ::Propelente{L},
    lambda_div      ::Float64;
    salvar_csv      ::Bool    = true,
    silencioso      ::Bool    = false,
    cfg             ::Union{Nothing, ConfigModelo} = nothing   # p/ a análise erosiva usar o modelo da simulação
) where {L}
    hist_t = state.hist_t
    hist_F = state.hist_F_empuxo
    hist_P = state.hist_Pcam

    # ── Impulso total (integração trapezoidal com dt variável) ────────────────
    It = 0.0
    for k in 2:length(hist_t)
        It += 0.5 * (hist_F[k] + hist_F[k-1]) * (hist_t[k] - hist_t[k-1])
    end

    # ── Métricas filtradas (exclui transiente de ignição) ─────────────────────
    idx_q   = findall(k -> hist_F[k] > 50.0 && hist_P[k] > 0.5, 1:length(hist_t))
    f_medio = isempty(idx_q) ? 0.0 : sum(hist_F[idx_q]) / length(idx_q)
    p_medio = isempty(idx_q) ? 0.0 : sum(hist_P[idx_q]) / length(idx_q)   # [MPa]

    # P_max = ESTÁTICA DE PICO (cabeça), NÃO a média da câmara: é a pressão que a
    # carcaça realmente vê (MEOP). hist_Pcam (média ponderada por área) subestima
    # em motores com gradiente axial (alto L/D). hist_Phead = referência de cabeça
    # (célula a 10% do comprimento, em atm). hist_Pcam permanece em P_avg/CF.
    # head ≥ média a cada instante → esta troca é sempre ≥ conservadora p/ MEOP.
    P_max_v = isempty(state.hist_Phead) ? 0.0 : maximum(state.hist_Phead) * 0.101325  # atm→MPa
    F_max_v = isempty(hist_F) ? 0.0 : maximum(hist_F)

    # Garganta de cada instante (com erosão) — mesma definição do 0D.
    # Antes: A_t inicial e média estática da câmara → C_F ~7 % alto e Kn ~6 % alto
    # no fim da queima de motores com erosão de garganta.
    At_hist = π .* (state.hist_D_garganta ./ 2000.0) .^ 2                  # [m²]
    cf_k    = [hist_F[k] / (state.hist_P0_bocal[k] * At_hist[k]) for k in idx_q
               if state.hist_P0_bocal[k] > 1e5]
    CF_avg  = isempty(cf_k) ? 0.0 : sum(cf_k) / length(cf_k)

    kn_hist = state.hist_A_queima_total ./ At_hist
    kn_ini  = isempty(kn_hist) ? 0.0 : kn_hist[1]
    kn_max  = isempty(kn_hist) ? 0.0 : maximum(kn_hist)

    isp = It / max(state.m_consumida * 9.80665, 1e-7)

    t_burnout = state.tempo_burnout > 0.0 ? state.tempo_burnout : state.t

    # ── Verificação de conservação de massa de propelente ─────────────────────
    # Compara m_consumida (integral de ṁ no tempo) com a massa de propelente
    # geometricamente disponível (ρ_p · V_sólido_inicial).
    #   • balanço ≈ 1.0          → integração da área de queima consistente
    #   • balanço > 1.0 (muito)  → injeção espúria de massa (BUG na A_b)
    #   • balanço < 1.0          → slivers/burnout antes do web total (normal p/ finocyl/star)
    # Detecta erros sutis de geometria ao estrear grãos novos.
    A_ext_sec   = π / 4.0 * prop.D_ext^2
    L_grain_tot = comprimento_camara(prop)
    V_porto_ini = 0.0
    for seg in prop.layout.segments
        L_seg = seg.x_end - seg.x_start
        A_p   = seg.is_transition ?
            0.5 * (seg.geom_a.interp_A(0.0) + seg.geom_b.interp_A(0.0)) :
            seg.geom_a.interp_A(0.0)
        V_porto_ini += A_p * L_seg
    end
    V_solido_ini = prop.grain_type === :end_burner ?
        A_ext_sec * L_grain_tot :
        max(A_ext_sec * L_grain_tot - V_porto_ini, 1e-12)
    m_prop_disp   = prop.rho_p * V_solido_ini
    balanco_massa = state.m_consumida / max(m_prop_disp, 1e-12)

    # Interpretação do balanço de massa (m_consumida / m_prop_disponível):
    #   • ~1.0                       → conservação consistente (inclui FACE ABERTA: o
    #                                  double-count de canto foi eliminado ao reverter o
    #                                  smear axial não-conservativo — h_smear=0. Ver
    #                                  VALIDACAO.md §7.3 e ConfigModelo.h_smear_faces_m).
    #   • > 1.05                     → injeção espúria REAL (bug na integração da A_b)
    #   • < 0.60 com burnout natural → sliver grande / A_b subestimada (normal finocyl/star)
    burnout_natural = !state.prop_queimando
    lim_bug = 1.05
    # SLIVER: propelente não queimado ao burnout natural (cantos de web de finocyl/star).
    # sliver_frac = 1 − balanço; perda de impulso ≈ massa de sliver × Isp médio (o impulso
    # que essa massa TERIA entregue). Só faz sentido no burnout NATURAL (senão é sim truncada).
    sliver_frac = (burnout_natural && balanco_massa < 1.0) ? clamp(1.0 - balanco_massa, 0.0, 1.0) : 0.0
    m_sliver    = sliver_frac * m_prop_disp
    I_perda_sliver = m_sliver * (It / max(state.m_consumida, 1e-9))   # N·s "perdidos" no sliver
    if balanco_massa > lim_bug
        @warn @sprintf(
            "[Conservação de massa] balanço = %.3f (consumida %.3f kg / disponível %.3f kg) — injeção espúria de massa; verifique a integração da área de queima.",
            balanco_massa, state.m_consumida, m_prop_disp)
    elseif burnout_natural && sliver_frac > 0.02
        @warn @sprintf(
            "[Sliver] %.1f%% da massa (%.3f kg) não queimou no burnout natural — perda de impulso ≈ %.0f N·s (%.1f%% de I_total). Normal em finocyl/star; reduza afinando as aletas/web.",
            100*sliver_frac, m_sliver, I_perda_sliver, 100*I_perda_sliver/max(It,1e-9))
    end

    if !silencioso
        println("\n================ RELATÓRIO FINAL (SimulationCore) ================")
        @printf("Caso:              %s\n",    nome_caso)
        @printf("Tempo Burnout:     %.2f s\n", t_burnout)
        @printf("Kn (Ini / Máx):    %.1f / %.1f\n", kn_ini, kn_max)
        @printf("Empuxo Médio:      %.2f N\n",  f_medio)
        @printf("Impulso Total:     %.1f N·s\n", It)
        @printf("Isp:               %.1f s\n",   isp)
        @printf("Massa Consumida:   %.3f kg\n",  state.m_consumida)
        @printf("Massa Prop. Disp.: %.3f kg  (balanço m_cons/m_disp = %.3f)\n",
                m_prop_disp, balanco_massa)
        if burnout_natural && sliver_frac > 0.005
            @printf("Sliver:            %.1f%% (%.3f kg) — perda de impulso ≈ %.0f N·s (%.1f%%)\n",
                    100*sliver_frac, m_sliver, I_perda_sliver, 100*I_perda_sliver/max(It,1e-9))
        end
        @printf("Fallbacks Núm.:    %d\n",       state.total_fallbacks)
        println("===================================================================")
    end

    # ── Salvar CSV ────────────────────────────────────────────────────────────
    csv_path = ""
    if salvar_csv
        csv_path = "resultados_v2_" * nome_caso * ".csv"
        writedlm(csv_path,
            hcat(hist_t, hist_P, state.hist_Phead, hist_F,
                 state.hist_mach_saida, state.hist_m_dot_saida,
                 state.hist_m_dot_gerada, state.hist_A_queima_total),
            ','
        )
        if !silencioso
            println("Dados salvos em: $csv_path")
        end

        # ── Perfis espaciais 1D (Mach / P / u / T) ───────────────────────────
        if !isempty(state.prof_snapshots)
            if !silencioso
                println("\n── Perfis espaciais 1D ──────────────────────────────────")
            end
            salvar_perfis_csv(state.prof_snapshots, state.x_centros_snap, nome_caso)
            state.alvo_capturado && salvar_perfil_alvo_csv(state, prop, nome_caso)
            plotar_perfis_camara(state.prof_snapshots, state.x_centros_snap,
                                  geom["x_garganta"], nome_caso; salvar = true)
        end

        # ── Regressão do grão ────────────────────────────────────────────────
        if !isempty(state.grain_snapshots)
            if !silencioso
                println("\n── Regressão do grão ────────────────────────────────────")
            end
            At_ini = π * (geom["D_garganta_ini"] / 2.0)^2
            plotar_regressao_grao(
                state.grain_snapshots, state.x_centros_snap,
                geom["x_garganta"], prop.y_max, prop.D_ext, At_ini,
                nome_caso; salvar = true
            )
        end

        # ── Análise de queima erosiva (Feature 1.2) ──────────────────────────
        if !isempty(state.prof_snapshots) && !isempty(state.grain_snapshots)
            if !silencioso
                println("\n── Análise de queima erosiva ─────────────────────────────")
            end
            try
                res_ero = analisar_erosao_espacial(
                    state.prof_snapshots, state.grain_snapshots,
                    prop, state.x_centros_snap;
                    cfg = cfg, nome_caso = nome_caso)
                plotar_analise_erosiva(res_ero; salvar = true)
                if !silencioso
                    me = res_ero["metricas"]
                    @printf("  G_max no grão:     %.1f kg/(m²·s)\n", me["G_max_geral"])
                    if me["alpha_e_ativo"]
                        @printf("  Aumento erosivo máx. aplicado (grão): %.1f %%\n", me["aug_max_pct"])
                    else
                        @printf("  Queima erosiva:    desligada na simulação\n")
                    end
                end
            catch e
                @warn "ErosiveAnalysis: $(sprint(showerror, e))"
            end
        end

    end

    # ── Erosão da garganta ────────────────────────────────────────────────────
    D_gar_ini_mm  = geom["D_garganta_ini"] * 1000.0
    D_gar_fin_mm  = state.D_garganta_atual * 1000.0
    erosao_rad_mm = max(0.0, (D_gar_fin_mm - D_gar_ini_mm) / 2.0)

    if !silencioso && erosao_rad_mm > 0.001
        @printf("Erosão garganta:   Δr = %.3f mm  (%.2f → %.2f mm diâm.)\n",
                erosao_rad_mm, D_gar_ini_mm, D_gar_fin_mm)
    end

    return SimulationResult(
        nome_caso,
        collect(hist_t), collect(hist_P), collect(hist_F),
        t_burnout, It, isp, state.m_consumida,
        P_max_v, p_medio, F_max_v, f_medio, CF_avg,
        kn_ini, kn_max,
        state.total_fallbacks, csv_path,
        # Fatores de correção: neutros aqui; CaseRunner substitui com valores reais.
        1.0, 1.0, 1.0, isp, It,
        0.0,   # d43_um — preenchido pelo CaseRunner após predição Hermsen
        # Erosão da garganta
        D_gar_fin_mm, erosao_rad_mm, collect(state.hist_D_garganta),
        # Kn(t)
        collect(kn_hist),
        # MEOP: placeholder; CaseRunner preenche a partir de inp.P_meop_MPa
        0.0, Inf
    )
end
