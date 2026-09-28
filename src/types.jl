# ============================================================
# MÓDULO: ESTRUTURAS DE DADOS BASE
# Versão: Mega Otimizada (Type-Stable) - Suporte à Biblioteca de Geometrias
# ============================================================

# ============================================================
# MALHA 1D
# ============================================================

Base.@kwdef struct Malha1D
    N::Int
    L::Float64
    dx::Float64
    x_faces::Vector{Float64}
    x_centros::Vector{Float64}
    A_faces::Vector{Float64}
    A_centros::Vector{Float64}
end

# ============================================================
# INTERPOLADOR UNIFORME 1D — ZERO ALOCAÇÃO NO HOT LOOP
# ============================================================
#
# Substitui Interpolations.jl na avaliação de geometria.
# Requisito: nós equiespaçados — verdade para todos os LUTs
# (construídos com range() ou iteração uniforme + possível truncagem).
#
# Por que zero alocação?
#   • clamp, unsafe_trunc, @inbounds: todos operam em registros/stack.
#   • Nenhum objeto heap é criado durante a chamada.
#   • Interpolations.jl alocava ~16 bytes/chamada (WeightedIndex interno).
#
# Campos:
#   y0     — primeiro nó
#   inv_dy — 1/espaçamento (pré-computado: troca divisão por multiplicação)
#   y_max  — último nó (usado no clamp de extrapolação flat)
#   vals   — valores amostrados
struct UniformInterp1D
    y0    ::Float64
    inv_dy::Float64
    y_max ::Float64
    vals  ::Vector{Float64}
end

function UniformInterp1D(y_vec::AbstractVector{Float64}, vals::AbstractVector{Float64})
    n = length(y_vec)
    n >= 2          || error("UniformInterp1D: precisa de pelo menos 2 pontos.")
    n == length(vals) || error("UniformInterp1D: y_vec e vals devem ter o mesmo comprimento.")
    dy = (y_vec[end] - y_vec[1]) / (n - 1)
    dy > 0.0        || error("UniformInterp1D: espaçamento deve ser positivo (dy=$dy).")
    return UniformInterp1D(Float64(y_vec[1]), 1.0 / dy, Float64(y_vec[end]), collect(Float64, vals))
end

# Avaliação: extrapolação flat implícita via clamp + blend linear O(1)
@inline function (itp::UniformInterp1D)(y::Float64)
    y_c   = clamp(y, itp.y0, itp.y_max)
    idx_f = (y_c - itp.y0) * itp.inv_dy + 1.0   # índice base-1 (ponto flutuante)
    i     = clamp(Base.unsafe_trunc(Int, idx_f), 1, length(itp.vals) - 1)
    frac  = idx_f - i
    @inbounds return itp.vals[i] + frac * (itp.vals[i + 1] - itp.vals[i])
end

# ============================================================
# GEOMETRIA: LUT E LAYOUT AXIAL
# ============================================================

# Parametrizado em T = tipo concreto do interpolador.
# Evita campos de tipo abstrato que causam type instability e ~60GB de alocações.
struct GrainGeometryLUT{T}
    interp_A    ::T
    interp_Pburn::T   # perímetro que queima ativamente [m]
    interp_Pflux::T   # perímetro total do vazio (P_burn + P_wall) [m]
    interp_Pwall::T   # perímetro em contato com a carcaça inerte [m]
    y_max::Float64
    name::Symbol
end
# Construtor keyword — interp_Pwall é opcional para compatibilidade com código existente.
# Default: usa interp_Pburn como fallback (conservador; build_geometry sempre passa o valor real).
function GrainGeometryLUT(; interp_A, interp_Pburn, interp_Pflux,
                            interp_Pwall = interp_Pburn,
                            y_max, name)
    GrainGeometryLUT(interp_A, interp_Pburn, interp_Pflux, interp_Pwall, y_max, name)
end

# Segmento unificado — elimina Union type para zero alocações no loop quente.
# is_transition=false → segmento puro (geom_b ignorado)
# is_transition=true  → transição cônica geom_a → geom_b
# TA e TB podem ser tipos diferentes (ex: cilindro vs finocyl têm interpoladores distintos)
struct GrainSegment{TA, TB}
    x_start      ::Float64
    x_end        ::Float64
    geom_a       ::GrainGeometryLUT{TA}
    geom_b       ::GrainGeometryLUT{TB}
    is_transition::Bool

    # Construtor para segmento puro
    GrainSegment(x_start::Float64, x_end::Float64, geom::GrainGeometryLUT{T}) where T =
        new{T,T}(x_start, x_end, geom, geom, false)
    # Construtor para transição (TA e TB podem ser diferentes)
    GrainSegment(x_start::Float64, x_end::Float64, geom_a::GrainGeometryLUT{TA},
                 geom_b::GrainGeometryLUT{TB}, is_transition::Bool) where {TA,TB} =
        new{TA,TB}(x_start, x_end, geom_a, geom_b, is_transition)
end

const GrainTransitionSegment = GrainSegment
const AnyGrainSegment        = GrainSegment

# ==============================================================================
# GrainLayout{S} — Fase 1b: elimina boxing de elementos no hot loop
#
# Problema anterior:  segments::Vector{<:GrainSegment}
#   → tipo abstracto → cada segs[k] faz heap allocation → 6G alocações/sim
#
# Solução: S é o tipo concreto do elemento do vector.
#   Caso homogéneo (BATES, STAR, FINOCYL puro):
#     S = GrainSegment{T,T}  → Vector{GrainSegment{T,T}} completamente concreto
#     → segs[k] sem boxing, zero alocações no loop de geometria
#
#   Caso heterogéneo (finocyl com transição):
#     S = Union{GrainSegment{T1,T1}, GrainSegment{T1,T2}, GrainSegment{T2,T2}}
#     → Julia gera "union splitting": código especializado para cada ramo
#     → muito melhor que Vector{<:GrainSegment} abstracto, sem heap allocation
#
# Com Propelente{L} where L = GrainLayout{S}, a cadeia completa é type-stable:
#   prop.layout::GrainLayout{S}            ← concreto (Fase 1a)
#   prop.layout.segments::Vector{S}        ← concreto (Fase 1b — agora)
#   segs[k]::S                             ← unboxed / union-split
#   segs[k].geom_a.interp_A(y)::Float64   ← zero alloc
# ==============================================================================
struct GrainLayout{S}
    segments::Vector{S}
    L_total::Float64
end

# ============================================================
# PROPELENTE
# ============================================================

# ==============================================================================
# Propelente{L} — parametrico no tipo do layout (Fase 1a)
#
# Por que parametrico?
#   Com layout::Any (versão anterior), Julia não conseguia especializar as
#   funções de física (obter_geom_em_xy, atualizar_regressao_radial_celula!, etc.)
#   porque o tipo concreto do layout era desconhecido em compile-time.
#   Isso forçava dynamic dispatch em CADA acesso a prop.layout dentro do hot loop,
#   gerando ~82 bilhões de alocações por simulação (7 TiB!).
#
#   Com Propelente{L}, quando Julia compila f(prop::Propelente{L}) where L,
#   ela conhece o tipo exato de prop.layout e gera código nativo especializado:
#   zero dynamic dispatch, zero boxing, zero alocações desnecessárias.
#
# Como construir:
#   Propelente(layout = meu_layout, rho_p = ..., ...)
#   → Julia infere L = typeof(meu_layout) automaticamente via @kwdef
# ==============================================================================
Base.@kwdef mutable struct Propelente{L}
    # Balística e Termodinâmica
    rho_p::Float64
    a::Float64
    n::Float64
    Tc::Float64
    gamma::Float64
    R::Float64
    eta_cstar::Float64 = 1.0
    # Derating global e opcional do empuxo (ver CaseInput.eta_tubeira). 1.0 = off.
    eta_tubeira::Float64 = 1.0

    # Térmica
    k_p::Float64 = 0.43
    cp_p::Float64 = 1465.0
    frac_alumina::Float64 = 0.0
    T_ignicao::Float64 = 600.0

    # Geometria e dimensões globais
    D_ext::Float64 = 0.0
    D_port_ini::Float64 = 0.0
    L_grao::Float64 = 0.0
    N_graos::Int64 = 1
    y_max::Float64 = 0.0

    # Fronteiras físicas dos grãos [m] — N_grãos_físicos+1 valores: [0, L₁, L₁+L₂, …].
    # VAZIO (default) → usa L_grao·N_graos uniforme (retrocompat BIT-IDÊNTICA: o caminho
    # de faces cai no fallback (g-1)·L_grao). NÃO-vazio → stack heterogêneo de grãos com
    # comprimentos livres (estilo OpenMotor: finocyl + N BATES empilhados).
    grain_boundaries::Vector{Float64} = Float64[]

    # Faces inibidas POR grão físico (0/1/2/3) — paralelo a grain_boundaries.
    # VAZIO = usa inhibited_ends global (retrocompat). Não-vazio = faces por-grão no stack.
    grain_inhibited::Vector{Int} = Int[]

    # L é o tipo concreto do GrainLayout — inferido automaticamente na construção.
    # Não há mais ::Any aqui: Julia especializa todas as funções que recebem
    # Propelente{L} e acessam prop.layout.
    layout::L

    # Faces axiais inibidas por grão (campo BATES).
    # 0 → nenhuma inibida: face dianteira (head) E face traseira (aft) queimam
    # 1 → face traseira inibida: só a face dianteira (head-end) queima
    # 2 → ambas inibidas: nenhuma face queima (padrão industrial, só superfície lateral)
    # 3 → face dianteira inibida: só a face traseira (aft/tubeira) queima
    inhibited_ends::Int64 = 2

    # Tipo de geometria do grão — usado apenas para lógica especial de queima.
    # :standard  → comportamento padrão (BATES, finocyl, star, moonburner)
    # :end_burner → sem regressão radial; área de face = A_ext (secção plena)
    grain_type::Symbol = :standard

    # Erosiva / térmica adicional
    # alpha_e: coeficiente Lenoir-Robillard em unidades SI [m^2.8 / (kg^0.8 · s^0.2)]
    # Conversão CGS→SI: alpha_e[SI] = alpha_e[CGS] × 10^(-3.2) ≈ alpha_e[CGS] × 6.31e-4
    # Valores típicos HTPB/AP: alpha_e_CGS ≈ 0.03–0.08  →  alpha_e_SI ≈ 2e-5 – 5e-5
    # ATENÇÃO: valores > 1e-4 causam runaway de pressão neste motor.
    alpha_e::Float64 = 0.5e-5   # realista (corcunda ~40%); literatura 2e-5-5e-5 super-prediz em L/D alto — calibrar
    beta_e::Float64  = 0.5   # coeficiente de blowing (adimensional); 0 desativa blowing
    sigma_p::Float64 = 0.0025
    T_ref::Float64   = 298.15
    T_grain::Float64 = 298.15
end

# ============================================================
# TUBEIRA / EROSÃO
# ============================================================

Base.@kwdef struct NozzleParams
    eta_div::Float64 = 0.985
    erosao_ref::Float64   = 1.5 / 1000.0
    P_ref_erosao::Float64 = 3.5e6
end

# ============================================================
# CONFIGURAÇÃO DO MODELO
# ============================================================

"""
    ConfigModelo

Configuração numérica e física do solver 1D.

# Campos principais
- `cfl=0.40`           : número de Courant–Friedrichs–Lewy (0.1–0.50); MUSCL-HLLC estável até ~0.50
- `n_geom_skip=100`    : atualiza geometria da câmara a cada N passos
- `n_hist_skip=1000`   : grava histórico temporal a cada N passos
- `usar_faces_axiais=true`: considera queima das faces dos grãos BATES
- `modo_silencioso=false` : suprime saída no terminal (útil para Monte Carlo)

# Burnout / tail-off
- `P_tailoff=315_000.0` : pressão de apagamento [Pa]  (~3.1 atm)
- `t_min_tailoff=2.0`   : ignora tail-off nos primeiros N segundos

# Exemplo
```julia
cfg = ConfigModelo(cfl=0.25, n_geom_skip=100, n_hist_skip=200, modo_silencioso=true)
```
"""
Base.@kwdef struct ConfigModelo
    # Região ativa de combustão
    usar_queima_somente_camara::Bool = true

    # Faces axiais
    usar_faces_axiais::Bool    = true
    f_faces_axiais::Float64    = 1.0
    f_regressao_axial::Float64 = 1.0
    # Largura [m] do "smear" axial da área da face do grão BATES. A área de cada
    # face seria distribuída por um box desta largura no INTERIOR do grão. 0.0 =
    # injeção conservativa (área da face no único cell da posição da face).
    #
    # DEFAULT 0.0 (era 0.015): o smear era NÃO-CONSERVATIVO — distribuía a área da
    # face por células interiores usando o ANEL LOCAL de cada célula (A_secao−A_port
    # do y_queima daquela célula), em vez do anel na POSIÇÃO da face. Isso injetava
    # mais massa do que a geometria removia (over-burn de face aberta: balanço 1.00→
    # 1.03 em N≥50, impulso +4.5%, mesh-dependente). Medição (2xBATES Ø96/40):
    # h_smear=0 conserva (balanço 0.99, impulso mesh-independente ~8100 N·s) SEM
    # piorar a rugosidade da pressão (idêntica em N=30..120). Ver VALIDACAO.md §7.3.
    # Internamente (se >0): h_smear = max(h_smear_faces_m, 3·dx).
    h_smear_faces_m::Float64   = 0.0

    # Ignição / aquecimento
    ganho_termico_ignitor::Float64 = 300.0
    coef_convectivo_base::Float64  = 0.023
    # Duração da rampa linear de ignição: m_dot cresce de 0→1 em t_ramp_ignicao [s].
    # Também define a janela de dt inicial conservador (ver dt_max_startup).
    t_ramp_ignicao::Float64  = 0.005     # [s]  — padrão: 5 ms
    dt_max_startup::Float64  = 1e-6      # [s]  — dt máximo durante a rampa de ignição
    # Flame-spread (ignição em tempo finito): false (default) = ignição INSTANTÂNEA de toda
    # a câmara (baseline V&V, = OpenMotor). true = só a região do ignitor (x ≤ posicao_final)
    # acende no t=0; o resto do grão acende por aquecimento convectivo (physics.jl, já existe)
    # → frente de chama com velocidade finita → transiente de ignição físico (subida de P gradual).
    usar_flame_spread::Bool  = false
    # Velocidade da frente de chama [m/s] (flame-spread): a chama avança do ignitor a esta
    # velocidade, acendendo as células que alcança. Típico p/ APCP: ~1–30 m/s (cresce com P).
    # Modelo explícito e robusto (o aquecimento convectivo puro estagna a baixa pressão).
    v_flame_spread::Float64  = 5.0

    # Física adicional
    usar_erosiva::Bool         = false
    usar_erosao_garganta::Bool = false
    # Limiar de fluxo de massa abaixo do qual a erosão NÃO é aplicada.
    # Valores típicos: 150–300 kg/(m²·s). Previne augmentação irreal durante
    # a fase do ignitor (onde G pode atingir 1000+ kg/(m²·s) artificialmente).
    G_erosao_lim::Float64 = 150.0
    # Teto da AUGMENTAÇÃO erosiva: r_total = r_ref + min(r_ero, cap·r_ref).
    # 3.0 = padrão (limita a erosiva a 3× a taxa base). Aumentar para sondar se o
    # pico de pressão satura (cap OK) ou dispara (cap escondendo o pico real).
    cap_erosivo_mult::Float64 = 3.0
    # ── Modelo de queima erosiva ──────────────────────────────────────────────
    # Correlação UNIVERSAL de Mukunda-Paul (1997): sem calibração, independente
    # de propelente (±10%, Al até 17%). Duas variantes na escala do porto d0:
    # :mukunda     → mod. finocyl (Mukunda et al. 2014): d0 = P/π. ← DEFAULT.
    # :mukunda_std → padrão: d0 = D_h = 4A/P (para grão cilíndrico; ≡ finocyl aí).
    # :lenoir      → Lenoir-Robillard (α_e/β_e; legado, fora da GUI). Ainda suportado.
    modelo_erosivo::Symbol = :mukunda
    mu_gas::Float64        = 9.0e-5   # viscosidade do gás [Pa·s] (Re0 do Mukunda; AP/HTPB ~2600 K)
    g_th_mukunda::Float64  = 35.0     # limiar universal do g adimensional (Mukunda-Paul, Eq. 12)
    # Teto do fator de correção de porto do D₄₃ (perda bifásica). Padrão 1.5.
    # f_porto = clamp((L/Dt)^0.35, 1.0, cap_f_porto). Maior = perda bifásica maior
    # (D₄₃ maior); menor = perda menor. Não-validado → calibrar via teste sub-escala.
    cap_f_porto::Float64 = 1.5
    # ── Retenção de escória (slag) de Al₂O₃ ───────────────────────────────────
    # Fração da Al₂O₃ produzida que fica RETIDA no motor (poça no domo aft, adere
    # à parede) e NÃO sai pela tubeira → perde impulso E vira massa inerte de
    # burnout. Empírico e MUITO dependente da geometria: tubeira submersa, alta
    # razão de aspecto e spin aumentam o slag. Faixa típica 0–0.10 (até maior em
    # submersa/spin). `frac_slag = 0` (default) → sem penalidade (baseline). A
    # massa de Al₂O₃ = frac_alumina·(102/54)·m_prop (2Al+1.5O₂→Al₂O₃, 54→102 g).
    # η_slag ≈ 1 − frac_slag·frac_alumina·(102/54); entra em η_tot (não em η_2ph).
    # Referências: Sutton&Biblarz cap.12; Salita AIAA-95-2706.
    frac_slag::Float64 = 0.0
    # ── Modelo de duas fases ──────────────────────────────────────────────────
    # false (default) → η_2ph pelo modelo ESCALAR calibrado (calcular_2fases_fisica).
    # true            → integra o escoamento bifásico ACOPLADO gás–partícula no
    #   divergente real (TwoPhaseNozzle.jl): partícula Lagrangiana (Schiller–Naumann
    #   + Ranz–Marshall), τ_res da geometria (não calibrado), enquadrado por
    #   congelado/equilíbrio. Sensível à tubeira. Fallback ao escalar se falhar.
    usar_2fases_acoplado::Bool = false
    # Distribuição de tamanho de partícula (log-normal, NASA SP-8039) no bifásico
    # acoplado: em vez de um d43 único, discretiza uma log-normal de mass-mean=d43 em
    # N classes e faz a média MÁSSICA da perda (a cauda de partículas grandes puxa a
    # perda p/ cima). Só tem efeito com usar_2fases_acoplado=true. σ_g = desvio-padrão
    # geométrico da distribuição (1.0 = monodisperso → recupera o d43 único).
    usar_distribuicao_particula::Bool = false
    sigma_g_particula::Float64 = 1.8

    # Numérico
    cfl::Float64 = 0.40   # MUSCL-HLLC Van Leer: estável até ~0.50; 0.40 = ~33% mais rápido vs 0.30
    usar_limitador_minmod::Bool = true
    # Frequências internas (passos do solver):
    n_geom_skip::Int = 100    # atualiza geometria da câmara a cada N passos (100 ≈ 2×mais rápido vs 50)
    n_hist_skip::Int = 1000   # grava histórico (P, F, …) a cada N passos

    # Tail-off / burnout
    # Critério: t > t_min_tailoff E P_câmara < P_tailoff → fim de queima.
    # t_min_tailoff evita falsos positivos durante a rampa inicial de pressão.
    P_tailoff::Float64      = 315_000.0  # [Pa]  — ~3.1 atm (pressão de apagamento)
    t_min_tailoff::Float64  = 2.0        # [s]   — ignora tail-off nos primeiros 2 s
    limiar_burnout_area_rel::Float64 = 0.15
    limiar_burnout_mdot_rel::Float64 = 0.10
    limiar_burnout_empuxo_rel::Float64 = 0.05
    tempo_confirmacao_burnout::Float64 = 0.03

    # Fallback / estabilidade
    limiar_fallback_gradp::Float64 = 0.8
    limiar_fallback_temp::Float64  = 4000.0
    limiar_fallback_p::Float64     = 0.8

    # ── Descarga sônica analítica na garganta ─────────────────────────────────
    # A garganta ocupa 1–2 células na malha uniforme → a transição sônica fica
    # sub-resolvida e o HLLC DESCARREGA DEMAIS (c*_efetivo baixo), suprimindo a
    # pressão de câmara ~5–15% em N prático (converge p/ 0D só com N muito alto).
    # Correção: no rosto da garganta impõe-se o fluxo de escoamento ESTRANGULADO
    # (ṁ=ρ*·a*) calculado a partir do P₀ LOCAL que o próprio 1D obtém a montante
    # — física correta de choke, não imposição do 0D. Só age quando realmente
    # estrangulado (P₀ > crítico e fluxo para +x); ignição/tail-off usam HLLC.
    # O campo 1D (gradiente, transiente, erosiva) continua sendo resolvido; só o
    # NÍVEL de pressão passa a fazer sentido físico (fica próximo do 0D validado).
    usar_garganta_sonica::Bool = true

    # ── Termo de área bem-balanceado (well-balanced) ──────────────────────────
    # O termo P·dA/dx discretizado como p_i·(A_{i+1/2} − A_{i−1/2}) com estados
    # de face reconstruídos em variáveis primitivas NÃO preserva o escoamento
    # isentrópico permanente quando a área muda bruscamente numa célula (degraus
    # do canal na transição cilindro/aletas, fim do grão, convergente, divergente):
    # a pressão de estagnação ganha/perde alguns % de forma espúria (medido: −4%
    # no canal no início da queima contra 0,3% da teoria; −7% no divergente).
    # Com true, os estados de face são reconstruídos pelos invariantes do
    # escoamento isentrópico (ṁ, H, K=p/ρ^γ) e projetados para a área da face, e o
    # termo de área vira ∫p·dA ao longo da isentrópica da própria célula
    # (Solver1D.termo_area_momento). Escoamento isentrópico permanente passa a ser
    # solução EXATA do esquema discreto; repouso continua exato; massa e energia
    # continuam conservativas. false = esquema antigo (só p/ comparação/V&V).
    usar_termo_area_wb::Bool = true

    # ── Atrito de parede na tubeira ───────────────────────────────────────────
    # O atrito de Darcy (Churchill, escoamento plenamente desenvolvido em tubo)
    # é um modelo de CANAL. Aplicado no divergente supersônico, derrubava p₀ em
    # ~10% (verificado contra a solução quase-1D com o mesmo atrito) e somava-se
    # ao η_bl, que já representa a camada limite da tubeira (perda contada duas
    # vezes). Como no NASA SP-8039 (§2.1.3.2.1.4), a tubeira é tratada como
    # escoamento sem atrito e a perda viscosa entra como perda de camada limite
    # no empuxo (η_bl). false = atrito só na câmara (padrão); true = antigo.
    usar_atrito_tubeira::Bool = false

    # ── Perfil axial num nível de pressão escolhido ──────────────────────────
    # > 0: grava o perfil completo (câmara + tubeira: A, perímetro de queima,
    # vazão injetada, ρ, u, P, T, M, P₀) na PRIMEIRA vez que a pressão média da
    # câmara atinge este valor [MPa], junto com t, P_cabeça, ṁ de saída, ṁ
    # gerada e empuxo nesse instante → "perfil_alvo_<caso>.csv" (com salvar_csv).
    # Serve para verificar o 1D contra a solução quase-1D num instante fixo.
    # 0 = desligado.
    P_perfil_alvo_MPa::Float64 = 0.0

    # Modo de simulação:
    #   :um_d    → solver 1D MUSCL-HLLC (padrão) — resultado completo com perfis axiais
    #   :zero_d  → modelo quasi-estático 0D (parâmetros concentrados) — ~1000× mais rápido
    modo_simulacao::Symbol = :um_d

    # ── Splitting de operadores ───────────────────────────────────────────────
    # usar_strang_splitting = false (padrão) → SSP-RK2 acoplado (como antes)
    # usar_strang_splitting = true           → Strang splitting 2ª ordem:
    #
    #   S(dt/2)  →  L(dt)  →  S(dt/2)
    #
    # onde  S = fonte de combustão (r·ρ_p·A_b, h0)  +  avanço do sólido (y_queima)
    #       L = transporte hiperbólico (HLLC) + atrito
    #
    # QUANDO USAR:  motores com L/D elevado onde se pretende capturar instabilidade
    # acústica. No modo acoplado padrão, o termo de combustão usa r(P^n) em AMBOS os
    # estágios do RK2; com Strang o estágio final usa r(P^(n+1)), dando acoplamento
    # acústica–combustão de 2ª ordem em tempo. Custo: ~25 % mais lento por passo.
    usar_strang_splitting::Bool = false

    # Monte Carlo — suprime println/CSV durante corridas paralelas
    modo_silencioso::Bool = false

    # Callback de progresso (opcional). Se definido, o loop 1D o chama
    # periodicamente com (t_atual, t_maximo) — a GUI usa p/ a barra de progresso.
    # `nothing` (padrão) = no-op: scripts/testes/headless não pagam nada.
    progress_cb::Union{Function, Nothing} = nothing
end

# ============================================================
# RESULTADO DE SIMULAÇÃO
# ============================================================

"""
    SimulationResult

Resultado completo de uma simulação RktPrisma retornado por [`simular_caso`](@ref).

# Históricos temporais
- `tempos`    : vetor de tempos [s]
- `pressoes`  : pressão na câmara em cada instante [MPa]
- `empuxos`   : empuxo líquido em cada instante [N]  (ao nível do mar)

# Métricas integradas
- `t_burn`      : tempo de queima [s]
- `I_total`     : impulso total [N·s]  (integração trapezoidal)
- `Isp`         : impulso específico [s]  (= I_total / m_consumida / g₀)
- `m_consumida` : massa de propelente consumida [kg]

# Pressão e empuxo (médias filtradas — exclui transiente de ignição)
- `P_max`, `P_avg` : pressão máxima e média [MPa]
- `F_max`, `F_avg` : empuxo máximo e médio [N]
- `CF_avg`         : coeficiente de empuxo médio ao nível do mar [-]

# Klemmung
- `Kn_ini`, `Kn_max` : Kn inicial e máximo [-]

# Qualidade numérica
- `n_fallbacks` : número de fallbacks numéricos do solver

# Saída
- `csv_path` : caminho do CSV de resultados salvo (vazio se `salvar_csv=false`)

## Compatibilidade retroativa
Suporta desestruturação para manter scripts existentes sem alteração:
```julia
tempos, pressoes, empuxos = simular_caso(inp; cfg=cfg)
```
"""
struct SimulationResult
    caso         ::String

    tempos       ::Vector{Float64}   # [s]
    pressoes     ::Vector{Float64}   # [MPa]  pressão na câmara
    empuxos      ::Vector{Float64}   # [N]    empuxo líquido (nível do mar)

    t_burn       ::Float64           # [s]
    I_total      ::Float64           # [N·s]
    Isp          ::Float64           # [s]
    m_consumida  ::Float64           # [kg]

    P_max        ::Float64           # [MPa]
    P_avg        ::Float64           # [MPa]  (filtrado)
    F_max        ::Float64           # [N]
    F_avg        ::Float64           # [N]    (filtrado)
    CF_avg       ::Float64           # [-]

    Kn_ini       ::Float64           # [-]
    Kn_max       ::Float64           # [-]

    n_fallbacks  ::Int
    csv_path     ::String

    # ── Fatores de correção de desempenho (calculados em CaseRunner) ──────────
    #
    # η_div   : eficiência de divergência cônica  λ = (1 + cos α_div) / 2
    #           Perda típica: ~1.7 % para α=15°, ~3 % para α=20°
    #
    # η_2ph   : eficiência de duas fases (partículas Al₂O₃)
    #           Modelo físico de Stokes (Lengellé-Hermsen), com D₄₃ auto-predito
    #           via Hermsen (1981) quando d_p_alumina_um = 0 em CaseInput.
    #           Perda típica: ~3–5 % para 16 % Al
    #
    # η_total : η_2ph somente (η_div e η_bl já aplicados no CFD 1D)
    #
    # Isp_corrigido    = Isp    × η_total   [s]
    # I_total_corrigido= I_total × η_total  [N·s]
    # F(t)_corrigido   = F(t)   × η_total  → usado no export .eng
    #
    # d43_um  : diâmetro D₄₃ das partículas Al₂O₃ usado no cálculo de η_2ph [µm]
    #           0.0 → frac_alumina = 0 ou modelo escalar antigo (sem Stokes)
    eta_div           ::Float64   # divergência cônica (0–1)
    eta_2ph           ::Float64   # duas fases Al₂O₃  (0–1)
    eta_total         ::Float64   # combinado          (0–1)
    Isp_corrigido     ::Float64   # [s]
    I_total_corrigido ::Float64   # [N·s]
    d43_um            ::Float64   # [µm]  D₄₃ Al₂O₃ (0 = sem alumínio)

    # ── Erosão da garganta ────────────────────────────────────────────────────
    D_garganta_final  ::Float64          # [mm]  diâmetro final após queima
    erosao_radial_mm  ::Float64          # [mm]  erosão radial total acumulada
    hist_D_garganta   ::Vector{Float64}  # [mm]  D_garganta(t) ao longo da queima

    kn_hist           ::Vector{Float64}  # Kn(t) = Ab(t)/At — mesmo eixo temporal de tempos

    # ── Porteira de MEOP (Maximum Expected Operating Pressure) ────────────────
    # Verificação de segurança: P_max simulada vs MEOP especificada pelo usuário.
    #   P_meop = 0.0  → verificação desativada (CaseInput.P_meop_MPa = 0)
    #   FS_meop = MEOP / P_max  (margem; < 1.0 → SOBREPRESSÃO → risco de ruptura)
    # Preenchidos pelo CaseRunner; placeholder (0.0, Inf) nos construtores internos.
    P_meop            ::Float64          # [MPa]  MEOP especificada (0 = não verificada)
    FS_meop           ::Float64          # [-]    MEOP / P_max  (Inf se não verificada)
end

# Desestruturação retroativa: tempos, pressoes, empuxos = resultado
Base.iterate(r::SimulationResult, s=1) =
    s == 1 ? (r.tempos,   2) :
    s == 2 ? (r.pressoes, 3) :
    s == 3 ? (r.empuxos,  nothing) : nothing
Base.length(::SimulationResult) = 3

function Base.show(io::IO, r::SimulationResult)
    println(io, "SimulationResult — $(r.caso)")
    @printf(io, "  t_burn          = %.3f s\n",   r.t_burn)
    @printf(io, "  Isp (s/ 2-fases)= %.1f s\n",   r.Isp)
    @printf(io, "  Isp corrigido   = %.1f s",     r.Isp_corrigido)
    if r.eta_total < 0.9999
        @printf(io, "  (η_div=%.4f já no empuxo; η_2ph=%.4f → η_corr=%.4f)\n",
                r.eta_div, r.eta_2ph, r.eta_total)
    else
        println(io)
    end
    @printf(io, "  I_total (s/2-fases)= %.1f N·s\n", r.I_total)
    @printf(io, "  I_total corrig. = %.1f N·s\n", r.I_total_corrigido)
    @printf(io, "  P_max           = %.3f MPa\n", r.P_max)
    if r.P_meop > 0.0
        if r.FS_meop < 1.0
            @printf(io, "  ⛔ MEOP EXCEDIDA  = %.3f MPa  (P_max=%.3f → FS=%.2f < 1.0 — RISCO DE RUPTURA)\n",
                    r.P_meop, r.P_max, r.FS_meop)
        else
            @printf(io, "  ✅ MEOP OK        = %.3f MPa  (FS = %.2f)\n", r.P_meop, r.FS_meop)
        end
    end
    @printf(io, "  F_max           = %.1f N\n",   r.F_max)
    @printf(io, "  m_consumida     = %.4f kg\n",  r.m_consumida)
    @printf(io, "  Kn_max          = %.1f\n",     r.Kn_max)
    if r.d43_um > 0.0
        @printf(io, "  D₄₃ Al₂O₃      = %.1f µm\n",  r.d43_um)
    end
    @printf(io, "  fallbacks       = %d\n",       r.n_fallbacks)
    if r.erosao_radial_mm > 0.001
        @printf(io, "  D_garg ini→fin  = %.2f → %.2f mm  (erosão radial: %.3f mm)\n",
                r.D_garganta_final - 2*r.erosao_radial_mm,
                r.D_garganta_final, r.erosao_radial_mm)
    end
end

# ============================================================
# IGNITOR
# ============================================================

Base.@kwdef struct Ignitor
    # Fase 2: API física directa — utilizador especifica pico de fluxo e janela temporal.
    # Perfil sinusoidal: m_dot(t) = m_dot_ign × sin(π·(t−t_start)/t_dur)  para t ∈ [t_start, t_end]
    # Massa total integrada = m_dot_ign × (t_end − t_start) × 2/π
    posicao_final ::Float64 = 0.15    # posição axial do fim do ignitor [m]
    m_dot_ign     ::Float64 = 0.05    # PICO de fluxo mássico do ignitor [kg/s] (média temporal = 2/π·pico ≈ 64%)
    t_ign_start   ::Float64 = 0.0     # instante de início da ignição [s]
    t_ign_end     ::Float64 = 0.010   # instante de fim da ignição [s]
    T_ign         ::Float64 = 2500.0  # temperatura dos gases do ignitor [K]
    gamma_ign     ::Float64 = 1.235             # γ dos gases do ignitor (Fase 0.2; default = APCP → comportamento atual)
    R_ign         ::Float64 = 8314.46 / 22.959  # R específico dos gases do ignitor [J/(kg·K)] (Fase 0.2)
end

# ============================================================
# ESPECIFICAÇÃO DE GEOMETRIAS
# ============================================================

abstract type AbstractGeometrySpec end

Base.@kwdef struct CylinderSpec <: AbstractGeometrySpec
    D_ext::Float64
    D_core::Float64
end

Base.@kwdef struct FinocylSpec <: AbstractGeometrySpec
    D_ext::Float64
    D_core::Float64
    n_fins::Int
    fin_width::Float64
    fin_length::Float64
    inverted::Bool = false
    # Fração axial do grão que contém as aletas (0 < fin_fraction <= 1.0).
    # Quando < 1.0, o restante do grão é tratado como cilindro puro (D_core).
    # Default = 1.0 mantém o comportamento original (aletas em 100% do comprimento).
    fin_fraction::Float64 = 1.0
    # Shape do slot: :rectangular (padrão), :capsule (oblongo, extremidades arredondadas),
    # :rounded_tip (retangular + semicírculo só na ponta).
    slot_shape          ::Symbol  = :rectangular
    # Raio de arredondamento da ponta [m]. 0 → automático (W_slot/2 para :capsule).
    fin_tip_radius      ::Float64 = 0.0
    # Raio de fillet na raiz do slot [m]. 0 → sem fillet.
    fin_root_radius     ::Float64 = 0.0
    # Resolução do buffer LibGEOS (segmentos por quadrante de curva).
    geometry_resolution ::Int     = 64
end

Base.@kwdef struct StarSpec <: AbstractGeometrySpec
    D_ext::Float64
    pontas::Int = 5
    raio_base::Float64
    raio_fenda::Float64
    angulo_fenda::Float64 = 0.4
    passos::Int = 400
end

Base.@kwdef struct BatesSpec <: AbstractGeometrySpec
    D_ext::Float64
    D_core::Float64
    inhibited_ends::Int = 2
end

# ──────────────────────────────────────────────────────────────────────────────
# End-Burner: queima apenas pela face traseira (sem porto interno).
# A_port = A_ext (bore pleno), P_burn = 0 (sem queima lateral).
# Tudo é controlado pela regressão axial via y_axial_faces (inhibited_ends = 3).
# ──────────────────────────────────────────────────────────────────────────────
Base.@kwdef struct EndBurnerSpec <: AbstractGeometrySpec
    D_ext::Float64
end

# ──────────────────────────────────────────────────────────────────────────────
# Moon-Burner (crescente): porto circular deslocado do eixo do grão.
# A excentricidade cria queima progressiva: a área de queima P_burn(y) cresce
# até o porto atingir a parede mais próxima, depois cai.
# Requer LibGEOS (mesmo pipeline do finocyl).
# ──────────────────────────────────────────────────────────────────────────────
Base.@kwdef struct MoonBurnerSpec <: AbstractGeometrySpec
    D_ext::Float64
    D_core::Float64        # diâmetro inicial do porto [m]
    eccentricity::Float64  # deslocamento do centro do porto em relação ao eixo [m]
                           # deve satisfazer:  eccentricity + D_core/2  <  D_ext/2
    passos::Int = 200      # resolução da LUT (pontos de y varridos)
end

# ──────────────────────────────────────────────────────────────────────────────
# Wagon-Wheel (roda): furo central + N fendas radiais FUNDAS (raios finos entre
# elas), atingindo perto da carcaça. Alta área de queima inicial → alto empuxo,
# quase-neutro. Mesma máquina do finocyl (fendas), mas fenda funda permitida.
# ──────────────────────────────────────────────────────────────────────────────
Base.@kwdef struct WagonWheelSpec <: AbstractGeometrySpec
    D_ext::Float64
    D_core::Float64          # furo do cubo (hub) [m]
    n_spokes::Int            # nº de fendas radiais (= nº de raios)
    slot_width::Float64      # largura de cada fenda [m]
    slot_length::Float64     # profundidade radial da fenda [m] (funda: perto da carcaça)
    slot_shape::Symbol = :rectangular
    geometry_resolution::Int = 64
end

# ──────────────────────────────────────────────────────────────────────────────
# Multi-perfurado: N perfurações circulares (1 central + N−1 num círculo de passo).
# Estilo grão de canhão (7/19-perf). Void = união de N círculos.
# ──────────────────────────────────────────────────────────────────────────────
Base.@kwdef struct MultiPerfSpec <: AbstractGeometrySpec
    D_ext::Float64
    n_perf::Int              # nº total de perfurações (>=1; 1 = cilindro central)
    D_perf::Float64          # diâmetro de cada perfuração [m]
    pitch_frac::Float64 = 0.5  # raio do círculo de passo das perifs, como fração de R_ext
    passos::Int = 300
end

# ──────────────────────────────────────────────────────────────────────────────
# GrainSpec — geometria de UM grão num stack heterogêneo (estilo OpenMotor).
# Cada grão é configurado por vez e empilhado na ordem (cabeça→tubeira) via
# CaseInput.grains. Propelente/tubeira/sim são compartilhados (ficam no CaseInput).
# v1: grãos RADIAIS (bates/finocyl/star/moonburner) com faces governadas pelo
# inhibited_ends GLOBAL do CaseInput. (end-burner no stack + face por-grão = v2.)
# ──────────────────────────────────────────────────────────────────────────────
Base.@kwdef struct GrainSpec
    tipo::Symbol = :bates           # :bates, :finocyl, :star, :moonburner
    L::Float64                      # comprimento do segmento [m]
    D_ext::Float64                  # diâmetro externo [m]
    D_core::Float64 = 0.0           # furo central [m] (bates/finocyl/moonburner)
    inhibited_ends::Int = 0         # faces deste grão: 0=ambas queimam, 1=só dianteira, 2=nenhuma, 3=só traseira
    # — finocyl —
    n_fins::Int = 0
    fin_width::Float64 = 0.0
    fin_length::Float64 = 0.0
    fin_fraction::Float64 = 1.0
    slot_shape::Symbol = :rectangular
    fin_tip_radius::Float64 = 0.0
    fin_root_radius::Float64 = 0.0
    geometry_resolution::Int = 64
    # — star —
    pontas::Int = 0
    raio_base::Float64 = 0.0
    raio_fenda::Float64 = 0.0
    angulo_fenda::Float64 = 0.4
    passos_geometria::Int = 300
    # — moonburner —
    eccentricity::Float64 = 0.0
end

# ============================================================
# ENTRADA DE CASO (INTERFACE SIMPLES)
# ============================================================

"""
    CaseInput

Interface de alto nível para definir um caso de simulação RktPrisma.
Passado diretamente para [`simular_caso`](@ref) e [`diagnostico_geometria`](@ref).

# Campos obrigatórios
- `name`           : identificador do caso (usado em nomes de arquivo CSV)
- `geometry_type`  : `:bates`, `:finocyl` ou `:star`
- `D_ext`, `L_grao`: diâmetro externo do grão [m] e comprimento [m]
- `D_core`         : diâmetro do porto interno [m]  (`:bates` e `:finocyl`)
- `rho_p`, `a`, `n`: densidade [kg/m³], coef. burn rate [m/s/Paⁿ], expoente
- `Tc`, `gamma`, `R`: temperatura de chama [K], razão de calores, constante de gás [J/kg/K]
- `x_garganta`    : posição axial da garganta [m]
- `D_garganta_ini`: diâmetro da garganta [m]
- `D_saida`       : diâmetro de saída da tubeira [m]
- `L_total`       : comprimento total do domínio 1D [m]

# Campos para `:finocyl`
- `n_fins`, `fin_width`, `fin_length`: número de fins, largura [m], comprimento radial [m]
- `fin_fraction`  : fração axial do grão com fins (0 < f ≤ 1; padrão=1.0)
- `slot_shape`    : `:rectangular` (padrão), `:capsule`, `:rounded_tip`

# Campos numéricos
- `N_malha`  : número de células na malha 1D (padrão=50; recomendado ≥100)
- `t_maximo` : tempo máximo de simulação [s]

# Exemplo
```julia
inp = CaseInput(
    name = "motor_A", geometry_type = :finocyl,
    D_ext = 0.096, D_core = 0.020, L_grao = 0.398,
    n_fins = 6, fin_width = 0.005, fin_length = 0.020,
    fin_fraction = 0.30, slot_shape = :rectangular,
    rho_p = 1624.0, a = 6.99e-5, n = 0.321,
    Tc = 2616.5, gamma = 1.235, R = 362.1,
    x_garganta = 0.598, D_garganta_ini = 0.025,
    D_saida = 0.050, L_total = 0.948,
    N_malha = 100, t_maximo = 10.0,
)
```
"""
Base.@kwdef struct CaseInput
    name::String
    geometry_type::Symbol   # :bates, :star, :finocyl, :end_burner, :moonburner

    # -------------------------
    # Geometria do grão
    # -------------------------
    D_ext::Float64
    L_grao::Float64
    N_graos::Int = 1

    # Usado por :bates e :finocyl
    D_core::Float64 = 0.0

    # Furo AFT para :finocyl_conico — furo CÔNICO: D_core (cabeça) → D_core_aft (bocal).
    # A seção de furo maior queima até a parede ANTES (teia menor) → sua área some,
    # o que achata a subida progressiva → platô mais neutro (ideia do booster cônico).
    # 0.0 = não cônico (usa D_core). Só relevante para geometry_type = :finocyl_conico.
    # Para :finocyl_cono_cil, D_core_aft = furo na EXTREMIDADE livre do trecho CILÍNDRICO
    # (o aletado fica em D_core constante; só o cilindro afila de D_core → D_core_aft).
    D_core_aft::Float64 = 0.0

    # Orientação do :finocyl_cono_cil: true = trecho ALETADO junto ao BOCAL (aft, padrão);
    # false = aletado na CABEÇA (fore) e cilíndrico cônico voltado ao bocal.
    # No 0D não muda a balística (lumped); afeta 1D/térmica/desenho.
    finocyl_no_bocal::Bool = true

    # Faces axiais inibidas por grão BATES (0, 1, 2 ou 3). Default = 2 (ambas inibidas,
    # comportamento padrão OpenMotor "inhibited"). Só relevante para :bates.
    # 0=nenhuma, 1=só face dianteira queima, 2=ambas inibidas, 3=só face traseira queima
    inhibited_ends::Int = 2

    # Usado por :finocyl
    n_fins::Int = 0
    fin_width::Float64 = 0.0
    fin_length::Float64 = 0.0
    # Fração axial com aletas (0 < fin_fraction <= 1.0).
    # Ex.: 0.30 → aletas nos primeiros 30% do grão, restante cilíndrico.
    # Default = 1.0 (aletas em todo o comprimento, comportamento original).
    fin_fraction::Float64 = 1.0
    # Fração do comprimento do grão usada como zona de transição cônica
    # entre cilindro e finocyl. Ex.: 0.05 → 5% do grão (~2cm). 0.0 = degrau abrupto.
    L_trans_frac::Float64 = 0.05
    # ── Afilamento axial das aletas (fin taper) ──────────────────────────────
    # Dois modos disponíveis (mutuamente exclusivos):
    #
    # MODO A — Discreto (fin_taper_n_segs > 1):
    #   N sub-segmentos com fin_length crescente na seção aft inteira.
    #   fin_taper_n_segs = 1  → sem afilamento (comportamento original)
    #   fin_taper_start_frac  → fração de fin_length na borda mais próxima do fore.
    #
    # MODO B — Geometria analítica 3D (fin_taper_zone_length > 0):  ← NOVO
    #   Zona curta de chanfro com integral contínua sobre fin_length.
    #   Não tem degraus — a curva de área de queima é suave (C¹).
    #   fin_taper_zone_length : comprimento físico da zona de chanfro [m].
    #     0.0   → modo A (comportamento original)
    #     0.5   → chanfro de 500 mm (típico para motores de ~9 m)
    #   fin_taper_n_segs      : nº de LUTs usados para integração numérica (≥ 5).
    #   fin_taper_start_frac  : fração de fin_length no início do chanfro (0 = zero).
    fin_taper_n_segs     ::Int     = 1
    fin_taper_start_frac ::Float64 = 0.0
    fin_taper_zone_length::Float64 = 0.0   # [m] 0 = modo A; > 0 = modo B (3D analítico)
    # Shape do slot: :rectangular (padrão, compatível), :capsule (oblongo arredondado),
    # :rounded_tip (retangular + semicírculo na ponta).
    slot_shape          ::Symbol  = :rectangular
    # Raio de arredondamento da ponta [m]. 0 → automático (W_slot/2 para :capsule).
    fin_tip_radius      ::Float64 = 0.0
    # Raio de fillet na raiz [m]. 0 → sem fillet.
    fin_root_radius     ::Float64 = 0.0
    # Resolução do buffer LibGEOS (segmentos por quadrante).
    geometry_resolution ::Int     = 64

    # Usado por :moonburner
    # Deslocamento do centro do porto em relação ao eixo do grão [m].
    # Deve satisfazer:  eccentricity + D_core/2  <  D_ext/2
    eccentricity::Float64 = 0.0

    # Usado por :star
    pontas::Int = 0
    raio_base::Float64 = 0.0
    raio_fenda::Float64 = 0.0
    angulo_fenda::Float64 = 0.4
    passos_geometria::Int = 300

    # -------------------------
    # Stack de grãos (estilo OpenMotor) — VAZIO = modo legado de 1 tipo (não quebra
    # nada). NÃO-vazio = grãos empilhados na ordem (cabeça→tubeira); aí geometry_type,
    # D_core e afins acima são ignorados — cada grão vem do seu GrainSpec.
    # -------------------------
    grains::Vector{GrainSpec} = GrainSpec[]

    # -------------------------
    # Termodinâmica / propelente
    # -------------------------
    rho_p::Float64
    a::Float64
    n::Float64
    Tc::Float64
    gamma::Float64
    R::Float64
    eta_cstar::Float64 = 0.98

    # -------------------------
    # Combustão bifásica — partículas Al₂O₃
    # -------------------------
    # Diâmetro médio das partículas de Al₂O₃ em µm.
    # 0.0 → usa o modelo escalar clássico (η = 1 − 0.14·ξ_ox).
    # > 0 → usa modelo físico de drag de Stokes (Lengellé-Hermsen).
    d_p_alumina_um::Float64 = 0.0

    # -------------------------
    # Térmica / erosiva
    # -------------------------
    k_p::Float64 = 0.43
    cp_p::Float64 = 1465.0
    frac_alumina::Float64 = 0.16
    T_ignicao::Float64 = 700.0
    # Lenoir-Robillard (SI): alpha_e típico HTPB/AP ≈ 2e-5 – 5e-5
    # Conversão: alpha_e[SI] = alpha_e[CGS] × 6.31e-4  (fator 10^-3.2)
    alpha_e::Float64 = 0.5e-5   # realista (corcunda ~40%); literatura 2e-5-5e-5 super-prediz em L/D alto — calibrar
    beta_e::Float64  = 0.5
    sigma_p::Float64 = 0.0020
    T_ref::Float64 = 298.15
    T_grain::Float64 = 298.15

    # -------------------------
    # Tubeira / domínio
    # -------------------------
    x_garganta::Float64
    D_garganta_ini::Float64
    D_saida::Float64
    L_total::Float64
    alpha_divergencia::Float64 = 15.0

    # Eficiência de tubeira — derating GLOBAL e OPCIONAL do empuxo (F ← η·F).
    # 1.0 = desligado (default), preservando o comportamento histórico: o código
    # já aplica λ_div (divergência cônica) e η_bl (camada limite) por dentro do
    # Cf, então este fator é um derating ADICIONAL de qualidade, no estilo do
    # campo "Efficiency" do OpenMotor. Não toca em c*, logo não altera a pressão
    # de câmara — só o empuxo. Usar só para casar convenção com outro código.
    eta_tubeira::Float64 = 1.0

    # -------------------------
    # Segurança — MEOP (Maximum Expected Operating Pressure)
    # -------------------------
    # Pressão máxima de operação esperada da carcaça [MPa]. Se > 0, a simulação
    # compara P_max contra ela e sinaliza ⛔ SOBREPRESSÃO quando FS = MEOP/P_max < 1.
    # 0.0 = verificação desativada (default).
    P_meop_MPa::Float64 = 0.0

    # -------------------------
    # Erosão da garganta
    # -------------------------
    # Modelo de potência: r_dot [m/s raio] = erosao_r_dot_ref × (P/P_ref)^erosao_n_exp
    # Taxa radial de referência a erosao_P_ref_MPa. Padrão: grafite denso (~0.15 mm/s a 5 MPa).
    erosao_ativa     ::Bool    = false
    erosao_r_dot_ref ::Float64 = 0.15   # [mm/s] taxa radial a P_ref
    erosao_P_ref_MPa ::Float64 = 5.0    # [MPa]  pressão de referência
    erosao_n_exp     ::Float64 = 0.8    # [-]    expoente de pressão

    # -------------------------
    # Numérico
    # -------------------------
    N_malha::Int = 50
    t_maximo::Float64 = 0.8
end