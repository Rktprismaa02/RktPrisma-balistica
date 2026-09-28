using StaticArrays

# ============================================================
# MÓDULO: FÍSICA E COMBUSTÃO
# Evolução 2: separação entre evolução do sólido e fonte do gás
# Evolução 3: Termodinâmica Variável (Gás Termicamente Perfeito)
# Evolução 4: Lookup Table (LUT) para Otimização Extrema
# ============================================================

# ------------------------------------------------------------
# HELPER PSEUDO-3D: SELETOR DE GEOMETRIA
# ------------------------------------------------------------
@inline function obter_geometria_local(x::Float64, y::Float64, prop::Propelente{L}) where {L}
    A, Pb, _, _ = obter_geom_em_xy(x, y, prop)
    return A, Pb
end

# ------------------------------------------------------------
# (P2 auditoria) Removidas `calcular_Cp_APCP` e `encontrar_T_pela_energia`:
# EOS Cp-polinomial bifásica + Newton de 8 iterações SEM checagem de convergência.
# Eram usadas SÓ no ramo `gamma==0` de criar_tabelas_LUT, que nunca rodava (γ é
# sempre > 0 → EOS de gás perfeito). Código morto — removido. O interpolador de
# produção é encontrar_T_e_Gamma_LUT (abaixo), que opera sobre a LUT GCP.
# ------------------------------------------------------------

# ------------------------------------------------------------
# LUT (LOOKUP TABLE) PARA ACELERAÇÃO
# ------------------------------------------------------------

function criar_tabelas_LUT(R::Float64, frac_alumina::Float64 = 0.0, gamma::Float64 = 1.235)
    # EOS de GÁS CALORICAMENTE PERFEITO (GCP): Cv = R/(γ−1), T = e_int/Cv, γ constante.
    # É o ÚNICO caminho em produção — a EOS resolve apenas o gás (consistente com OpenMotor).
    # A fase condensada (Al₂O₃) NÃO entra no núcleo: entra como correção de Isp/impulso no
    # pós-processamento (CaseRunner: η_2ph, Hermsen+Stokes), coerente entre 0D e 1D. As
    # tabelas T(e)/γ(e) são triviais (γ const) mas mantidas pela interface do interpolador.
    # (P2 auditoria: removido o caminho legado Cp-polinomial + Newton, que era código morto —
    # γ é sempre > 0; ver commit da auditoria.)
    gamma > 0.0 || error("criar_tabelas_LUT: γ=$gamma deve ser > 0 (EOS GCP).")

    E_min = 100_000.0
    E_max = 12_000_000.0
    N_tab = 8000

    E_tab = collect(range(E_min, E_max, length=N_tab))
    T_tab = zeros(N_tab)
    G_tab = zeros(N_tab)

    Cv = R / (gamma - 1.0)
    @inbounds for i in 1:N_tab
        T_tab[i] = max(E_tab[i] / Cv, 100.0)
        G_tab[i] = gamma
    end

    return E_tab, T_tab, G_tab
end

# =========================================================
# INTERPOLADOR ULTRA RÁPIDO DA LUT (Roda milhões de vezes)
# =========================================================
@inline function encontrar_T_e_Gamma_LUT(e_internal, e_tab, T_tab, G_tab)
    # 1. Verificação de Sanidade: Se for NaN ou Inf, não tenta converter
    if !isfinite(e_internal)
        return T_tab[1], G_tab[1]
    end

    N = length(e_tab)
    delta_e = e_tab[2] - e_tab[1]

    # 2. Cálculo do índice base (piso), limitado a [1, N-1]
    idx_float = (e_internal - e_tab[1]) / delta_e + 1.0
    idx = clamp(floor(Int, idx_float), 1, N - 1)

    # 3. Interpolação linear entre idx e idx+1 (elimina erro de quantização)
    @inbounds begin
        frac = (e_internal - e_tab[idx]) / delta_e
        frac = clamp(frac, 0.0, 1.0)
        T = T_tab[idx] + frac * (T_tab[idx + 1] - T_tab[idx])
        G = G_tab[idx] + frac * (G_tab[idx + 1] - G_tab[idx])
    end

    return T, G
end

# (Revisão final) Removida `encontrar_T_pela_energia_LUT` — busca binária de T
# que NÃO era chamada em produção (o interpolador em uso é encontrar_T_e_Gamma_LUT,
# acima). Código morto.



# ------------------------------------------------------------
# INICIALIZAÇÃO E FONTES
# ------------------------------------------------------------

# ATUALIZADO: Remoção do gamma fixo, uso do Cp(T) para energia inicial
function inicializar_estado(malha::Malha1D, P_atm::Float64, T_atm::Float64, R::Float64, gamma_init::Float64 = 1.235)
    N = malha.N
    
    # Otimização 1: Vetores na memória Stack ao invés de matrizes na Heap
    U = zeros(SVector{3, Float64}, N)
    W = zeros(SVector{3, Float64}, N)

    rho_ini = P_atm / (R * T_atm)
    u_ini = 0.0
    p_ini = P_atm
    
    # e_int = Cv*T com Cv = R/(gamma-1) — consistente com LUT GCP
    # gamma padrão 1.235 (APCP); argumento opcional para outros propelentes
    Cv_ini = R / (gamma_init - 1.0)
    E_ini  = Cv_ini * T_atm

    @inbounds for i in 1:N
        W[i] = SVector{3, Float64}(rho_ini, u_ini, p_ini)
        U[i] = SVector{3, Float64}(rho_ini, rho_ini * u_ini, rho_ini * E_ini)
    end

    return U, W
end

function calcular_vazao_ignitor(t::Float64, ig::Ignitor)
    # Fase 2: usa nova API (m_dot_ign directo + janela temporal)
    t_dur = ig.t_ign_end - ig.t_ign_start
    t_rel = t - ig.t_ign_start
    if t_rel < 0.0 || t_rel > t_dur || t_dur <= 0.0
        return 0.0
    end
    return ig.m_dot_ign * sin(π * t_rel / t_dur)
end

# ------------------------------------------------------------
# AUXILIARES TERMODINÂMICOS
# ------------------------------------------------------------

# h0 = ENTALPIA específica do propelente injetado = Cp * Tc
# A equação de energia do Euler conserva E_total = rho*(e_int + u²/2).
# O termo fonte injeta: dE_total/dt = m_dot * h0, onde h0 = entalpia = Cp*Tc.
# Com Cp = gamma*R/(gamma-1) (GCP consistente com a LUT):
#   Balanço em regime: m_dot_in * Cp*Tc = m_dot_out * Cp*T_cam => T_cam = Tc ✓
#   c* = sqrt(R*Tc) / Gamma_func ← igual ao OpenMotor ✓
# eta_cstar reduz Tc efetiva: Tc_eff = Tc * eta²
function calcular_h0_propelente(prop::Propelente{L}) where {L}
    Cp_consist = prop.gamma * prop.R / (prop.gamma - 1.0)
    Tc_eff     = prop.Tc * prop.eta_cstar^2
    return Cp_consist * Tc_eff
end

function calcular_h0_ignitor(ig::Ignitor)
    # Fase 0.2: γ e R do ignitor agora são campos do struct (default = APCP).
    Cp_ign = ig.gamma_ign * ig.R_ign / (ig.gamma_ign - 1.0)
    return Cp_ign * ig.T_ign
end

@inline function calcular_propriedades_escoamento_local(
    rho::Float64,
    u::Float64,
    P::Float64,
    Area_port::Float64,
    Perimetro_port::Float64,
    prop::Propelente{L}       # Fase 2 (acede só a scalares Float64, mas consistente)
) where {L}
    # ------------------------------------------------------------
    # 1. PROTEÇÕES NUMÉRICAS BÁSICAS
    # ------------------------------------------------------------
    rho_eff = max(rho, 1e-6)
    P_eff = max(P, 1e-2)
    A_eff = max(Area_port, 1e-8)
    Pm_eff = max(Perimetro_port, 1e-8)

    # ------------------------------------------------------------
    # 2. TEMPERATURA LOCAL DO GÁS  (B10 fix — dupla contagem de (1−χ))
    # ------------------------------------------------------------
    # No modo GCP o core resolve APENAS o gás: P = ρ·R·T com R CHEIO (ver
    # criar_tabelas_LUT). A fase condensada (Al₂O₃) NÃO é transportada aqui —
    # entra só no η_2ph pós-processado (Hermsen/Stokes no CaseRunner). A forma
    # antiga  T = P / (ρ·(1−χ) · R·(1−χ))  contava (1−χ) DUAS vezes, superestimando
    # T_gas por 1/(1−χ) (≈ +11% para χ=0,10). Isso inflava a viscosidade de
    # Sutherland e distorcia Re/f_darcy. Usa-se o gás cheio (χ=0 ⇒ inalterado):
    T_gas = P_eff / (max(rho_eff, 1e-6) * prop.R)
    T_gas = max(T_gas, 100.0)

    # ------------------------------------------------------------
    # 3. DIÂMETRO HIDRÁULICO DA SEÇÃO REAL
    # ------------------------------------------------------------
    D_h = max(1e-6, 4.0 * A_eff / Pm_eff)

    # ------------------------------------------------------------
    # 4. VISCOSIDADE / REYNOLDS
    # ------------------------------------------------------------
    # NOTA (Fase 0.3): as constantes de Sutherland (1.458e-6, 110.4) são do AR.
    # Usadas como aproximação para os produtos de combustão (μ ~ 7-9e-5 Pa·s em
    # 2500-3500 K). Erro aceitável — μ entra APENAS em Re/atrito (sink de momento
    # pequeno), não na energia nem na pressão de equilíbrio.
    mu_gas = 1.458e-6 * (T_gas^1.5) / (T_gas + 110.4)
    mu_gas = max(mu_gas, 1e-8)

    Re_d = max((rho_eff * abs(u) * D_h) / mu_gas, 1.0)

    # ------------------------------------------------------------
    # 5. ATRITO — Churchill (1977), contínuo em todos os regimes de Re
    #    • Laminar  (Re < ~2300): f → 64/Re  (exato)
    #    • Transição e turbulento: concorda com Haaland dentro de ~0.5%
    #    • Sem descontinuidade em Re = 2300 (B2 fix)
    #    • ε = 0.03 mm — superfície típica de grão APCP fundido (B3 fix;
    #      valor anterior 0.2 mm era 5-7× excessivo)
    # ------------------------------------------------------------
    rugosidade_absoluta = 3.0e-5   # 0.03 mm — grão APCP fundido/curado
    rugosidade_relativa = rugosidade_absoluta / D_h

    # Churchill (1977): f = 8·[(8/Re)^12 + (A+B)^(-3/2)]^(1/12)
    A_ch    = (-2.457 * log((7.0 / Re_d)^0.9 + 0.27 * rugosidade_relativa))^16
    B_ch    = (37530.0 / Re_d)^16
    f_darcy = 8.0 * ((8.0 / Re_d)^12 + (A_ch + B_ch)^(-1.5))^(1.0 / 12.0)

    f_darcy = clamp(f_darcy, 0.005, 0.1)

    return T_gas, mu_gas, Re_d, f_darcy, D_h
end

# ------------------------------------------------------------
# ATRITO
# ------------------------------------------------------------

function aplicar_atrito_viscoso!(S::Vector{SVector{3, Float64}}, i::Int, rho::Float64, u::Float64, f_darcy::Float64, D_h::Float64)
    # 1. Calcula a força de arrasto (Momento)
    termo_atrito = -(f_darcy / D_h) * (0.5 * rho * u * abs(u))
    
    # 2. Aplica na conservação. 
    # NOTA CFD PROFISSIONAL: O termo de Energia Total é ZERO para atrito adiabático.
    # O solver converterá a perda de cinética em energia interna (calor) naturalmente.
    S[i] += SVector{3, Float64}(0.0, termo_atrito, 0.0)
end

# ------------------------------------------------------------
# QUEIMA AXIAL
# ------------------------------------------------------------

function atualizar_queima_axial!(
    y_axial_faces,
    dt::Float64,
    malha::Malha1D,
    W,
    prop::Propelente{L},      # Fase 2: explicitamente parametrizado em L
    cfg::ConfigModelo,
    est_ign
) where {L}
    N = malha.N
    dx = malha.dx

    @inbounds for g in 1:n_graos_fisicos(prop)
        inh = _inh_grao(g, prop)              # faces inibidas DESTE grão (per-grão no stack)
        idx_esq_face = 2g - 1
        idx_dir_face = 2g

        if (y_axial_faces[idx_esq_face] + y_axial_faces[idx_dir_face]) < comprimento_grao(g, prop)
            x_face_esq, x_face_dir = limites_grao_ativo(g, prop, y_axial_faces)

            idx_esq = clamp(ceil(Int, x_face_esq / dx), 1, N)
            # Face TRASEIRA: amostra a célula à ESQUERDA de x_fim (interior do grão).
            # Com `ceil` (célula à direita/aft), a face aft do ÚLTIMO grão cai em
            # x = L_camara → a célula fica ALÉM de L_camara → est_ign nunca liga →
            # a recessão axial travava em 0 (o furo não encurtava, inflando Kn/P no
            # burnout do BATES não-inibido). `ceil-1` a coloca no interior ignitado.
            # End-burner mantém `ceil` (sua recessão aft é tratada à parte, B2).
            idx_dir = (prop.grain_type === :end_burner) ?
                clamp(ceil(Int, x_face_dir / dx),     1, N) :
                clamp(ceil(Int, x_face_dir / dx) - 1, 1, N)

            # Face esquerda: regride se não inibida (inh < 2) e faces axiais ativas
            face_esq_ativa = (inh < 2) && cfg.usar_faces_axiais
            if face_esq_ativa && est_ign[idx_esq]
                P_local_esq = max(1e-6, W[idx_esq][3])
                r_ax_esq = cfg.f_regressao_axial * prop.a * (P_local_esq ^ prop.n)
                y_axial_faces[idx_esq_face] += r_ax_esq * dt
            end

            # Face direita: regride se desinibida (inh == 0: ambas livres; inh == 3: só direita)
            face_dir_ativa = (inh == 0 || inh == 3) && cfg.usar_faces_axiais
            if face_dir_ativa && est_ign[idx_dir]
                P_local_dir = max(1e-6, W[idx_dir][3])
                r_ax_dir = cfg.f_regressao_axial * prop.a * (P_local_dir ^ prop.n)
                y_axial_faces[idx_dir_face] += r_ax_dir * dt
            end
        end
    end
end

# ------------------------------------------------------------
# IGNIÇÃO E AQUECIMENTO
# ------------------------------------------------------------

function atualizar_ignicao_e_temperatura!(
    i::Int,
    dt::Float64,
    malha::Malha1D,
    rho::Float64,
    u::Float64,
    D_h::Float64,
    T_gas::Float64,
    prop::Propelente{L},      # Fase 2
    ig::Ignitor,
    cfg::ConfigModelo,
    T_sup,
    est_ign,
    t_heating,
    alpha_p::Float64,
    m_dot_ign_total::Float64
) where {L}
    if est_ign[i]
        return
    end

    G_flux = rho * abs(u)
    h_c = cfg.coef_convectivo_base * (G_flux^0.8) * (prop.k_p / D_h^0.2)

    if m_dot_ign_total > 0.0 && malha.x_centros[i] <= ig.posicao_final
        h_c += cfg.ganho_termico_ignitor
    end

    q_flux = h_c * max(0.0, T_gas - T_sup[i])

    # Solução analítica corpo semi-infinito: T(t) = T_grain + (2q/k)·√(α·t/π).
    # Usar t_heating acumulado em vez de somar √dt a cada passo elimina o erro
    # de malha (resultado antes escalava com √N em vez de ser independente de dt).
    if q_flux > 0.0
        t_new = t_heating[i] + dt
        T_new = prop.T_grain + (q_flux * 2.0 / prop.k_p) * sqrt(alpha_p * t_new / pi)
        T_sup[i] = max(T_sup[i], T_new)
        t_heating[i] = t_new
    end

    if T_sup[i] >= prop.T_ignicao
        est_ign[i] = true
    end
end

# ------------------------------------------------------------
# REGRESSÃO RADIAL DO SÓLIDO
# ------------------------------------------------------------

# Queima erosiva — MODELO UNIVERSAL de Mukunda & Paul (1997) com modificação Mukunda et al. (2014).
#   η = r/r0 = 1 + 0.023·(g^0.8 − g_th^0.8),  g = g0·(Re0/1000)^(−0.125),  g_th = 35
#   g0 = G/(ρ·r0)  (fluxo de massa adimensional) ;  Re0 = ρ·r0·d0/μ_gas
# d0 = P/π  onde P é o perímetro do porto (Mukunda et al. 2014, Acta Astronaut. 93:176).
# Para porto cilíndrico: P=πD → d0=D (degenera para o modelo original, i.e., sem mudança).
# Para finocyl: P/π > D_h=4A/P → Re0 maior → g menor → MENOS erosão. Corrige a
# superpredição que o modelo original (d0=D_h) cometia em geometrias não-axissimétricas.
# Coeficientes (0.023, −0.125, g_th=35) permanecem universais; μ = cfg.mu_gas.
@inline function _taxa_erosiva_mukunda(r_base::Float64, G::Float64, d0::Float64,
                                       prop::Propelente, cfg::ConfigModelo)
    (G > 0.0 && r_base > 0.0 && d0 > 0.0) || return r_base
    ρ   = prop.rho_p
    g0  = G / (ρ * r_base)                     # fluxo de massa adimensional
    Re0 = ρ * r_base * d0 / cfg.mu_gas         # Reynolds com d0 = P/π
    g   = g0 * (Re0 / 1000.0)^(-0.125)
    g > cfg.g_th_mukunda || return r_base       # abaixo do limiar universal → sem erosiva
    r_ero = 0.023 * (g^0.8 - cfg.g_th_mukunda^0.8) * r_base
    return r_base + min(r_ero, cfg.cap_erosivo_mult * r_base)
end

# Queima erosiva (Lenoir-Robillard) — FONTE ÚNICA, chamada pelo preditor
# (atualizar_regressao_radial_celula!) e pelo corretor (recalcular_r_de_W!).
# r_base = taxa base (Saint-Robert + térmica); G = ρ·|u| (fluxo de massa);
# d0 = parâmetro de escala do porto: P/π (Mukunda 2014) ou 4A/P (L-R).
# Chamador calcula d0 antes de chamar esta função. Retorna r_base + contribuição erosiva.
@inline function _taxa_erosiva(r_base::Float64, G::Float64, d0::Float64,
                               prop::Propelente, cfg::ConfigModelo)
    # :mukunda      → Mukunda-Paul modificado (d0 = P/π, correção finocyl 2014)
    # :mukunda_std  → Mukunda-Paul padrão     (d0 = D_h = 4A/P; chamador passa D_h)
    (cfg.modelo_erosivo === :mukunda || cfg.modelo_erosivo === :mukunda_std) &&
        return _taxa_erosiva_mukunda(r_base, G, d0, prop, cfg)
    G > cfg.G_erosao_lim || return r_base
    termo_conv = (prop.alpha_e * (G^0.8)) / (d0^0.2)
    blowing    = exp(clamp(-(prop.beta_e * prop.rho_p * r_base) / G, -50.0, 0.0))
    x_blend    = clamp((G - cfg.G_erosao_lim) / (0.5 * cfg.G_erosao_lim), 0.0, 1.0)
    s_blend    = x_blend * x_blend * (3.0 - 2.0 * x_blend)
    r_ero      = termo_conv * blowing * s_blend
    return r_base + min(r_ero, cfg.cap_erosivo_mult * r_base)
end

@inline function atualizar_regressao_radial_celula!(
    i::Int,
    x_centro::Float64,
    dt::Float64,
    rho::Float64,
    u::Float64,
    P::Float64,
    prop::Propelente{L},      # Fase 2: L = typeof(layout) → zero dispatch em obter_geometria_local
    cfg::ConfigModelo,
    y_queima,
    r_local_cache,
    T_superficie_local,
    ignitado::Bool,
    ignitor_ativo::Bool = false   # NEW: quando true, erosão é desativada nesta célula
) where {L}
    # End-burner: sem regressão radial (toda a queima é axial, pela face traseira).
    # r_local_cache = 0 → montar_fonte_propelente_celula! não injeta massa pelas
    # paredes laterais; a injeção ocorre apenas via calcular_area_queima_celula
    # quando a face axial (y_axial_faces) está na célula.
    if prop.grain_type === :end_burner
        r_local_cache[i] = 0.0
        return 0.0
    end

    web_max = obter_ymax_local(x_centro, prop)
    y_lut = clamp(y_queima[i], 0.0, web_max)

    # 1. Verificação de Burnout (Fim da queima)
    if y_lut >= web_max
        r_local_cache[i] = 0.0
        return 0.0
    end

    # 2. TRAVA DE IGNIÇÃO: Se a célula não está ignitada, a taxa de regressão é ZERO.
    # Isso impede o pico de pressão irreal no início (t < 0.02s).
    if !ignitado
        r_local_cache[i] = 0.0
        return 0.0
    end

    # 3. Cálculo da Geometria Local
    A_lut, P_lut = obter_geometria_local(x_centro, y_lut, prop)
    Area_atual = max(A_lut, 1e-8)
    Perimetro_atual = max(P_lut, 1e-6)
    D_h = max(1e-6, 4.0 * Area_atual / max(Perimetro_atual, 1e-6))

    # SEGURANÇA: teto elevado de 20→50 MPa. O teto anterior (20 MPa) SATURAVA a
    # taxa de queima em picos transitórios (ex.: spike erosivo de 32.7 MPa),
    # suprimindo a realimentação queima→pressão e SUBESTIMANDO o pico real —
    # não-conservador para dimensionamento estrutural. 50 MPa cobre qualquer MEOP
    # plausível; acima disso o motor já está em falha catastrófica.
    P_segura = clamp(P, 50000.0, 50e6)

    # 4. Lei de Saint-Robert: r = a · P^n
    # Removido o 'fator_dinamico' conforme sua observação para manter paridade com OpenMotor
    # NOTA (I3, revisão Fase 1): resposta de queima ESTÁTICA — r responde
    # instantaneamente à pressão local, sem atraso/admitância dinâmica da chama.
    # Portanto o 1D NÃO captura instabilidade de combustão (acoplamento queima↔acústica).
    # A margem de estabilidade balística (SM = 1−n) e os modos acústicos são avaliados
    # à parte em StabilityAnalysis.jl.
    r_base = prop.a * (P_segura ^ prop.n)
    
    # Correção de sensibilidade térmica (baseada na temperatura inicial do grão)
    fator_termico = exp(prop.sigma_p * (prop.T_grain - prop.T_ref))
    r_ref = r_base * fator_termico

    # 5. Queima Erosiva — via _taxa_erosiva (fonte única).
    # d0: :mukunda → P/π (mod. finocyl 2014); :mukunda_std / :lenoir → D_h = 4A/P.
    # ignitor_ativo: G artificialmente alto na ignição → erosiva desligada nessa fase.
    r_total = r_ref
    if cfg.usar_erosiva && !ignitor_ativo
        G    = max(1e-6, rho * abs(u))
        d0   = cfg.modelo_erosivo === :mukunda ?
               max(1e-6, Perimetro_atual / π) : D_h
        r_total = _taxa_erosiva(r_ref, G, d0, prop, cfg)
    end

    # 6. Atualização do estado sólido
    y_queima[i] = min(web_max, y_queima[i] + r_total * dt)
    r_local_cache[i] = r_total

    return r_total
end

# ------------------------------------------------------------
# ÁREA DE QUEIMA DA CÉLULA (BATES)
# ------------------------------------------------------------

@inline function calcular_area_queima_celula(
    i::Int,
    y_queima_local::Float64,
    malha::Malha1D,
    prop::Propelente{L},      # Fase 2
    cfg::ConfigModelo,
    y_axial_faces
) where {L}
    x_centro = malha.x_centros[i]
    dx = malha.dx
    x_esq_cel = x_centro - dx / 2
    x_dir_cel = x_centro + dx / 2

    # Soma TODOS os grãos que a célula sobrepõe. Antes só entrava o grão do centro:
    # a célula que cruza a divisa entre dois grãos perdia o trecho do outro grão
    # (≈4 mm no BATES 2×300 mm com N=80 e com N=120 — por isso o V3 "convergia"),
    # e a célula de borda com centro além do fim do grão era pulada (ver tem_propelente).
    g_lo, g_hi = _faixa_graos(x_esq_cel, x_dir_cel, prop)
    A_lateral = 0.0
    A_faces   = 0.0
    @inbounds for g in g_lo:g_hi
        a_lat, a_fac = _area_queima_grao_na_celula(g, x_centro, x_esq_cel, x_dir_cel, dx,
                                                   y_queima_local, prop, cfg, y_axial_faces)
        A_lateral += a_lat
        A_faces   += a_fac
    end
    return A_lateral + A_faces, A_lateral, A_faces
end

# Índices (contíguos) dos grãos que sobrepõem o intervalo [x_esq, x_dir].
@inline function _faixa_graos(x_esq::Float64, x_dir::Float64, prop::Propelente)
    if isempty(prop.grain_boundaries)                                # uniforme (legado)
        g_lo = clamp(floor(Int, x_esq / prop.L_grao) + 1, 1, prop.N_graos)
        g_hi = clamp(ceil(Int,  x_dir / prop.L_grao),     1, prop.N_graos)
    else                                                             # stack heterogêneo
        g_lo = _grao_em_x(x_esq, prop)
        g_hi = _grao_em_x(x_dir, prop)
    end
    return g_lo, g_hi
end

# Área lateral e de faces do grão g dentro da célula [x_esq_cel, x_dir_cel].
@inline function _area_queima_grao_na_celula(
    g::Int, x_centro::Float64, x_esq_cel::Float64, x_dir_cel::Float64, dx::Float64,
    y_queima_local::Float64, prop::Propelente{L}, cfg::ConfigModelo, y_axial_faces
) where {L}
    idx_esq_face = 2g - 1
    idx_dir_face = 2g

    if (y_axial_faces[idx_esq_face] + y_axial_faces[idx_dir_face]) >= comprimento_grao(g, prop)
        return 0.0, 0.0
    end

    x_ini_g, x_fim_g = limites_grao_ativo(g, prop, y_axial_faces)

    x_a = max(x_esq_cel, x_ini_g)
    x_b = min(x_dir_cel, x_fim_g)
    L_efetivo = max(0.0, x_b - x_a)

    if L_efetivo <= 0.0
        return 0.0, 0.0
    end

    # ==========================================
    # LUT Pseudo-3D (Cilindro ou Estrela)
    # ==========================================
    y_lut = clamp(y_queima_local, 0.0, prop.y_max)

    # Geometria no meio do trecho de grão dentro da célula (= centro da célula
    # quando ela está inteira no grão; nas células de borda, o meio da sobreposição).
    x_geo = (x_a == x_esq_cel && x_b == x_dir_cel) ? x_centro : 0.5 * (x_a + x_b)
    A_lut, P_lut = obter_geometria_local(x_geo, y_lut, prop)

    Area_secao_cheia = (pi / 4.0) * prop.D_ext^2
    Perimetro_local = max(0.0, P_lut)
    Area_port_local = clamp(A_lut, 0.0, Area_secao_cheia)

    A_lateral = Perimetro_local * L_efetivo

    A_faces = 0.0
    f_faces = cfg.usar_faces_axiais ? cfg.f_faces_axiais : 0.0

    # End-burner: toda a secção transversal é propelente na face (sem porto central).
    # Para outros tipos: a face disponível é a coroa (A_ext − A_port).
    Area_face_disponivel = prop.grain_type === :end_burner ?
        Area_secao_cheia :
        max(0.0, Area_secao_cheia - Area_port_local)

    # ── Lógica de faces axiais (BATES) ───────────────────────────────────────
    # No BATES clássico, cada grão é uma peça física separada.
    # As faces são expostas ou inibidas dependendo da configuração:
    #
    #   inhibited_ends = 0 → NENHUMA face inibida: todas as 2×N_graos faces queimam.
    #                         Isso inclui as faces INTERNAS entre grãos adjacentes,
    #                         pois há espaçador entre eles mas as faces são expostas ao gás.
    #                         (Configuração do OpenMotor "uninhibited")
    #
    #   inhibited_ends = 1 → 1 face inibida por grão: só a face esquerda queima.
    #
    #   inhibited_ends = 2 → AMBAS inibidas: nenhuma face queima.
    #                         (Grão colado à carcaça ou com inibidor nas duas faces)
    #
    inh = _inh_grao(g, prop)

    # Face esquerda do grão g: livre se não inibida (inh < 2)
    face_esq_livre = (inh < 2)

    # Face direita do grão g: livre se desinibida (inh == 0: ambas livres; inh == 3: só direita)
    face_dir_livre = (inh == 0 || inh == 3)

    A_face_unit = f_faces * Area_face_disponivel

    # ── Smear axial da área da face (corrige degrau espúrio em BATES não-inibido) ──
    # A área da face é distribuída por um box de largura h_eff no INTERIOR do grão,
    # em vez de despejada no cell único da posição da face. Assim a face "acende"
    # gradualmente conforme a chama varre os cells (o loop principal só soma cells
    # ignitados), eliminando o degrau. Σ(overlap/h_eff)=1 → área total conservada
    # (pico/MEOP intactos). End-burner mantém sua lógica dedicada (face = secção cheia).
    if cfg.h_smear_faces_m > 0.0 && prop.grain_type !== :end_burner
        h_smear = max(cfg.h_smear_faces_m, 3.0 * dx)
        h_eff   = min(h_smear, x_fim_g - x_ini_g)        # box não excede o grão
        if h_eff > 0.0
            if face_esq_livre                            # interior = +x: [x_ini_g, x_ini_g+h_eff]
                ovl = min(x_dir_cel, x_ini_g + h_eff) - max(x_esq_cel, x_ini_g)
                ovl > 0.0 && (A_faces += (ovl / h_eff) * A_face_unit)
            end
            if face_dir_livre                            # interior = -x: [x_fim_g-h_eff, x_fim_g]
                ovl = min(x_dir_cel, x_fim_g) - max(x_esq_cel, x_fim_g - h_eff)
                ovl > 0.0 && (A_faces += (ovl / h_eff) * A_face_unit)
            end
        end
    else
        # Área da face no cell único que contém a posição da face. Face exatamente
        # numa divisa de células vai para a célula do lado do PROPELENTE (a da
        # direita para a face esquerda; a da esquerda para a face direita) — senão a
        # face aft em x = L_camara cairia na célula além do grão e se perderia.
        if face_esq_livre && x_esq_cel <= x_ini_g < x_dir_cel
            A_faces += A_face_unit
        end
        if face_dir_livre && x_esq_cel < x_fim_g <= x_dir_cel
            A_faces += A_face_unit
        end
    end

    return A_lateral, A_faces
end
# ------------------------------------------------------------
# FONTE DO PROPELENTE
# ------------------------------------------------------------

@inline function montar_fonte_propelente_celula!(
    S,
    i::Int,
    malha::Malha1D,
    prop::Propelente{L},      # Fase 2
    cfg::ConfigModelo,
    y_queima_geom::Float64,
    y_axial_faces,
    r_local_cache,
    h0_prop::Float64,
    A_vol_local::Float64,
    t_atual::Float64
) where {L}
    dx = malha.dx
    r_total = r_local_cache[i]

    # ------------------------------------------------------------
    # 1. Se não há regressão local, não há fonte
    # ------------------------------------------------------------
    if r_total <= 0.0
        return 0.0, 0.0, 0.0, 0.0
    end

    # ------------------------------------------------------------
    # 2. Usa o estado geométrico local pré-regressão / saturado
    # ------------------------------------------------------------
    y_lut = clamp(y_queima_geom, 0.0, prop.y_max)

    # ------------------------------------------------------------
    # 3. Calcula a área de queima da célula pela função dedicada
    # ------------------------------------------------------------
    A_queima, A_lateral, A_faces = calcular_area_queima_celula(
        i,
        y_lut,
        malha,
        prop,
        cfg,
        y_axial_faces
    )

    # ------------------------------------------------------------
    # 4. Montagem da fonte de massa e energia
    # ------------------------------------------------------------
    if A_queima > 0.0
        m_dot_inj = r_total * prop.rho_p * A_queima

        # Fator de ramp de ignição: m_dot cresce de 0→1 em cfg.t_ramp_ignicao [s].
        # Smoothstep S(x) = 3x²−2x³: derivada zero nos extremos → sem quinas
        # em t=0 (arranque suave) e t=t_ramp (transição suave para queima estável).
        x_ramp = clamp(t_atual / cfg.t_ramp_ignicao, 0.0, 1.0)
        ramp   = x_ramp * x_ramp * (3.0 - 2.0 * x_ramp)
        m_dot_inj *= ramp

        A_vol = max(A_vol_local, 1e-10)
        termo_massa  = m_dot_inj / (A_vol * dx)
        termo_energia = (m_dot_inj * h0_prop) / (A_vol * dx)

        S[i] += SVector{3, Float64}(termo_massa, 0.0, termo_energia)
        return A_queima, A_lateral, A_faces, m_dot_inj
    end

    return A_queima, A_lateral, A_faces, 0.0
end

# ------------------------------------------------------------
# FONTE DO IGNITOR
# ------------------------------------------------------------

function aplicar_fonte_ignitor!(
    S,
    malha::Malha1D,
    ig::Ignitor,
    m_dot_ign_total::Float64
)
    if m_dot_ign_total <= 0.0
        return
    end

    h0_ign = calcular_h0_ignitor(ig)
    dx = malha.dx

    # Correção: Loop explícito para não gerar alocação de memória
    n_ign_cells = 0
    @inbounds for i in 1:malha.N
        if malha.x_centros[i] <= ig.posicao_final
            n_ign_cells += 1
        end
    end
    n_ign_cells = max(1, n_ign_cells)

    @inbounds for i in 1:n_ign_cells
        m_dot_local = m_dot_ign_total / n_ign_cells
        termo_massa = m_dot_local / (malha.A_centros[i] * dx)
        termo_energia = (m_dot_local * h0_ign) / (malha.A_centros[i] * dx)
        
        S[i] += SVector{3, Float64}(termo_massa, 0.0, termo_energia)
    end
end

# ------------------------------------------------------------
# FUNÇÃO PRINCIPAL ORQUESTRADORA
# ------------------------------------------------------------

@inline function calcular_termo_fonte_e_regressao!(
    S,
    y_queima,
    y_axial_faces,
    r_local_cache,
    dt,
    malha,
    W,
    prop::Propelente{L},      # Fase 2: L concreto → toda a chain de geometria é type-stable
    ig::Ignitor,
    cfg::ConfigModelo,
    T_sup,
    est_ign,
    t_heating,
    t,
    geom_cache_A,
    geom_cache_P,
    h0_prop::Float64
) where {L}
    fill!(S, zero(SVector{3, Float64}))
    r_local_cache .= 0.0

    m_dot_ign_total = calcular_vazao_ignitor(t, ig)
    L_camara = comprimento_camara(prop)
    alpha_p = prop.k_p / (prop.rho_p * prop.cp_p)

    A_total_queima = 0.0
    A_total_lateral = 0.0
    A_total_faces = 0.0
    m_dot_total = 0.0

    atualizar_queima_axial!(y_axial_faces, dt, malha, W, prop, cfg, est_ign)

    # Cache de geometria: usa A_centros já atualizado por atualizar_geometria_camara!
    # (chamada a cada 50 passos no main loop). Perimetro calculado diretamente.
    # Isso elimina 100 chamadas de interpolação de spline por passo.
    L_camara_gc = comprimento_camara(prop)
    @inbounds for i in 1:malha.N
        A = max(malha.A_centros[i], 1e-10)
        geom_cache_A[i] = A
        if malha.x_centros[i] <= L_camara_gc
            # Câmara: perimetro de queima via P_burn da LUT
            y_lut_gc = clamp(y_queima[i], 0.0, prop.y_max)
            _, Pb, _, _ = obter_geom_em_xy(malha.x_centros[i], y_lut_gc, prop)
            geom_cache_P[i] = max(Pb, 1e-8)
        else
            # Tubeira: perimetro circular equivalente
            geom_cache_P[i] = perimetro_circular_por_area(A)
        end
    end

    @inbounds for i in 1:malha.N
        rho   = max(1e-6, W[i][1])
        u     = W[i][2]
        P     = max(101325.0, W[i][3])
        y_geom = clamp(y_queima[i], 0.0, prop.y_max)

        Area_fluxo      = geom_cache_A[i]
        Perimetro_fluxo = geom_cache_P[i]
        D_h = max(1e-6, 4.0 * Area_fluxo / Perimetro_fluxo)

        T_gas, _, _, f_darcy, _ = calcular_propriedades_escoamento_local(
            rho, u, P, Area_fluxo, Perimetro_fluxo, prop
        )
        # Atrito de parede só na câmara: na tubeira a perda viscosa é a de camada
        # limite (η_bl no empuxo; SP-8039 §2.1.3.2.1.4) — ver cfg.usar_atrito_tubeira.
        (cfg.usar_atrito_tubeira || malha.x_centros[i] <= L_camara) &&
            aplicar_atrito_viscoso!(S, i, rho, u, f_darcy, D_h)

        if tem_propelente(i, malha, L_camara)
            atualizar_ignicao_e_temperatura!(
                i, dt, malha, rho, u, D_h, T_gas, prop, ig, cfg,
                T_sup, est_ign, t_heating, alpha_p, m_dot_ign_total
            )
            if est_ign[i]
                atualizar_regressao_radial_celula!(
                   i, malha.x_centros[i], dt, rho, u, P, prop, cfg, y_queima, r_local_cache,
                   T_sup[i], est_ign[i], m_dot_ign_total > 0.0
                )
                A_queima, A_lateral, A_faces, m_dot_inj = montar_fonte_propelente_celula!(
                    S, i, malha, prop, cfg, y_geom, y_axial_faces, r_local_cache, h0_prop, Area_fluxo, t
                )
                A_total_queima  += A_queima
                A_total_lateral += A_lateral
                A_total_faces   += A_faces
                m_dot_total     += m_dot_inj
            end
        end
    end

    # ── B2 FIX: End-burner — população de r_local_cache para face ativa ────────
    # O loop principal acima zerou r_local_cache para todos os cells end_burner
    # (atualizar_regressao_radial_celula! retorna 0 imediatamente).
    # Consequência sem este patch: montar_fonte_propelente_celula! vê r_total=0
    # e retorna sem injetar massa → pressão nunca sobe → simulação inválida.
    #
    # Correção: após o loop, identificamos a célula da face traseira activa de
    # cada grão, atribuímos r = a·P^n e relançamos montar_fonte_propelente_celula!
    # para injetar massa/energia apenas nessa célula.
    # r_local_cache persiste para calcular_fontes_gas_rk2! (dois estágios RK2).
    if prop.grain_type === :end_burner
        @inbounds for g in 1:n_graos_fisicos(prop)
            idx_dir_face = 2g
            # Grão ainda tem propelente a consumir
            if y_axial_faces[idx_dir_face] < comprimento_grao(g, prop)
                _, x_face_dir = limites_grao_ativo(g, prop, y_axial_faces)
                idx_face   = clamp(ceil(Int, x_face_dir / malha.dx), 1, malha.N)
                if est_ign[idx_face]
                    P_face = max(101325.0, W[idx_face][3])
                    r_local_cache[idx_face] = prop.a * (P_face ^ prop.n)
                    A_q, A_lat, A_fac, m_inj = montar_fonte_propelente_celula!(
                        S, idx_face, malha, prop, cfg, 0.0, y_axial_faces,
                        r_local_cache, h0_prop, geom_cache_A[idx_face], t
                    )
                    A_total_queima  += A_q
                    A_total_lateral += A_lat
                    A_total_faces   += A_fac
                    m_dot_total     += m_inj
                end
            end
        end
    end

    aplicar_fonte_ignitor!(S, malha, ig, m_dot_ign_total)
    return A_total_queima, A_total_lateral, A_total_faces, m_dot_total
end

# =========================================================
# NOVA FUNÇÃO: Cálculo do Termo Fonte Dinâmico (Para RK2)
# CORREÇÃO: Fim da queima (Burnout/Web Max) respeitado!
# =========================================================
@inline function calcular_fontes_gas_rk2!(
    S,
    W,
    y_queima_pre,
    y_axial_faces,
    r_local_cache,
    malha,
    prop::Propelente{L},      # Fase 2
    cfg::ConfigModelo,
    est_ign,
    t::Float64,
    ig::Ignitor,
    geom_cache_A,
    geom_cache_P,
    h0_prop::Float64
) where {L}
    fill!(S, zero(eltype(S)))

    L_camara = comprimento_camara(prop)
    m_dot_ign_total = calcular_vazao_ignitor(t, ig)

    @inbounds for i in 1:malha.N
        rho = max(1e-6, W[i][1])
        u   = W[i][2]
        P   = max(1e-6, W[i][3])

        x_i       = malha.x_centros[i]
        web_max_i = obter_ymax_local(x_i, prop)
        y_lut     = clamp(y_queima_pre[i], 0.0, web_max_i)

        Area_fluxo      = geom_cache_A[i]
        Perimetro_fluxo = geom_cache_P[i]

        _, _, _, f_darcy, D_h = calcular_propriedades_escoamento_local(
            rho, u, P, Area_fluxo, Perimetro_fluxo, prop
        )
        (cfg.usar_atrito_tubeira || x_i <= L_camara) &&
            aplicar_atrito_viscoso!(S, i, rho, u, f_darcy, D_h)

        if tem_propelente(i, malha, L_camara) && est_ign[i] && y_lut < web_max_i
            montar_fonte_propelente_celula!(
                S, i, malha, prop, cfg, y_lut, y_axial_faces,
                r_local_cache, h0_prop, Area_fluxo, t
            )
        end
    end

    aplicar_fonte_ignitor!(S, malha, ig, m_dot_ign_total)
end

# ==============================================================================
# STRANG SPLITTING — AUXILIARES
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. recalcular_r_de_W!
#    Recalcula r_local_cache a partir do estado de gás W ATUAL (sem avançar y_queima).
#    Chamado antes do estágio 2 do RK2 para que S_star use r(P*) em vez de r(P^n).
#    Inclui a mesma lógica de erosão de Lenoir-Robillard de atualizar_regressao_radial_celula!.
# ------------------------------------------------------------------------------
function recalcular_r_de_W!(
    r_local_cache ::Vector{Float64},
    W             ::Vector{SVector{3,Float64}},
    y_queima      ::Vector{Float64},
    malha         ::Malha1D,
    prop          ::Propelente{L},
    cfg           ::ConfigModelo,
    est_ign       ::Vector{Bool},
    geom_cache_A  ::Vector{Float64},
    geom_cache_P  ::Vector{Float64}
) where {L}
    L_camara = comprimento_camara(prop)
    @inbounds for i in 1:malha.N
        x_i = malha.x_centros[i]
        if !tem_propelente(i, malha, L_camara) || !est_ign[i]
            r_local_cache[i] = 0.0
            continue
        end
        web_max = obter_ymax_local(x_i, prop)
        y_lut   = clamp(y_queima[i], 0.0, web_max)
        if y_lut >= web_max
            r_local_cache[i] = 0.0
            continue
        end
        rho = max(1e-6,    W[i][1])
        u   =              W[i][2]
        P   = clamp(W[i][3], 50000.0, 50e6)   # teto 50 MPa (ver nota em atualizar_regressao_radial_celula!)

        # Lei de Saint-Robert com sensibilidade térmica
        r_base = prop.a * (P ^ prop.n) *
                 exp(prop.sigma_p * (prop.T_grain - prop.T_ref))

        # Queima erosiva — via _taxa_erosiva (mesma fonte do preditor).
        # d0: :mukunda → P/π (mod. finocyl); :mukunda_std / :lenoir → D_h = 4A/P.
        if cfg.usar_erosiva
            Peri  = max(geom_cache_P[i], 1e-6)
            d0    = cfg.modelo_erosivo === :mukunda ?
                    max(1e-6, Peri / π) :
                    max(1e-6, 4.0 * geom_cache_A[i] / Peri)
            G     = max(1e-6, rho * abs(u))
            r_base = _taxa_erosiva(r_base, G, d0, prop, cfg)
        end
        r_local_cache[i] = r_base
    end
end

# ------------------------------------------------------------------------------
# 2. calcular_fontes_somente_atrito!
#    Termo fonte apenas de atrito viscoso (sem injeção de combustão).
#    Usado no passo hiperbólico L do Strang splitting.
# ------------------------------------------------------------------------------
function calcular_fontes_somente_atrito!(
    S            ::Vector{SVector{3,Float64}},
    W            ::Vector{SVector{3,Float64}},
    malha        ::Malha1D,
    prop         ::Propelente{L},
    cfg          ::ConfigModelo,
    geom_cache_A ::Vector{Float64},
    geom_cache_P ::Vector{Float64}
) where {L}
    fill!(S, zero(SVector{3,Float64}))
    L_camara = comprimento_camara(prop)
    @inbounds for i in 1:malha.N
        (cfg.usar_atrito_tubeira || malha.x_centros[i] <= L_camara) || continue
        rho = max(1e-6, W[i][1])
        u   = W[i][2]
        P   = max(1e-6, W[i][3])
        Area_fluxo      = geom_cache_A[i]
        Perimetro_fluxo = geom_cache_P[i]
        _, _, _, f_darcy, D_h = calcular_propriedades_escoamento_local(
            rho, u, P, Area_fluxo, Perimetro_fluxo, prop)
        aplicar_atrito_viscoso!(S, i, rho, u, f_darcy, D_h)
    end
end

# ------------------------------------------------------------------------------
# 3. aplicar_meio_passo_combustao!
#    Operador S(dt_half): avança y_queima e injeta massa/energia em U.
#    Retorna (A_queima_total, m_dot_total) para estatísticas do passo.
#    NÃO recalcula r_local_cache — usa o cache atual.  Chamar
#    recalcular_r_de_W! antes se quiser r(P^(n+1)) no 2º meio-passo.
# ------------------------------------------------------------------------------
function aplicar_meio_passo_combustao!(
    U            ::Vector{SVector{3,Float64}},
    W            ::Vector{SVector{3,Float64}},
    y_queima     ::Vector{Float64},
    y_axial_faces::Vector{Float64},
    r_local_cache::Vector{Float64},
    malha        ::Malha1D,
    prop         ::Propelente{L},
    ig           ::Ignitor,
    cfg          ::ConfigModelo,
    est_ign      ::Vector{Bool},
    t_atual      ::Float64,
    geom_cache_A ::Vector{Float64},
    geom_cache_P ::Vector{Float64},
    h0_prop      ::Float64,
    dt_half      ::Float64
) where {L}
    L_camara = comprimento_camara(prop)

    # Avanço axial de meio-passo
    atualizar_queima_axial!(y_axial_faces, dt_half, malha, W, prop, cfg, est_ign)

    A_total  = 0.0
    m_total  = 0.0
    # Smoothstep S(x) = 3x²−2x³ (derivada zero nos dois extremos)
    x_ramp   = clamp(t_atual / cfg.t_ramp_ignicao, 0.0, 1.0)
    ramp     = x_ramp * x_ramp * (3.0 - 2.0 * x_ramp)

    @inbounds for i in 1:malha.N
        x_i = malha.x_centros[i]
        !tem_propelente(i, malha, L_camara) && continue
        !est_ign[i]    && continue

        web_max = obter_ymax_local(x_i, prop)
        y_lut   = clamp(y_queima[i], 0.0, web_max)
        y_lut >= web_max && continue

        r_i = r_local_cache[i]
        r_i <= 0.0 && continue

        # Avança sólido
        y_queima[i] = min(web_max, y_queima[i] + r_i * dt_half)

        # Área de queima
        A_q, _, _, m_inj_cell = calcular_area_e_injecao(
            i, y_lut, y_axial_faces, r_i, ramp, h0_prop,
            malha, prop, cfg, geom_cache_A)

        A_total += A_q
        m_total += m_inj_cell

        # Injeta massa e energia em U (sem componente de momentum — injeção normal)
        if m_inj_cell > 0.0
            A_vol = max(geom_cache_A[i], 1e-10)
            dx    = malha.dx
            dm    = m_inj_cell * dt_half / (A_vol * dx)   # Δρ
            dE    = dm * h0_prop                           # ΔρE
            U[i]  = U[i] + SVector{3,Float64}(dm, 0.0, dE)
        end
    end

    # Ignitor
    m_ign = calcular_vazao_ignitor(t_atual, ig)
    if m_ign > 0.0
        h0_ign = calcular_h0_ignitor(ig)
        n_ign  = max(1, sum(x -> x <= ig.posicao_final, malha.x_centros))
        @inbounds for i in 1:n_ign
            A_vol   = max(malha.A_centros[i], 1e-10)
            dx      = malha.dx
            dm_loc  = (m_ign / n_ign) * dt_half / (A_vol * dx)
            U[i]   += SVector{3,Float64}(dm_loc, 0.0, dm_loc * h0_ign)
        end
    end

    return A_total, m_total
end

# Helper interno usado por aplicar_meio_passo_combustao!
@inline function calcular_area_e_injecao(
    i, y_lut, y_axial_faces, r_i, ramp, h0_prop,
    malha, prop, cfg, geom_cache_A
)
    A_q, A_lat, A_fac = calcular_area_queima_celula(i, y_lut, malha, prop, cfg, y_axial_faces)
    if A_q > 0.0
        m_inj = r_i * prop.rho_p * A_q * ramp
        return A_q, A_lat, A_fac, m_inj
    end
    return A_q, A_lat, A_fac, 0.0
end