using StaticArrays

# ==========================================
# MÓDULO: SOLVER NUMÉRICO (Euler 1D / MUSCL)
# Versão: Mega Otimizada (Type-Stable)
# ==========================================

# Otimização: @inline força o compilador a inserir o código no loop, zero alocação
@inline @fastmath function propriedades_locais(U::SVector{3, Float64}, R::Float64, E_tab::Vector{Float64}, T_tab::Vector{Float64}, G_tab::Vector{Float64})
    rho = max(U[1], 1e-6)
    u = U[2] / rho
    
    # Energia interna específica
    e_int = (U[3] / rho) - 0.5 * u^2
    
    # Chama a NOVA função O(1) dupla via LUT (Pega T e Gamma de uma vez só!)
    T, gamma_loc = encontrar_T_e_Gamma_LUT(e_int, E_tab, T_tab, G_tab)
    
    p = max(rho * R * T, 1e-2)
    a = sqrt(gamma_loc * p / rho)
    
    return p, u, a, gamma_loc, T
end

@inline @fastmath function fluxo_euler(U::SVector{3, Float64}, p::Float64)::SVector{3, Float64}
    rho = max(U[1], 1e-6)
    u = U[2] / rho
    return SVector{3, Float64}(U[2], U[2] * u + p, u * (U[3] + p))
end

# Otimização: Limitador Van Leer, melhor que o minmod
@inline @fastmath function van_leer(a::Float64, b::Float64)::Float64
    ab = a * b
    if ab > 0.0
        return (2.0 * ab) / (a + b)
    else
        return 0.0
    end
end
# Fluxo Rusanov (Local Lax-Friedrichs) - Usado como Fallback Seguro
@fastmath function fluxo_rusanov(U_L::SVector{3, Float64}, U_R::SVector{3, Float64}, R::Float64, E_tab::Vector{Float64}, T_tab::Vector{Float64}, G_tab::Vector{Float64})::SVector{3, Float64} # <-- G_tab adicionado aqui
    pL, uL, aL, _, _ = propriedades_locais(U_L, R, E_tab, T_tab, G_tab) # <-- e aqui
    pR, uR, aR, _, _ = propriedades_locais(U_R, R, E_tab, T_tab, G_tab) # <-- e aqui
    
    FL = fluxo_euler(U_L, pL)
    FR = fluxo_euler(U_R, pR)
    
    Smax = max(abs(uL) + aL, abs(uR) + aR)
    return 0.5 * (FL + FR) - 0.5 * Smax * (U_R - U_L)
end

# Fluxo HLLC - Alta fidelidade na resolução de contatos
@fastmath function fluxo_hllc(U_L::SVector{3, Float64}, U_R::SVector{3, Float64}, R::Float64, E_tab::Vector{Float64}, T_tab::Vector{Float64}, G_tab::Vector{Float64})::SVector{3, Float64} # <-- G_tab aqui
    pL, uL, aL, _, _ = propriedades_locais(U_L, R, E_tab, T_tab, G_tab) # <-- aqui
    pR, uR, aR, _, _ = propriedades_locais(U_R, R, E_tab, T_tab, G_tab) # <-- e aqui
    
    SL = min(uL - aL, uR - aR)
    SR = max(uL + aL, uR + aR)
    
    if SL >= 0.0
        return fluxo_euler(U_L, pL)
    elseif SR <= 0.0
        return fluxo_euler(U_R, pR)
    end
    
    rhoL = max(U_L[1], 1e-6)
    rhoR = max(U_R[1], 1e-6)
    
    SM_num = pR - pL + rhoL * uL * (SL - uL) - rhoR * uR * (SR - uR)
    SM_den = rhoL * (SL - uL) - rhoR * (SR - uR)
    SM = abs(SM_den) > 1e-10 ? SM_num / SM_den : 0.5 * (uL + uR)
    
    FL = fluxo_euler(U_L, pL)
    FR = fluxo_euler(U_R, pR)
    
    if SM >= 0.0
        fator_L = rhoL * (SL - uL) / (SL - SM)
        Us1 = fator_L
        Us2 = fator_L * SM
        Us3 = fator_L * (U_L[3] / rhoL + (SM - uL) * (SM + pL / (rhoL * (SL - uL))))
        
        Us = SVector{3, Float64}(Us1, Us2, Us3)
        return FL + SL * (Us - U_L)
    else
        fator_R = rhoR * (SR - uR) / (SR - SM)
        Us1 = fator_R
        Us2 = fator_R * SM
        Us3 = fator_R * (U_R[3] / rhoR + (SM - uR) * (SM + pR / (rhoR * (SR - uR))))
        
        Us = SVector{3, Float64}(Us1, Us2, Us3)
        return FR + SR * (Us - U_R)
    end
end

# Otimização: Acesso vetorial blindado com tipos
@inline function get_ghost_state_parede(W_1::SVector{3, Float64})
    # Condição de Parede Sólida: Densidade e Pressão são copiadas.
    # Velocidade é INVERTIDA para forçar u = 0 na face de contato.
    return SVector{3, Float64}(W_1[1], -W_1[2], W_1[3])
end

@inline function get_ghost_state_saida(W_N::SVector{3, Float64}, P_atm::Float64, gamma::Float64)
    rho_N = W_N[1]
    u_N   = W_N[2]
    P_N   = W_N[3]
    
    a_N = sqrt(gamma * P_N / rho_N)
    Mach = abs(u_N) / a_N
    
    if Mach >= 1.0
        # ESCOAMENTO SUPERSÔNICO
        # As ondas são mais lentas que o fluido. Nenhuma informação volta.
        # Extrapolação de ordem zero perfeita (deixa tudo fluir).
        return W_N
    else
        # ESCOAMENTO SUBSÔNICO (Non-Reflecting Characteristic Boundary)
        # O fluido sai, mas as ondas de som da atmosfera conseguem entrar.
        
        # 1. A pressão do fantasma é forçada para a atmosférica
        P_ghost = P_atm
        
        # 2. Conservação da Entropia s = P / rho^gamma (processo isentrópico na fronteira)
        rho_ghost = rho_N * (P_ghost / P_N)^(1.0 / gamma)
        
        # 3. Calculamos a nova velocidade do som no fantasma
        a_ghost = sqrt(gamma * P_ghost / rho_ghost)
        
        # 4. Usamos o Invariante de Riemann Positivo (J+) que viaja de dentro para fora
        J_mais = u_N + (2.0 * a_N) / (gamma - 1.0)
        
        # 5. Reconstruímos a velocidade do fantasma (u_ghost) baseada na onda que sai
        u_ghost = J_mais - (2.0 * a_ghost) / (gamma - 1.0)
        
        return SVector{3, Float64}(rho_ghost, u_ghost, P_ghost)
    end
end

# =========================================================================
# HELPER 1: Roteador de Fronteiras Físicas
# Movido para escopo de módulo para evitar closure e permitir especialização.
# =========================================================================
@inline function obter_W_borda(
    idx::Int,
    W::Vector{SVector{3, Float64}},
    N::Int,
    P_atm::Float64,
    gamma_N::Float64
)::SVector{3, Float64}
    if idx >= 1 && idx <= N
        return W[idx]
    elseif idx == 0
        return get_ghost_state_parede(W[1])
    elseif idx == -1
        return SVector{3, Float64}(W[2][1], -W[2][2], W[2][3])
    elseif idx == N + 1
        return get_ghost_state_saida(W[N], P_atm, gamma_N)
    else # idx >= N + 2
        W_N1 = get_ghost_state_saida(W[N], P_atm, gamma_N)
        return get_ghost_state_saida(W_N1, P_atm, gamma_N)
    end
end

# =========================================================================
# HELPER 2: Consistência Termodinâmica
# Movido para escopo de módulo para evitar closure e permitir especialização.
# =========================================================================
@inline function e_int_de_T(
    T_alvo::Float64,
    E_tab::Vector{Float64},
    T_tab::Vector{Float64}
)::Float64
    if T_alvo <= T_tab[1]
        return E_tab[1]
    end
    if T_alvo >= T_tab[end]
        return E_tab[end]
    end

    idx = searchsortedfirst(T_tab, T_alvo)
    idx = clamp(idx, 2, length(T_tab))

    @inbounds begin
        t0 = T_tab[idx - 1]
        t1 = T_tab[idx]
        e0 = E_tab[idx - 1]
        e1 = E_tab[idx]
    end

    return e0 + ((T_alvo - t0) / (t1 - t0)) * (e1 - e0)
end

# =========================================================================
# TERMO DE ÁREA BEM-BALANCEADO (cfg.usar_termo_area_wb)
# -------------------------------------------------------------------------
# Numa variação de área, a solução estacionária do escoamento quase-1D (a
# "onda estacionária") conserva a vazão ρuA, a entalpia total h + u²/2 e a
# entropia S — Kröner, LeFloch & Thanh, M2AN 42 (2008), Eq. 4.5; Kröner &
# Thanh, SIAM J. Numer. Anal. 43 (2005). Com gás caloricamente perfeito,
# entalpia total e entropia constantes equivalem a T₀ e p₀ constantes, então
# o estado de equilíbrio fica descrito pelos invariantes Q = (ṁ, T₀, p₀).
#
# Projetar um estado para outra área = achar, com os MESMOS (ṁ, T₀, p₀), o
# Mach na nova área, no MESMO regime (sub/supersônico) do estado de origem
# (KLT 2008, Eqs. 4.12–4.13). Se a vazão não passa na nova área (acima da
# vazão crítica), o estado é o sônico (estrangulamento).
#
# Uso:
#   • estados de face: MUSCL/van Leer aplicado a Q (e não às primitivas) e
#     projeção para a área da FACE → num escoamento isentrópico permanente os
#     dois lados de cada face coincidem com a solução exata;
#   • termo de área da célula: ∫p·dA ao longo da isentrópica da própria
#     célula, entre as áreas das suas faces. Como d[(ρu²+p)A] = p·dA num
#     escoamento isentrópico permanente, essa integral é exatamente
#     (ρu²+p)A|face_dir − (ρu²+p)A|face_esq dos estados projetados.
# Resultado: escoamento isentrópico permanente é solução EXATA do esquema
# (com qualquer degrau de área); repouso continua exato; massa e energia
# continuam conservativas (fluxo único por face × área da face). Onde há
# injeção de massa (grão), a perda de estagnação física aparece pela fonte
# de massa, como deve — ela não é mais contaminada pelos degraus de área.
# Limite do modelo quase-1D (não do esquema): num alargamento brusco real há
# uma perda de Borda-Carnot ≤ ½ρu²(1−A₁/A₂)², de natureza 2D, que a onda
# estacionária isentrópica não representa (≲0,3% de p no motor típico).
# =========================================================================

# Vazão crítica adimensional g_max = (1 + (γ−1)/2)^(−(γ+1)/(2(γ−1))) (M = 1).
# Depende só de γ: calculada 1× por laço e passada adiante (evita uma potência
# por projeção).
@inline _gmax_isen(γ::Float64)::Float64 = (0.5 * (γ + 1.0))^(-(γ + 1.0) / (2.0 * (γ - 1.0)))

# Faixa de área plana: se as áreas envolvidas diferem menos que isto (relativo),
# o termo de área é desprezível e a reconstrução primitiva é exata no equilíbrio;
# a projeção bem-balanceada (Newton + potências) só é feita onde a área varia —
# degraus e transições do grão, fim do grão, convergente e divergente. O
# desequilíbrio residual nas regiões planas é da ordem de γM²·1e-4 (desprezível).
const _TOL_AREA_PLANA = 1e-4
@inline _area_plana(a::Float64, b::Float64, c::Float64)::Bool =
    max(a, b, c) - min(a, b, c) <= _TOL_AREA_PLANA * min(a, b, c)
@inline _area_plana(a::Float64, b::Float64, c::Float64, d::Float64, e::Float64)::Bool =
    max(a, b, c, d, e) - min(a, b, c, d, e) <= _TOL_AREA_PLANA * min(a, b, c, d, e)

# Resolve M·(1 + c·M²)^(−e) = g no ramo pedido (c=(γ−1)/2, e=(γ+1)/(2(γ−1))).
# g é a vazão por área adimensionalizada por ρ₀a₀; g ≥ g_max ⇒ sônico.
# (Sem @fastmath: aqui a potência padrão é mais rápida que a pow_fast.)
@inline function _mach_de_vazao(g::Float64, γ::Float64, gmax::Float64,
                                supersonico::Bool, M_ini::Float64)::Float64
    c = 0.5 * (γ - 1.0)
    e = (γ + 1.0) / (2.0 * (γ - 1.0))
    g <= 0.0 && return 0.0
    g >= gmax && return 1.0
    lo, hi = supersonico ? (1.0, 100.0) : (0.0, 1.0)
    M = supersonico ? clamp(M_ini, 1.0 + 1e-6, 100.0) : clamp(M_ini, 0.0, 1.0 - 1e-6)
    # Perto do sônico f'(M) → 0 e o Newton a partir de M_ini converge mal (caía
    # na bissecção). Chute pela expansão em torno de M = 1:
    #   ln(g/g_max) ≈ −2(M−1)²/(γ+1)  ⇒  M ≈ 1 ∓ √((γ+1)/2 · ln(g_max/g))
    if g > 0.95 * gmax
        δ = sqrt(0.5 * (γ + 1.0) * log(gmax / g))
        M = supersonico ? 1.0 + δ : max(1.0 - δ, 0.0)
    end
    @inbounds for _ in 1:60
        q   = 1.0 + c * M * M
        qe1 = q^(-e - 1.0)                          # (1+cM²)^(−e−1)
        f   = M * qe1 * q - g                       # M·(1+cM²)^(−e) − g
        # Convergência verificada ANTES de mexer no intervalo: quando o chute já
        # é a solução (o caso comum), f = 0 e um passo de Newton nulo cairia na
        # borda do intervalo e dispararia uma bissecção para LONGE da raiz.
        abs(f) <= 4.0 * eps(g) && return M
        # f cresce com M no ramo subsônico e decresce no supersônico
        if supersonico
            f > 0.0 ? (lo = M) : (hi = M)
        else
            f > 0.0 ? (hi = M) : (lo = M)
        end
        df = qe1 * (1.0 - M * M)
        Mn = abs(df) > 1e-14 ? M - f / df : 0.5 * (lo + hi)
        (lo <= Mn <= hi) || (Mn = 0.5 * (lo + hi))      # Newton salvaguardado
        abs(Mn - M) <= 1e-15 * max(1.0, M) && return Mn
        M = Mn
    end
    return M
end

# Invariantes de equilíbrio de um estado primitivo W=(ρ,u,p) na área A.
@inline function _Q_de_W(W::SVector{3, Float64}, A::Float64, γ::Float64, R::Float64)
    ρ = max(W[1], 1e-6); u = W[2]; p = max(W[3], 1e-2)
    M  = abs(u) / sqrt(γ * p / ρ)
    q  = 1.0 + 0.5 * (γ - 1.0) * M * M
    T0 = (p / (ρ * R)) * q
    p0 = p * q^(γ / (γ - 1.0))
    return ρ * u * A, T0, p0, M
end

# Estado primitivo na área A a partir dos invariantes (ṁ, T₀, p₀), no ramo dado.
@inline function _W_de_Q(m::Float64, T0::Float64, p0::Float64, A::Float64,
                         supersonico::Bool, M_ini::Float64, γ::Float64, R::Float64,
                         gmax::Float64)::SVector{3, Float64}
    T0 = max(T0, 1.0); p0 = max(p0, 1e-2)
    ρ0 = p0 / (R * T0)
    a0 = sqrt(γ * R * T0)
    M  = _mach_de_vazao(abs(m) / (max(A, 1e-12) * ρ0 * a0), γ, gmax, supersonico, M_ini)
    q  = 1.0 + 0.5 * (γ - 1.0) * M * M
    T  = T0 / q
    p  = p0 * q^(-γ / (γ - 1.0))
    ρ  = p / (R * T)
    u  = copysign(M * sqrt(γ * R * T), m)
    return SVector{3, Float64}(ρ, u, p)
end

# Projeção de W (na área A_de) para a área A_para ao longo da onda estacionária.
@inline function _projetar_estado(W::SVector{3, Float64}, A_de::Float64, A_para::Float64,
                                  γ::Float64, R::Float64)::SVector{3, Float64}
    m, T0, p0, M = _Q_de_W(W, A_de, γ, R)
    return _W_de_Q(m, T0, p0, A_para, M >= 1.0, M, γ, R, _gmax_isen(γ))
end

"""
    termo_area_momento(W, A_e, A_c, A_d, γ, R, wb) -> Float64

Termo de área da equação da quantidade de movimento da célula (integral de
p·dA entre a face esquerda, área `A_e`, e a direita, `A_d`; `A_c` é a área do
centro). `wb=false`: esquema antigo p_i·(A_d − A_e). `wb=true`: ∫p·dA ao longo
da isentrópica da célula = (ρu²+p)A|d − (ρu²+p)A|e dos estados projetados
(com área plana, p_i·(A_d − A_e) — mesma regra das faces).
"""
@inline function termo_area_momento(W::SVector{3, Float64}, A_e::Float64, A_c::Float64, A_d::Float64,
                                    γ::Float64, R::Float64, wb::Bool)::Float64
    (wb && !_area_plana(A_e, A_c, A_d)) || return W[3] * (A_d - A_e)
    return termo_area_momento_Q(_Q_de_W(W, A_c, γ, R), A_e, A_d, γ, R, _gmax_isen(γ))
end

# Mesmo termo a partir dos invariantes já calculados da célula (cache do
# SimulationCore). Usa exatamente a mesma chamada _W_de_Q que os estados de
# face com inclinação nula — por isso o equilíbrio fecha bit a bit.
@inline function termo_area_momento_Q(Q::NTuple{4, Float64}, A_e::Float64, A_d::Float64,
                                      γ::Float64, R::Float64, gmax::Float64)::Float64
    m, T0, p0, M = Q
    Wd = _W_de_Q(m, T0, p0, A_d, M >= 1.0, M, γ, R, gmax)
    We = _W_de_Q(m, T0, p0, A_e, M >= 1.0, M, γ, R, gmax)
    return (Wd[1] * Wd[2]^2 + Wd[3]) * A_d - (We[1] * We[2]^2 + We[3]) * A_e
end

# Área da célula de índice idx, incluindo as fantasmas (espelho na cabeça,
# extrapolação na saída) — mesma convenção de obter_W_borda.
@inline function _A_celula(idx::Int, A_c::Vector{Float64}, N::Int)::Float64
    @inbounds begin
        1 <= idx <= N && return A_c[idx]
        idx == 0      && return A_c[1]
        idx == -1     && return A_c[min(2, N)]
        return A_c[N]
    end
end

# =========================================================================
# HELPER 3: Kernel de uma face — sem alocações, inline no hot loop
# Recebe os 4 estados vizinhos já extraídos e calcula o fluxo MUSCL.
# Separar isso do loop permite usar @fastmath sem afetar o roteamento de fronteira.
# =========================================================================
@inline @fastmath function _calcular_fluxo_face!(
    F_faces::Vector{SVector{3, Float64}},
    i::Int,
    W_LL::SVector{3, Float64},
    W_L ::SVector{3, Float64},
    W_R ::SVector{3, Float64},
    W_RR::SVector{3, Float64},
    R   ::Float64,
    cfg ::ConfigModelo,
    E_tab::Vector{Float64},
    T_tab::Vector{Float64},
    G_tab::Vector{Float64}
)::Int
    dW_L = SVector{3, Float64}(
        van_leer(W_R[1] - W_L[1], W_L[1] - W_LL[1]),
        van_leer(W_R[2] - W_L[2], W_L[2] - W_LL[2]),
        van_leer(W_R[3] - W_L[3], W_L[3] - W_LL[3])
    )

    dW_R = SVector{3, Float64}(
        van_leer(W_RR[1] - W_R[1], W_R[1] - W_L[1]),
        van_leer(W_RR[2] - W_R[2], W_R[2] - W_L[2]),
        van_leer(W_RR[3] - W_R[3], W_R[3] - W_L[3])
    )

    W_fL = W_L + 0.5 * dW_L
    W_fR = W_R - 0.5 * dW_R

    return _fluxo_de_estados_face!(F_faces, i, W_fL, W_fR, R, cfg, E_tab, T_tab, G_tab)
end

# Variante bem-balanceada: MUSCL/van Leer nos invariantes Q = (ṁ, T₀, p₀) de
# cada célula (calculados na área da própria célula) e projeção para a área da
# face, no regime (sub/supersônico) da célula de origem.
@inline @fastmath function _fluxo_face_wb_Q!(
    F_faces::Vector{SVector{3, Float64}},
    i::Int,
    QLL::NTuple{4, Float64}, QL::NTuple{4, Float64},
    QR ::NTuple{4, Float64}, QRR::NTuple{4, Float64},
    A_f::Float64,
    γ  ::Float64,
    R  ::Float64,
    gmax::Float64,
    cfg::ConfigModelo,
    E_tab::Vector{Float64},
    T_tab::Vector{Float64},
    G_tab::Vector{Float64}
)::Int
    mLL, T0LL, p0LL, _  = QLL
    mL,  T0L,  p0L,  ML = QL
    mR,  T0R,  p0R,  MR = QR
    mRR, T0RR, p0RR, _  = QRR

    mfL  = mL  + 0.5 * van_leer(mR  - mL,  mL  - mLL)
    T0fL = T0L + 0.5 * van_leer(T0R - T0L, T0L - T0LL)
    p0fL = p0L + 0.5 * van_leer(p0R - p0L, p0L - p0LL)

    mfR  = mR  - 0.5 * van_leer(mRR  - mR,  mR  - mL)
    T0fR = T0R - 0.5 * van_leer(T0RR - T0R, T0R - T0L)
    p0fR = p0R - 0.5 * van_leer(p0RR - p0R, p0R - p0L)

    W_fL = _W_de_Q(mfL, T0fL, p0fL, A_f, ML >= 1.0, ML, γ, R, gmax)
    W_fR = _W_de_Q(mfR, T0fR, p0fR, A_f, MR >= 1.0, MR, γ, R, gmax)

    return _fluxo_de_estados_face!(F_faces, i, W_fL, W_fR, R, cfg, E_tab, T_tab, G_tab)
end

# Estados primitivos de face (esq/dir) → estados conservativos → fluxo HLLC
# (ou Rusanov em gradiente extremo). Comum aos dois esquemas de reconstrução.
@inline @fastmath function _fluxo_de_estados_face!(
    F_faces::Vector{SVector{3, Float64}},
    i::Int,
    W_fL::SVector{3, Float64},
    W_fR::SVector{3, Float64},
    R   ::Float64,
    cfg ::ConfigModelo,
    E_tab::Vector{Float64},
    T_tab::Vector{Float64},
    G_tab::Vector{Float64}
)::Int
    rho_fL, u_fL, p_fL = W_fL[1], W_fL[2], max(1e-2, W_fL[3])
    rho_fR, u_fR, p_fR = W_fR[1], W_fR[2], max(1e-2, W_fR[3])

    T_fL = p_fL / (max(1e-6, rho_fL) * R)
    T_fR = p_fR / (max(1e-6, rho_fR) * R)

    e_int_fL = e_int_de_T(T_fL, E_tab, T_tab)
    e_int_fR = e_int_de_T(T_fR, E_tab, T_tab)

    UfL = SVector{3, Float64}(rho_fL, rho_fL * u_fL, rho_fL * (e_int_fL + 0.5 * u_fL^2))
    UfR = SVector{3, Float64}(rho_fR, rho_fR * u_fR, rho_fR * (e_int_fR + 0.5 * u_fR^2))

    grad_p = abs(p_fR - p_fL) / max(p_fL, p_fR, 1e-5)

    if grad_p > cfg.limiar_fallback_gradp || p_fL < cfg.limiar_fallback_p || p_fR < cfg.limiar_fallback_p
        @inbounds F_faces[i] = fluxo_rusanov(UfL, UfR, R, E_tab, T_tab, G_tab)
        return 1
    else
        @inbounds F_faces[i] = fluxo_hllc(UfL, UfR, R, E_tab, T_tab, G_tab)
        return 0
    end
end

# Laço de faces do esquema bem-balanceado. Invariantes de cada célula
# (inclusive as fantasmas −1, 0, N+1, N+2) calculados UMA vez: Qb[k+2] ↔
# célula k. O SimulationCore reutiliza o mesmo vetor no termo de área da
# célula (termo_area_momento_Q).
function _fluxos_faces_wb!(
    F_faces::Vector{SVector{3, Float64}},
    W::Vector{SVector{3, Float64}},
    malha::Malha1D, N::Int, P_atm::Float64, γ::Float64, R::Float64,
    cfg::ConfigModelo,
    E_tab::Vector{Float64}, T_tab::Vector{Float64}, G_tab::Vector{Float64},
    Qb::Vector{NTuple{4, Float64}}
)::Int
    A_c  = malha.A_centros
    A_f  = malha.A_faces
    gmax = _gmax_isen(γ)
    @inbounds for k in -1:(N + 2)
        Qb[k + 2] = _Q_de_W(obter_W_borda(k, W, N, P_atm, γ), _A_celula(k, A_c, N), γ, R)
    end
    fb = 0
    # face i: células i−2, i−1 | i, i+1  →  Qb[i], Qb[i+1] | Qb[i+2], Qb[i+3]
    @inbounds for i in 1:(N + 1)
        A_LL = _A_celula(i - 2, A_c, N); A_L  = _A_celula(i - 1, A_c, N)
        A_R  = _A_celula(i,     A_c, N); A_RR = _A_celula(i + 1, A_c, N)
        if _area_plana(A_LL, A_L, A_R, A_RR, A_f[i])
            # área uniforme em torno da face: reconstrução primitiva (exata aqui)
            fb += _calcular_fluxo_face!(F_faces, i,
                      obter_W_borda(i - 2, W, N, P_atm, γ), obter_W_borda(i - 1, W, N, P_atm, γ),
                      obter_W_borda(i,     W, N, P_atm, γ), obter_W_borda(i + 1, W, N, P_atm, γ),
                      R, cfg, E_tab, T_tab, G_tab)
        else
            fb += _fluxo_face_wb_Q!(F_faces, i, Qb[i], Qb[i + 1], Qb[i + 2], Qb[i + 3],
                                    A_f[i], γ, R, gmax, cfg, E_tab, T_tab, G_tab)
        end
    end
    return fb
end

# --- ATUALIZAÇÃO: Loop dividido — fronteira vs interior sem desvios ---
# Faces i=1..2 e i=N..N+1 precisam de ghost states (obter_W_borda).
# Faces i=3..N-1 acessam W[i-2..i+1] diretamente — sem branch, sem bounds check.
function calcular_fluxos_nas_faces!(
    F_faces::Vector{SVector{3, Float64}},
    U::Vector{SVector{3, Float64}},
    W::Vector{SVector{3, Float64}},
    malha::Malha1D,
    R::Float64,
    P_atm::Float64,
    cfg::ConfigModelo,
    E_tab::Vector{Float64},
    T_tab::Vector{Float64},
    G_tab::Vector{Float64};
    Q_cache::Union{Nothing, Vector{NTuple{4, Float64}}} = nothing
)
    N = malha.N
    fallbacks = 0

    # 1. Extrai o Gamma da última célula para a condição de saída
    _, _, _, gamma_N, _ = propriedades_locais(U[N], R, E_tab, T_tab, G_tab)

    if cfg.usar_termo_area_wb
        # Barreira de função: o laço roda num método com o cache de tipo concreto
        # (o Union{Nothing,…} do argumento opcional não chega ao laço quente).
        Qb = Q_cache === nothing ? Vector{NTuple{4, Float64}}(undef, N + 4) : Q_cache
        fallbacks += _fluxos_faces_wb!(F_faces, W, malha, N, P_atm, gamma_N, R, cfg,
                                       E_tab, T_tab, G_tab, Qb)
    else

    # ── Fronteira esquerda: faces i = 1 e 2 (precisam de estados fantasma) ──
    @inbounds for i in 1:2
        W_LL = obter_W_borda(i - 2, W, N, P_atm, gamma_N)
        W_L  = obter_W_borda(i - 1, W, N, P_atm, gamma_N)
        W_R  = obter_W_borda(i,     W, N, P_atm, gamma_N)
        W_RR = obter_W_borda(i + 1, W, N, P_atm, gamma_N)
        fallbacks += _calcular_fluxo_face!(F_faces, i, W_LL, W_L, W_R, W_RR, R, cfg, E_tab, T_tab, G_tab)
    end

    # ── Interior: faces i = 3..N-1 — acesso direto, zero desvios de fronteira ──
    @inbounds for i in 3:(N - 1)
        W_LL = W[i - 2]
        W_L  = W[i - 1]
        W_R  = W[i    ]
        W_RR = W[i + 1]
        fallbacks += _calcular_fluxo_face!(F_faces, i, W_LL, W_L, W_R, W_RR, R, cfg, E_tab, T_tab, G_tab)
    end

    # ── Fronteira direita: faces i = N..N+1 (precisam de estados fantasma).
    #    max(3, N) garante que não há sobreposição com o loop de fronteira esquerda
    #    quando N for muito pequeno (N < 3, situação improvável mas segura). ──
    @inbounds for i in max(3, N):(N + 1)
        W_LL = obter_W_borda(i - 2, W, N, P_atm, gamma_N)
        W_L  = obter_W_borda(i - 1, W, N, P_atm, gamma_N)
        W_R  = obter_W_borda(i,     W, N, P_atm, gamma_N)
        W_RR = obter_W_borda(i + 1, W, N, P_atm, gamma_N)
        fallbacks += _calcular_fluxo_face!(F_faces, i, W_LL, W_L, W_R, W_RR, R, cfg, E_tab, T_tab, G_tab)
    end

    end  # esquema de reconstrução

    # ── Propriedades da face de saída (para cálculo de empuxo) ──────────────
    W_saida_L = obter_W_borda(N,     W, N, P_atm, gamma_N)
    W_saida_R = obter_W_borda(N + 1, W, N, P_atm, gamma_N)

    p_exit   = 0.5 * (W_saida_L[3] + W_saida_R[3])
    rho_exit = 0.5 * (W_saida_L[1] + W_saida_R[1])
    u_exit   = 0.5 * (W_saida_L[2] + W_saida_R[2])

    return fallbacks, p_exit, rho_exit, u_exit
end

# =========================================================================
# Descarga sônica analítica na garganta (ver nota em ConfigModelo)
# Substitui o fluxo HLLC do rosto da garganta pelo fluxo de choke calculado do
# P₀ LOCAL a montante — corrige o erro de sub-resolução (c*_ef baixo) sem impor
# o 0D. Retorna sem alterar nada quando o escoamento não está estrangulado.
#
# Célula de referência para (P₀, T₀): NÃO a vizinha imediata da garganta, e sim a
# primeira célula a montante com A ≥ fator_ref·A_t (entrada do convergente, Mach
# baixo). Num convergente íngreme (45°), a célula colada à garganta atravessa uma
# razão de área ~3–4× numa só célula; o termo P·ΔA discretizado não é balanceado
# para isso e INFLA o P₀ reconstruído ali (+2–3% medido), o que fazia a garganta
# descarregar vazão demais e a câmara assentar abaixo do 0D (deriva de c*_ef
# 1,014→0,976 ao longo da queima, independente de malha). Como o convergente é
# isentrópico, o P₀ da garganta é o da sua entrada.
# Sem A_centros (chamada legada, e o que o SimulationCore passa quando o termo
# de área bem-balanceado está ligado), usa a célula idx_fg−1: com o esquema
# bem-balanceado o P₀ dela já é o correto, e o recuo não é necessário.
# =========================================================================
@inline @fastmath function _impor_fluxo_garganta!(
    F_faces::Vector{SVector{3, Float64}},
    idx_fg::Int,
    W::Vector{SVector{3, Float64}},
    gamma::Float64,
    R::Float64,
    P_atm::Float64,
    A_centros::Union{Nothing, Vector{Float64}} = nothing,
    A_t::Float64 = 0.0;
    fator_ref::Float64 = 4.0
)
    iu    = max(1, idx_fg - 1)                 # célula a montante (lado convergente)
    if A_centros !== nothing && A_t > 0.0
        @inbounds while iu > 1 && A_centros[iu] < fator_ref * A_t
            iu -= 1
        end
    end
    rho_u = max(W[iu][1], 1e-6)
    u_u   = W[iu][2]
    P_u   = max(W[iu][3], 1e-2)

    a_u = sqrt(gamma * P_u / rho_u)
    M_u = abs(u_u) / max(a_u, 1e-6)
    fac = 1.0 + 0.5 * (gamma - 1.0) * M_u^2
    P0  = P_u * fac^(gamma / (gamma - 1.0))     # estagnação local (invariante isentrópico)

    # Só impõe se realmente estrangulado (P₀ acima do crítico) e fluxo para +x.
    razao_crit = ((gamma + 1.0) / 2.0)^(gamma / (gamma - 1.0))
    (u_u > 0.0 && P0 >= razao_crit * P_atm) || return

    T_u = P_u / (rho_u * R)
    T0  = T_u * fac
    T_s = T0 * 2.0 / (gamma + 1.0)              # condições sônicas no rosto (choke)
    P_s = P0 * (2.0 / (gamma + 1.0))^(gamma / (gamma - 1.0))
    ρ_s = P_s / (R * T_s)
    a_s = sqrt(gamma * R * T_s)
    u_s = a_s
    e_s = R / (gamma - 1.0) * T_s               # e_int = Cv·T (GCP)
    E_s = ρ_s * (e_s + 0.5 * u_s^2)

    @inbounds F_faces[idx_fg] = SVector{3, Float64}(ρ_s * u_s, ρ_s * u_s^2 + P_s, u_s * (E_s + P_s))
    return
end

# (Revisão final) Removido o bloco comentado `calcular_dt` — o dt adaptativo (CFL)
# é calculado inline no laço principal (SimulationCore.jl). Código morto.