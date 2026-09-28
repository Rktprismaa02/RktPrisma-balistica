# ==========================================
# MÓDULO: GEOMETRIA (Câmara com LUT + Tubeira)
# Versão: Mega Otimizada (Type-Stable) com LibGEOS
# ==========================================


# ------------------------------------------------------------
# 1. FUNÇÕES DE MALHA E COORDENADAS 1D
# ------------------------------------------------------------
function criar_malha(N::Int, L::Float64)
    dx = L / N
    x_faces = collect(range(0, L, length=N+1))
    x_centros = x_faces[1:end-1] .+ dx/2.0
    A_faces = zeros(N+1)
    A_centros = zeros(N)
    return Malha1D(N=N, L=L, dx=dx, x_faces=x_faces, x_centros=x_centros, A_faces=A_faces, A_centros=A_centros)
end

# @inline força o processador a não ter overhead de chamada de função
@inline function get_y_queima_interp(x::Float64, dx::Float64, x_c1::Float64, x_cend::Float64, y_queima::Vector{Float64}, N::Int)
    if x <= x_c1
        return y_queima[1]
    elseif x >= x_cend
        return y_queima[N]
    end
    
    idx_float = (x - x_c1) / dx + 1.0
    # unsafe_trunc é a forma mais rápida que existe na CPU para pegar a base de um número
    idx_esq = Base.unsafe_trunc(Int, idx_float) 
    idx_dir = idx_esq + 1
    
    if idx_dir > N
        idx_dir = N
    end
    
    x_esq = x_c1 + (idx_esq - 1) * dx
    frac = (x - x_esq) / dx
    
    return y_queima[idx_esq] + frac * (y_queima[idx_dir] - y_queima[idx_esq])
end

@inline function perimetro_circular_por_area(A::Float64)
    return 2.0 * sqrt(pi * max(A, 1e-10))
end

# ── Helpers de grãos físicos ──────────────────────────────────────────────────
# Unificam o caso uniforme (grain_boundaries vazio → L_grao·N_graos de hoje) com o
# stack heterogêneo (grain_boundaries preenchido). Toda a lógica de faces deriva
# daqui, então o comportamento legado fica BIT-IDÊNTICO quando não há stack.
@inline n_graos_fisicos(prop::Propelente) =
    isempty(prop.grain_boundaries) ? prop.N_graos : length(prop.grain_boundaries) - 1

@inline comprimento_grao(g::Int, prop::Propelente) =
    isempty(prop.grain_boundaries) ? prop.L_grao :
        (prop.grain_boundaries[g + 1] - prop.grain_boundaries[g])

@inline comprimento_camara(prop::Propelente) =
    isempty(prop.grain_boundaries) ? prop.L_grao * prop.N_graos : prop.grain_boundaries[end]

# Célula com propelente = a célula SOBREPÕE a região do grão [0, L_camara].
# O critério antigo (centro ≤ L_camara) descartava a célula de borda cujo centro cai
# além do fim do grão, e o trecho de grão dentro dela nunca queimava (até dx/2 de
# comprimento de queima: −0,9 % de A_b no motor de 3 m com N=100, erro que muda de
# sinal com o alinhamento da malha). A área dentro da célula vem de
# calcular_area_queima_celula (sobreposição célula∩grão), então incluir a célula
# de borda não conta nada além do grão.
@inline tem_propelente(i::Int, malha, L_camara::Float64) = malha.x_faces[i] < L_camara

# Faces inibidas do grão físico g (0/1/2/3). Vazio = inhibited_ends global (legado/single).
@inline _inh_grao(g::Int, prop::Propelente) =
    isempty(prop.grain_inhibited) ? prop.inhibited_ends : prop.grain_inhibited[g]

# Grão físico que contém x (busca por fronteira — análogo a ceil(x/L_grao) clamp,
# mas para comprimentos arbitrários). Só usado no caminho stack (grain_boundaries≠∅).
@inline function _grao_em_x(x::Float64, prop::Propelente)
    gb = prop.grain_boundaries
    @inbounds for g in 1:length(gb) - 1
        x < gb[g + 1] && return g
    end
    return length(gb) - 1
end

@inline function limites_grao_ativo(g::Int, prop::Propelente{L}, y_axial_faces::Vector{Float64}) where {L}
    if isempty(prop.grain_boundaries)
        x_ini = (g - 1) * prop.L_grao + y_axial_faces[2g - 1]   # uniforme (legado)
        x_fim =  g      * prop.L_grao - y_axial_faces[2g]
    else
        x_ini = prop.grain_boundaries[g]     + y_axial_faces[2g - 1]   # stack heterogêneo
        x_fim = prop.grain_boundaries[g + 1] - y_axial_faces[2g]
    end
    return x_ini, x_fim
end

function encontrar_grao_ativo(x::Float64, prop::Propelente{L}, y_axial_faces::Vector{Float64}) where {L}
    melhor_g   = 0
    melhor_dist = Inf
    @inbounds for g in 1:n_graos_fisicos(prop)
        x_ini, x_fim = limites_grao_ativo(g, prop, y_axial_faces)
        if x_fim > x_ini
            if x >= x_ini && x <= x_fim
                return g, x_ini, x_fim   # dentro do grão — retorna direto
            end
            # guarda o grão mais próximo para o fallback
            dist = min(abs(x - x_ini), abs(x - x_fim))
            if dist < melhor_dist
                melhor_dist = dist
                melhor_g    = g
            end
        end
    end
    # fallback: x está no espaço entre grãos (imprecisão numérica ou espaçador).
    # Retorna o grão mais próximo para evitar retornar g=0 e cortar a queima.
    if melhor_g > 0
        x_ini, x_fim = limites_grao_ativo(melhor_g, prop, y_axial_faces)
        return melhor_g, x_ini, x_fim
    end
    return 0, 0.0, 0.0
end

@inline function _geom_from_seg_pure(g::GrainGeometryLUT{<:Any}, y_eff::Float64)
    return g.interp_A(y_eff), g.interp_Pburn(y_eff), g.interp_Pflux(y_eff), g.y_max
end

@inline function _geom_from_seg_trans(ga::GrainGeometryLUT{<:Any}, gb::GrainGeometryLUT{<:Any}, y_eff::Float64, frac::Float64)
    A  = ga.interp_A(y_eff)     + frac * (gb.interp_A(y_eff)     - ga.interp_A(y_eff))
    Pb = ga.interp_Pburn(y_eff) + frac * (gb.interp_Pburn(y_eff) - ga.interp_Pburn(y_eff))
    Pf = ga.interp_Pflux(y_eff) + frac * (gb.interp_Pflux(y_eff) - ga.interp_Pflux(y_eff))
    return A, Pb, Pf, min(ga.y_max, gb.y_max)
end

# Retorna (A, P_burn, P_flux, y_max) sem nenhuma alocação.
# O truque: usar @generated ou múltiplos métodos por tipo concreto para
# que o compilador especialize e elimine o Union dispatch.
@inline function obter_geom_em_xy(x::Float64, y::Float64, prop::Propelente{L}) where {L}
    # Fase 2: prop::Propelente{L} → Julia gera uma especialização por tipo de layout.
    # segs::Vector{S} (S concreto por GrainLayout{S}) → zero boxing, zero dispatch.
    segs    = prop.layout.segments
    y_max_g = prop.y_max
    # Busca linear — para N_segs pequeno (2-4) é mais rápido que busca binária
    @inbounds for k in eachindex(segs)
        seg = segs[k]
        if seg.x_start <= x <= seg.x_end
            if !seg.is_transition
                y_eff = clamp(y, 0.0, min(y_max_g, seg.geom_a.y_max))
                return _geom_from_seg_pure(seg.geom_a, y_eff)
            else  # GrainTransitionSegment
                frac  = clamp((x - seg.x_start) / (seg.x_end - seg.x_start), 0.0, 1.0)
                y_max_local = min(seg.geom_a.y_max, seg.geom_b.y_max)
                y_eff = clamp(y, 0.0, min(y_max_g, y_max_local))
                return _geom_from_seg_trans(seg.geom_a, seg.geom_b, y_eff, frac)
            end
        end
    end
    # Fallback: usar extremo mais próximo
    seg = x < segs[1].x_start ? segs[1] : segs[end]
    if !seg.is_transition
        y_eff = clamp(y, 0.0, min(y_max_g, seg.geom_a.y_max))
        return _geom_from_seg_pure(seg.geom_a, y_eff)
    else
        frac  = clamp((x - seg.x_start) / (seg.x_end - seg.x_start), 0.0, 1.0)
        y_max_local = min(seg.geom_a.y_max, seg.geom_b.y_max)
        y_eff = clamp(y, 0.0, min(y_max_g, y_max_local))
        return _geom_from_seg_trans(seg.geom_a, seg.geom_b, y_eff, frac)
    end
end

@inline function obter_ymax_local(x::Float64, prop::Propelente{L}) where {L}
    segs = prop.layout.segments
    @inbounds for k in eachindex(segs)
        seg = segs[k]
        if seg.x_start <= x <= seg.x_end
            return !seg.is_transition ? seg.geom_a.y_max : min(seg.geom_a.y_max, seg.geom_b.y_max)
        end
    end
    seg = x < segs[1].x_start ? segs[1] : segs[end]
    return !seg.is_transition ? seg.geom_a.y_max : min(seg.geom_a.y_max, seg.geom_b.y_max)
end

# Mantido para compatibilidade com atualizar_geometria_camara
@inline function geom_segment_at_x(x::Float64, layout::GrainLayout)
    @inbounds for seg in layout.segments
        if seg.x_start <= x <= seg.x_end
            return seg
        end
    end
    return x < layout.segments[1].x_start ? layout.segments[1] : layout.segments[end]
end


function calcular_secao_camara_local(
    x::Float64,
    malha::Malha1D,
    prop::Propelente{L},      # Fase 2
    y_queima::Vector{Float64},
    y_axial_faces::Vector{Float64}
) where {L}
    N = malha.N
    dx = malha.dx
    x_c1 = malha.x_centros[1]
    x_cend = malha.x_centros[end]

    A_secao_cheia = (pi / 4.0) * prop.D_ext^2
    P_secao_cheia = pi * prop.D_ext

    g, _, _ = encontrar_grao_ativo(x, prop, y_axial_faces)
    if g == 0
        return A_secao_cheia, P_secao_cheia, 0.0, prop.y_max
    end

    y_local = get_y_queima_interp(x, dx, x_c1, x_cend, y_queima, N)
    A_port_raw, P_burn, P_flux, y_max_local = obter_geom_em_xy(x, y_local, prop)
    A_port = clamp(A_port_raw, 1e-10, A_secao_cheia)
    P_burn = max(P_burn, 0.0)
    P_flux = max(P_flux, 1e-8)

    return A_port, P_flux, P_burn, clamp(y_local, 0.0, min(prop.y_max, y_max_local))
end

function calcular_geometria_escoamento_local(
    i::Int,
    malha::Malha1D,
    prop::Propelente{L},      # Fase 2
    y_queima::Vector{Float64},
    y_axial_faces::Vector{Float64}
) where {L}
    x = malha.x_centros[i]
    L_camara = comprimento_camara(prop)

    if x <= L_camara
        return calcular_secao_camara_local(x, malha, prop, y_queima, y_axial_faces)
    end

    A_fluxo = max(malha.A_centros[i], 1e-8)
    P_fluxo = perimetro_circular_por_area(A_fluxo)
    return A_fluxo, P_fluxo, 0.0, prop.y_max
end


# ------------------------------------------------------------
# 2. ATUALIZAÇÃO DA GEOMETRIA NO CFD (CHAMADO A CADA PASSO)
# ------------------------------------------------------------
# ------------------------------------------------------------
# ATUALIZAÇÃO DA GEOMETRIA: TUBEIRA (FIXA)
# Deve ser chamada UMA VEZ na inicialização e novamente
# apenas se D_garganta mudar (erosão ativa).
# A tubeira não tem propelente — sua geometria só muda
# com erosão da garganta, não com y_queima.
# ------------------------------------------------------------
function atualizar_geometria_tubeira!(
    malha::Malha1D,
    prop::Propelente{L},      # Fase 2
    x_garganta::Float64,
    D_garganta::Float64,
    D_saida::Float64
) where {L}
    L_camara  = comprimento_camara(prop)
    R_garganta = D_garganta / 2.0
    R_saida    = D_saida    / 2.0
    L_conv = x_garganta - L_camara
    L_div  = malha.L    - x_garganta

    L_conv > 0.0 || error("Geometria inválida: x_garganta deve ser maior que L_camara.")
    L_div  > 0.0 || error("Geometria inválida: L_total deve ser maior que x_garganta.")

    # Raio no fim da câmara (sem propelente = seção cheia)
    R_camara_fim = prop.D_ext / 2.0

    @inline function area_tubeira(x::Float64)
        if x <= x_garganta
            frac = (x - L_camara) / L_conv
            r = R_camara_fim - frac * (R_camara_fim - R_garganta)
            return pi * r^2
        else
            frac = (x - x_garganta) / L_div
            r = R_garganta + frac * (R_saida - R_garganta)
            return pi * r^2
        end
    end

    @inbounds for i in 1:malha.N
        x_c = malha.x_centros[i]
        if x_c > L_camara
            malha.A_centros[i] = area_tubeira(x_c)
        end
    end
    @inbounds for i in 1:(malha.N + 1)
        x_f = malha.x_faces[i]
        if x_f > L_camara
            malha.A_faces[i] = area_tubeira(x_f)
        end
    end
end

# ------------------------------------------------------------
# ATUALIZAÇÃO DA GEOMETRIA: CÂMARA (DINÂMICA)
# Chamada todo passo do solver — só toca as células
# com propelente (x_centro <= L_camara).
# ------------------------------------------------------------
function atualizar_geometria_camara!(
    malha::Malha1D,
    prop::Propelente{L},      # Fase 2
    y_queima::Vector{Float64},
    y_axial_faces::Vector{Float64}
) where {L}
    L_camara     = comprimento_camara(prop)
    A_secao_cheia = (pi / 4.0) * prop.D_ext^2

    @inbounds for i in 1:malha.N
        x_c = malha.x_centros[i]
        if x_c <= L_camara
            A_local, _, _, _ = calcular_secao_camara_local(x_c, malha, prop, y_queima, y_axial_faces)
            malha.A_centros[i] = clamp(A_local, 1e-10, A_secao_cheia)
        end
    end
    @inbounds for i in 1:(malha.N + 1)
        x_f = malha.x_faces[i]
        if x_f <= L_camara
            A_local, _, _, _ = calcular_secao_camara_local(x_f, malha, prop, y_queima, y_axial_faces)
            malha.A_faces[i] = clamp(A_local, 1e-10, A_secao_cheia)
        end
    end
end

# ------------------------------------------------------------
# WRAPPER DE COMPATIBILIDADE: atualizar_geometria!
# Mantido para não quebrar nenhuma chamada legada.
# Internamente delega para câmara + tubeira.
# ------------------------------------------------------------
function atualizar_geometria!(
    malha::Malha1D,
    prop::Propelente{L},      # Fase 2
    y_queima::Vector{Float64},
    y_axial_faces::Vector{Float64},
    x_garganta::Float64,
    D_garganta::Float64,
    D_saida::Float64
) where {L}
    atualizar_geometria_camara!(malha, prop, y_queima, y_axial_faces)
    atualizar_geometria_tubeira!(malha, prop, x_garganta, D_garganta, D_saida)
end