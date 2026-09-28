module ModuloGeometria

using LibGEOS
using Printf
# Referência dinâmica ao módulo pai:
#   • em scripts (include de Main):         _par = Main
#   • no pacote  (include de RktPrisma):     _par = RktPrisma
# Aliases são idênticos ao tipo original (mesma referência, não cópia) →
# dispatch de funções como build_geometry(spec::CylinderSpec) continua a funcionar.
const _par = parentmodule(@__MODULE__)
const CylinderSpec    = _par.CylinderSpec
const FinocylSpec     = _par.FinocylSpec
const StarSpec        = _par.StarSpec
const BatesSpec       = _par.BatesSpec
const EndBurnerSpec   = _par.EndBurnerSpec
const MoonBurnerSpec  = _par.MoonBurnerSpec
const WagonWheelSpec  = _par.WagonWheelSpec
const MultiPerfSpec   = _par.MultiPerfSpec
const GrainGeometryLUT = _par.GrainGeometryLUT
const GrainSegment    = _par.GrainSegment
const GrainLayout     = _par.GrainLayout
const UniformInterp1D = _par.UniformInterp1D

export gerar_interpoladores_cilindro, gerar_interpoladores_slots_geos, gerar_interpoladores_estrela
export gerar_interpoladores_end_burner, gerar_interpoladores_moonburner
export build_geometry, single_segment_layout, two_segment_layout, two_segment_layout_with_transition, validar_layout
export repeat_layout, repeat_alternating_layout, multi_segment_layout
export criar_slot_finocyl, hydraulic_diameter, validar_lut
export build_tapered_zone_lut, taper3d_layout


# =========================================================================
# FUNÇÃO AUXILIAR
# =========================================================================
function calcular_perimetro_real(geometria_vazio, borda_carcaca)
    borda_vazio = LibGEOS.boundary(geometria_vazio)
    contato_parede = LibGEOS.intersection(borda_vazio, borda_carcaca)

    P_total = LibGEOS.geomLength(borda_vazio)
    P_morto = LibGEOS.geomLength(contato_parede)

    return max(0.0, P_total - P_morto)
end

# =========================================================================
# ZONA 1: CILINDRO
# =========================================================================
function gerar_interpoladores_cilindro(D_ext_m::Float64, D_furo_m::Float64)
    R_ext = D_ext_m / 2.0
    R_furo = D_furo_m / 2.0
    web_max = R_ext - R_furo

    y_vetor = collect(range(0.0, stop=web_max * 1.05, length=100))

    A_list      = Float64[]
    P_list      = Float64[]
    P_wall_list = Float64[]   # para cilindro: superfície lateral queima → P_wall = 0

    for y in y_vetor
        r_atual = min(R_furo + y, R_ext)
        push!(A_list, π * r_atual^2)

        if r_atual < R_ext
            push!(P_list,      2π * r_atual)
            push!(P_wall_list, 0.0)
        else
            # Burnout: toda a seção é carcaça, P_burn → 0
            push!(P_list,      0.0)
            push!(P_wall_list, 2π * R_ext)
        end
    end

    # UniformInterp1D: zero alocação por chamada (substitui Interpolations.jl)
    interp_A      = UniformInterp1D(y_vetor, A_list)
    interp_P      = UniformInterp1D(y_vetor, P_list)
    interp_P_wall = UniformInterp1D(y_vetor, P_wall_list)

    # P_flux == P_burn + P_wall == 2π·r para cilindro (toda borda é molhada)
    return interp_A, interp_P, interp_P, interp_P_wall, web_max
end

# =========================================================================
# ZONA 2: FINOCYL / SLOTS
# =========================================================================

"""
    criar_slot_finocyl(theta, R_furo, W_slot, L_slot; shape, tip_radius,
                       root_radius, resolution) -> LibGEOS geometry

Cria um único slot finocyl já rotacionado por `theta` radianos.
Retorna uma geometria LibGEOS pronta para `union` com o furo central.

Shapes disponíveis:
  :rectangular  — 4 vértices, cantos vivos (comportamento original).
  :capsule      — buffer de linha radial → oblongo com extremidades arredondadas.
                  Equivale a um retângulo com semicírculos em raiz e ponta.
  :rounded_tip  — retangular com semicírculo apenas na ponta do slot.

Parâmetros:
  theta       — ângulo de rotação [rad]
  R_furo      — raio do porto circular [m]
  W_slot      — largura total do slot [m]
  L_slot      — profundidade radial do slot (além do furo) [m]
  tip_radius  — raio de arredondamento da ponta [m]; 0 → W_slot/2 (automático)
  root_radius — raio de fillet na raiz [m]; 0 → sem fillet
  resolution  — segmentos por quadrante no buffer LibGEOS
"""
function criar_slot_finocyl(
    theta      ::Float64,
    R_furo     ::Float64,
    W_slot     ::Float64,
    L_slot     ::Float64;
    shape      ::Symbol  = :rectangular,
    tip_radius ::Float64 = 0.0,
    root_radius::Float64 = 0.0,
    resolution ::Int     = 64,
)
    w     = W_slot / 2.0
    r_max = R_furo + L_slot
    r_tip = tip_radius <= 0.0 ? w : tip_radius

    rot(x, y) = (x * cos(theta) - y * sin(theta), x * sin(theta) + y * cos(theta))

    if shape === :rectangular
        # ── Comportamento original: 4 vértices, sem arredondamento ──────────
        rv1 = rot(-w, 0.0)
        rv2 = rot( w, 0.0)
        rv3 = rot( w, r_max)
        rv4 = rot(-w, r_max)
        coords = [[[rv1[1], rv1[2]], [rv2[1], rv2[2]],
                   [rv3[1], rv3[2]], [rv4[1], rv4[2]],
                   [rv1[1], rv1[2]]]]
        return LibGEOS.Polygon(coords)

    elseif shape === :capsule
        # ── Linha radial + buffer → oblongo com ambas extremidades arredondadas
        # A linha vai da raiz (R_furo) à ponta (r_max), no eixo local θ.
        # buffer(line, w, resolution) gera naturalmente a cápsula.
        # A parte que cai dentro do furo central desaparece ao fazer
        # union(geometria_base, slot) mais adiante.
        p1 = rot(0.0, R_furo)
        p2 = rot(0.0, r_max)
        line = LibGEOS.LineString([[p1[1], p1[2]], [p2[1], p2[2]]])
        return LibGEOS.buffer(line, w, resolution)

    elseif shape === :rounded_tip
        # ── Retangular + semicírculo na ponta ───────────────────────────────
        rv1 = rot(-w, 0.0)
        rv2 = rot( w, 0.0)
        rv3 = rot( w, r_max)
        rv4 = rot(-w, r_max)
        coords = [[[rv1[1], rv1[2]], [rv2[1], rv2[2]],
                   [rv3[1], rv3[2]], [rv4[1], rv4[2]],
                   [rv1[1], rv1[2]]]]
        rect = LibGEOS.Polygon(coords)

        # Semicírculo centrado na ponta (r_max, no eixo local θ)
        tip_pt = rot(0.0, r_max)
        tip_cap = LibGEOS.buffer(
            LibGEOS.Point(tip_pt[1], tip_pt[2]), r_tip, resolution
        )
        slot = LibGEOS.union(rect, tip_cap)

        # Fillet opcional na raiz
        if root_radius > 0.0
            root_pt = rot(0.0, R_furo)
            root_cap = LibGEOS.buffer(
                LibGEOS.Point(root_pt[1], root_pt[2]), root_radius, resolution
            )
            slot = LibGEOS.union(slot, root_cap)
        end
        return slot

    else
        error("criar_slot_finocyl: slot_shape desconhecido '$shape'. " *
              "Use :rectangular, :capsule ou :rounded_tip.")
    end
end

# Suaviza saltos artificiais de P_burn(y) nas transições topológicas das aletas do
# finocyl (parente do degrau do BATES). Aplica média móvel simétrica (janela ±`janela`,
# `npass` passadas) em P_flux e P_wall e RECOMPÕE P_burn = max(0, P_flux − P_wall) — o
# que preserva EXATAMENTE a consistência P_flux = P_burn + P_wall. Não toca A_port.
# Smoothing leve (afeta ~poucos pontos): de-sharpa a transição sem apagar o boost.
function _suavizar_perimetros!(P::Vector{Float64}, Pf::Vector{Float64},
                               Pw::Vector{Float64}; janela::Int = 2, npass::Int = 2)
    n = length(P)
    n < 2*janela + 3 && return nothing
    for arr in (Pf, Pw)
        for _ in 1:npass
            orig = copy(arr)
            @inbounds for i in 1:n
                lo = max(1, i - janela); hi = min(n, i + janela)
                s = 0.0
                for k in lo:hi
                    s += orig[k]
                end
                arr[i] = s / (hi - lo + 1)
            end
        end
    end
    @inbounds for i in 1:n
        P[i] = max(0.0, Pf[i] - Pw[i])   # recompõe → consistência P_flux=P_burn+P_wall exata
    end
    return nothing
end

function gerar_interpoladores_slots_geos(
    D_ext_m    ::Float64,
    D_furo_m   ::Float64,
    N_pontas   ::Int,
    W_slot_m   ::Float64,
    L_slot_m   ::Float64;
    passos     ::Int    = 300,
    slot_shape ::Symbol = :rectangular,
    tip_radius ::Float64 = 0.0,
    root_radius::Float64 = 0.0,
    resolution ::Int    = 64,
)
    R_ext  = D_ext_m  / 2.0
    R_furo = D_furo_m / 2.0

    web_max = R_ext - R_furo

    ponto_central = LibGEOS.Point(0.0, 0.0)
    carcaca       = LibGEOS.buffer(ponto_central, R_ext, 128)
    borda_carcaca = LibGEOS.boundary(carcaca)

    geometria_base = LibGEOS.buffer(ponto_central, R_furo, 128)

    for i in 1:N_pontas
        theta = (i - 1) * (2π / N_pontas)
        slot  = criar_slot_finocyl(
            theta, R_furo, W_slot_m, L_slot_m;
            shape       = slot_shape,
            tip_radius  = tip_radius,
            root_radius = root_radius,
            resolution  = resolution,
        )
        geometria_base = LibGEOS.union(geometria_base, slot)
    end

    geometria_base = LibGEOS.intersection(geometria_base, carcaca)

    y_vetor = range(0.0, stop=web_max * 1.05, length=passos)

    A_list      = Float64[]
    P_list      = Float64[]
    P_flux_list = Float64[]
    P_wall_list = Float64[]
    y_efetivo   = Float64[]

    for y in y_vetor
        queima_expandida = LibGEOS.buffer(geometria_base, y, 128)
        geometria_vazio  = LibGEOS.intersection(queima_expandida, carcaca)
        borda_vazio      = LibGEOS.boundary(geometria_vazio)

        # Uma única chamada a boundary() — usada para todos os perímetros
        perimetro_fluxo  = LibGEOS.geomLength(borda_vazio)
        contato_carcaca  = LibGEOS.intersection(borda_vazio, borda_carcaca)
        perimetro_wall   = LibGEOS.geomLength(contato_carcaca)
        perimetro_queima = max(0.0, perimetro_fluxo - perimetro_wall)
        A_p              = LibGEOS.area(geometria_vazio)

        push!(A_list,      A_p)
        push!(P_list,      perimetro_queima)
        push!(P_flux_list, perimetro_fluxo)
        push!(P_wall_list, perimetro_wall)
        push!(y_efetivo,   y)

        if perimetro_queima < 1e-5 && y > web_max * 0.5
            break
        end
    end

    # Guard: garante ao menos 2 pontos para UniformInterp1D (edge case defensivo)
    if length(y_efetivo) < 2
        push!(y_efetivo, web_max)
        push!(A_list,      π * R_ext^2)
        push!(P_list,      0.0)
        push!(P_flux_list, 2.0 * π * R_ext)
        push!(P_wall_list, 2.0 * π * R_ext)
    end

    # Suaviza saltos das transições das aletas (preserva P_flux = P_burn + P_wall).
    _suavizar_perimetros!(P_list, P_flux_list, P_wall_list)

    # y_efetivo é uniforme (vem de range(), com possível truncagem por burnout)
    interp_A      = UniformInterp1D(y_efetivo, A_list)
    interp_P      = UniformInterp1D(y_efetivo, P_list)
    interp_P_flux = UniformInterp1D(y_efetivo, P_flux_list)
    interp_P_wall = UniformInterp1D(y_efetivo, P_wall_list)

    return interp_A, interp_P, interp_P_flux, interp_P_wall, web_max
end

function desenhar_estrela(pontas::Int, raio_interno::Float64, raio_externo::Float64, angulo_fenda::Float64)
    coords = Vector{Vector{Float64}}()
    for i in 1:pontas
        theta_base = (i - 1) * (2 * pi / pontas)
        push!(coords, [raio_interno * cos(theta_base - angulo_fenda/2), raio_interno * sin(theta_base - angulo_fenda/2)])
        push!(coords, [raio_externo * cos(theta_base), raio_externo * sin(theta_base)])
        push!(coords, [raio_interno * cos(theta_base + angulo_fenda/2), raio_interno * sin(theta_base + angulo_fenda/2)])
    end
    push!(coords, coords[1])
    return [coords]
end

function gerar_interpoladores_estrela(
    D_ext_m::Float64;
    pontas::Int = 5,
    raio_fenda_m::Float64,
    raio_base_m::Float64,
    angulo_fenda::Float64 = 0.4,
    passos::Int = 400
)
    r_carcaca = D_ext_m / 2.0
    r_fenda = raio_fenda_m
    r_base = raio_base_m

    centro = LibGEOS.Point(0.0, 0.0)
    carcaca = LibGEOS.buffer(centro, r_carcaca, 64)
    borda_carcaca = LibGEOS.boundary(carcaca)
    area_maxima = LibGEOS.area(carcaca)
    
    coords_lista = desenhar_estrela(pontas, r_base, r_fenda, angulo_fenda)
    nucleo_bruto = LibGEOS.Polygon(coords_lista)

    raio_fillet = 0.0015
    nucleo_inicial = LibGEOS.buffer(nucleo_bruto, raio_fillet, 64)
    nucleo_inicial = LibGEOS.buffer(nucleo_inicial, -raio_fillet, 64)

    espessura_teia = r_carcaca - r_base
    dy = espessura_teia / passos

    y_vetor      = Float64[]
    A_vetor      = Float64[]
    P_vetor      = Float64[]
    P_wall_vetor = Float64[]
    y_max_real   = espessura_teia

    for i in 0:passos
        y_atual = i * dy

        nucleo_atual     = LibGEOS.buffer(nucleo_inicial, y_atual, 64)
        geometria_valida = LibGEOS.intersection(nucleo_atual, carcaca)
        borda_geometria  = LibGEOS.boundary(geometria_valida)
        contato_parede   = LibGEOS.intersection(borda_geometria, borda_carcaca)

        P_total = LibGEOS.geomLength(borda_geometria)
        P_morto = LibGEOS.geomLength(contato_parede)
        P_w     = max(0.0, P_total - P_morto)

        A_p = LibGEOS.area(geometria_valida)

        if A_p >= 0.999 * area_maxima
            P_w     = 0.0
            P_morto = P_total   # toda a borda é carcaça no burnout
            A_p     = area_maxima
            if y_max_real == espessura_teia
                y_max_real = y_atual
            end
        end

        push!(y_vetor,      y_atual)
        push!(A_vetor,      A_p)
        push!(P_vetor,      P_w)
        push!(P_wall_vetor, P_morto)
    end

    interp_A      = UniformInterp1D(y_vetor, A_vetor)
    interp_P      = UniformInterp1D(y_vetor, P_vetor)
    interp_P_wall = UniformInterp1D(y_vetor, P_wall_vetor)

    # P_flux para estrela: P_burn + P_wall = P_total (toda borda é molhada)
    P_flux_vetor = P_vetor .+ P_wall_vetor
    interp_P_flux = UniformInterp1D(y_vetor, P_flux_vetor)

    return interp_A, interp_P, interp_P_flux, interp_P_wall, y_max_real
end

"""
    build_geometry(spec) → GrainGeometryLUT

Constrói a tabela de geometria 1D (LUT) para um grão.
Aceita qualquer subtipo de `AbstractGeometrySpec`:

| Spec           | Geometria               |
|----------------|-------------------------|
| `CylinderSpec` | Cilindro inibido (BATES)|
| `BatesSpec`    | Cilindro com faces livres|
| `FinocylSpec`  | Finocyl (LibGEOS)       |
| `StarSpec`     | Estrela (LibGEOS)       |

O objeto retornado contém interpoladores para A_port(y), P_burn(y),
P_flux(y) e P_wall(y) ao longo da regressão radial y ∈ [0, y_max].

Normalmente não é chamado diretamente — use [`simular_caso`](@ref)
ou [`diagnostico_geometria`](@ref) que chamam `build_geometry` internamente.
"""
function build_geometry(spec::CylinderSpec)
    interp_A, interp_P, interp_PF, interp_PW, y_max = gerar_interpoladores_cilindro(
        spec.D_ext,
        spec.D_core
    )

    return GrainGeometryLUT(
        interp_A     = interp_A,
        interp_Pburn = interp_P,
        interp_Pflux = interp_PF,
        interp_Pwall = interp_PW,
        y_max = y_max,
        name  = :cylinder
    )
end

function build_geometry(spec::FinocylSpec)
    # ── Validação antes de invocar LibGEOS ──────────────────────────────────
    R_ext  = spec.D_ext  / 2.0
    R_core = spec.D_core / 2.0
    web    = R_ext - R_core
    web > 0.0 ||
        error("build_geometry(FinocylSpec): D_core ($(spec.D_core)) deve ser < D_ext ($(spec.D_ext)).")
    spec.n_fins >= 1 ||
        error("build_geometry(FinocylSpec): n_fins deve ser ≥ 1 (recebido: $(spec.n_fins)).")
    spec.fin_width > 0.0 ||
        error("build_geometry(FinocylSpec): fin_width deve ser positivo (recebido: $(spec.fin_width)).")
    spec.fin_length > 0.0 ||
        error("build_geometry(FinocylSpec): fin_length deve ser positivo (recebido: $(spec.fin_length)).")
    spec.fin_length < web ||
        error("build_geometry(FinocylSpec): fin_length ($(round(spec.fin_length*1e3,digits=2)) mm) " *
              "deve ser < espessura da teia ($(round(web*1e3,digits=2)) mm). " *
              "As aletas não podem atingir a carcaça.")
    perim_core = π * spec.D_core
    slot_total = spec.fin_width * spec.n_fins
    slot_total < perim_core ||
        error("build_geometry(FinocylSpec): $(spec.n_fins) aletas × fin_width " *
              "($(round(slot_total*1e3,digits=2)) mm total) deve ser < π·D_core " *
              "($(round(perim_core*1e3,digits=2)) mm). As aletas se sobreporiam.")
    # ── Construção ───────────────────────────────────────────────────────────
    interp_A, interp_P, interp_PF, interp_PW, y_max = gerar_interpoladores_slots_geos(
        spec.D_ext,
        spec.D_core,
        spec.n_fins,
        spec.fin_width,
        spec.fin_length;
        slot_shape  = spec.slot_shape,
        tip_radius  = spec.fin_tip_radius,
        root_radius = spec.fin_root_radius,
        resolution  = spec.geometry_resolution,
    )

    return GrainGeometryLUT(
        interp_A     = interp_A,
        interp_Pburn = interp_P,
        interp_Pflux = interp_PF,
        interp_Pwall = interp_PW,
        y_max = y_max,
        name  = :finocyl
    )
end

function build_geometry(spec::StarSpec)
    interp_A, interp_P, interp_PF, interp_PW, y_max = gerar_interpoladores_estrela(
        spec.D_ext;
        pontas       = spec.pontas,
        raio_fenda_m = spec.raio_fenda,
        raio_base_m  = spec.raio_base,
        angulo_fenda = spec.angulo_fenda,
        passos       = spec.passos
    )

    return GrainGeometryLUT(
        interp_A     = interp_A,
        interp_Pburn = interp_P,
        interp_Pflux = interp_PF,
        interp_Pwall = interp_PW,
        y_max = y_max,
        name  = :star
    )
end

function build_geometry(spec::BatesSpec)
    interp_A, interp_P, interp_PF, interp_PW, y_max = gerar_interpoladores_cilindro(
        spec.D_ext,
        spec.D_core
    )

    return GrainGeometryLUT(
        interp_A     = interp_A,
        interp_Pburn = interp_P,
        interp_Pflux = interp_PF,
        interp_Pwall = interp_PW,
        y_max = y_max,
        name  = :bates
    )
end

# =========================================================================
# END-BURNER
# =========================================================================
"""
    gerar_interpoladores_end_burner(D_ext_m) → (interp_A, interp_P, interp_PF, interp_PW, y_max)

LUT para grão end-burner:
  • A_port  = π·(D_ext/2)² em todo y  → bore pleno (gás preenche toda a secção)
  • P_burn  = 0 em todo y             → sem queima lateral
  • P_flux  = π·D_ext                  → para cálculo de D_h / atrito viscoso
  • y_max   = R_ext  (não usado na prática: a flag grain_type bloqueia regressão radial)
"""
function gerar_interpoladores_end_burner(D_ext_m::Float64)
    R_ext   = D_ext_m / 2.0
    A_cheia = π * R_ext^2
    P_ext   = 2π * R_ext

    # Dois pontos são suficientes para uma LUT constante (UniformInterp1D exige ≥ 2)
    y_vetor = [0.0, R_ext]
    interp_A  = UniformInterp1D(y_vetor, [A_cheia, A_cheia])
    interp_P  = UniformInterp1D(y_vetor, [0.0,     0.0    ])  # P_burn = 0
    interp_PF = UniformInterp1D(y_vetor, [P_ext,   P_ext  ])  # perímetro molhado
    interp_PW = UniformInterp1D(y_vetor, [P_ext,   P_ext  ])  # toda a parede é carcaça

    return interp_A, interp_P, interp_PF, interp_PW, R_ext
end

function build_geometry(spec::EndBurnerSpec)
    spec.D_ext > 0.0 || error("build_geometry(EndBurnerSpec): D_ext deve ser positivo.")

    interp_A, interp_P, interp_PF, interp_PW, y_max =
        gerar_interpoladores_end_burner(spec.D_ext)

    return GrainGeometryLUT(
        interp_A     = interp_A,
        interp_Pburn = interp_P,
        interp_Pflux = interp_PF,
        interp_Pwall = interp_PW,
        y_max        = y_max,
        name         = :end_burner
    )
end

# =========================================================================
# WAGON-WHEEL (roda) e MULTI-PERFURADO — geometrias adicionais (#9)
# =========================================================================
"""
    _lut_regressao_geos(geometria_base, carcaca, borda_carcaca, web_max; passos)
        → (interp_A, interp_P, interp_PF, interp_PW, y_max)

Core GENÉRICO de LUT por dilatação de Minkowski (LibGEOS): regride o porto inicial
`geometria_base` (polígono qualquer) até à carcaça, medindo A_port, P_queima,
P_fluxo e P_wall em cada web. Mesmo laço validado do finocyl/estrela — reutilizável
por qualquer geometria que saiba montar o seu polígono inicial.
"""
function _lut_regressao_geos(geometria_base, carcaca, borda_carcaca, web_max::Float64;
                             passos::Int = 300)
    y_vetor = range(0.0, stop = web_max * 1.05, length = passos)
    A_list=Float64[]; P_list=Float64[]; PF_list=Float64[]; PW_list=Float64[]; y_ef=Float64[]
    for y in y_vetor
        expandida = LibGEOS.buffer(geometria_base, y, 128)
        vazio     = LibGEOS.intersection(expandida, carcaca)
        borda     = LibGEOS.boundary(vazio)
        pf        = LibGEOS.geomLength(borda)
        contato   = LibGEOS.intersection(borda, borda_carcaca)
        pw        = LibGEOS.geomLength(contato)
        pq        = max(0.0, pf - pw)
        push!(A_list, LibGEOS.area(vazio)); push!(P_list, pq)
        push!(PF_list, pf); push!(PW_list, pw); push!(y_ef, y)
        (pq < 1e-5 && length(y_ef) > 5) && break
    end
    if length(y_ef) < 2
        R = sqrt(LibGEOS.area(carcaca) / π)
        push!(y_ef, web_max); push!(A_list, π*R^2); push!(P_list, 0.0)
        push!(PF_list, 2π*R); push!(PW_list, 2π*R)
    end
    _suavizar_perimetros!(P_list, PF_list, PW_list)
    return UniformInterp1D(y_ef, A_list), UniformInterp1D(y_ef, P_list),
           UniformInterp1D(y_ef, PF_list), UniformInterp1D(y_ef, PW_list), y_ef[end]
end

# ── Wagon-wheel: reusa a máquina de fendas do finocyl, mas com fenda FUNDA ─────
function build_geometry(spec::WagonWheelSpec)
    R_ext = spec.D_ext/2.0; R_core = spec.D_core/2.0; web = R_ext - R_core
    web > 0.0             || error("build_geometry(WagonWheelSpec): D_core deve ser < D_ext.")
    spec.n_spokes >= 2    || error("build_geometry(WagonWheelSpec): n_spokes deve ser ≥ 2.")
    spec.slot_width > 0.0 || error("build_geometry(WagonWheelSpec): slot_width deve ser > 0.")
    spec.slot_length > 0.0|| error("build_geometry(WagonWheelSpec): slot_length deve ser > 0.")
    spec.slot_length < web * 0.98 ||
        error("build_geometry(WagonWheelSpec): slot_length ($(round(spec.slot_length*1e3,digits=1)) mm) " *
              "deve ser < 0.98·web ($(round(web*0.98*1e3,digits=1)) mm).")
    interp_A, interp_P, interp_PF, interp_PW, y_max = gerar_interpoladores_slots_geos(
        spec.D_ext, spec.D_core, spec.n_spokes, spec.slot_width, spec.slot_length;
        slot_shape = spec.slot_shape, resolution = spec.geometry_resolution)
    return GrainGeometryLUT(interp_A=interp_A, interp_Pburn=interp_P,
        interp_Pflux=interp_PF, interp_Pwall=interp_PW, y_max=y_max, name=:wagon_wheel)
end

# ── Multi-perfurado: N círculos (1 central + N−1 num círculo de passo) ─────────
function build_geometry(spec::MultiPerfSpec)
    R_ext = spec.D_ext/2.0; R_perf = spec.D_perf/2.0
    spec.n_perf >= 1 || error("build_geometry(MultiPerfSpec): n_perf deve ser ≥ 1.")
    R_perf > 0.0     || error("build_geometry(MultiPerfSpec): D_perf deve ser > 0.")
    R_pitch = clamp(spec.pitch_frac, 0.0, 0.95) * R_ext
    n_anel  = spec.n_perf - 1
    (R_pitch + R_perf) < R_ext ||
        error("build_geometry(MultiPerfSpec): perfurações do anel atingem a carcaça " *
              "(R_pitch+R_perf = $(round((R_pitch+R_perf)*1e3,digits=1)) ≥ R_ext = $(round(R_ext*1e3,digits=1)) mm) — " *
              "reduza D_perf ou pitch_frac.")
    if n_anel >= 2
        gap = 2*R_pitch*sin(π/n_anel)   # distância centro-a-centro entre perifs vizinhas
        gap > 2*R_perf ||
            error("build_geometry(MultiPerfSpec): perfurações do anel se sobrepõem — " *
                  "reduza D_perf/n_perf ou aumente pitch_frac.")
    end
    centro        = LibGEOS.Point(0.0, 0.0)
    carcaca       = LibGEOS.buffer(centro, R_ext, 128)
    borda_carcaca = LibGEOS.boundary(carcaca)
    geometria_base = LibGEOS.buffer(centro, R_perf, 64)                  # perif central
    for i in 1:n_anel
        θ = (i-1) * (2π / n_anel)
        p = LibGEOS.Point(R_pitch*cos(θ), R_pitch*sin(θ))
        geometria_base = LibGEOS.union(geometria_base, LibGEOS.buffer(p, R_perf, 64))
    end
    interp_A, interp_P, interp_PF, interp_PW, y_max = _lut_regressao_geos(
        geometria_base, carcaca, borda_carcaca, R_ext; passos = spec.passos)
    return GrainGeometryLUT(interp_A=interp_A, interp_Pburn=interp_P,
        interp_Pflux=interp_PF, interp_Pwall=interp_PW, y_max=y_max, name=:multiperf)
end

# =========================================================================
# MOON-BURNER (porto excêntrico)
# =========================================================================
"""
    gerar_interpoladores_moonburner(D_ext_m, D_core_m, eccentricity; passos)
        → (interp_A, interp_P, interp_PF, interp_PW, y_max)

Usa LibGEOS para calcular A_port(y) e P_burn(y) de um porto circular deslocado.

O porto cresce concentricamente a partir de seu centro deslocado em `eccentricity`
do eixo do grão. A curva P_burn(y) é tipicamente **progressiva** (cresce antes
de cair), o que gera empuxo crescente — desejável para motores de aceleração.

`y_max` = `R_ext + e − R_core` = regressão para queima completa do lado mais distante.
"""
function gerar_interpoladores_moonburner(
    D_ext_m     ::Float64,
    D_core_m    ::Float64,
    eccentricity::Float64;
    passos      ::Int = 200
)
    R_ext  = D_ext_m  / 2.0
    R_core = D_core_m / 2.0
    e      = eccentricity

    R_core > 0.0            || error("MoonBurner: D_core deve ser positivo.")
    e >= 0.0                || error("MoonBurner: eccentricity deve ser ≥ 0.")
    e + R_core < R_ext - 1e-4 || error(
        "MoonBurner: eccentricity ($(round(e*1e3,digits=2)) mm) + R_core " *
        "($(round(R_core*1e3,digits=2)) mm) deve ser < R_ext " *
        "($(round(R_ext*1e3,digits=2)) mm). O porto ultrapassa a carcaça.")

    ponto_central = LibGEOS.Point(0.0, 0.0)
    carcaca       = LibGEOS.buffer(ponto_central, R_ext, 128)
    borda_carcaca = LibGEOS.boundary(carcaca)

    ponto_porto = LibGEOS.Point(e, 0.0)   # centro do porto deslocado

    # y_max = distância do lado oposto da carcaça ao porto expandido (burnout completo)
    y_max_lut = R_ext + e - R_core

    y_vetor     = collect(range(0.0, stop = y_max_lut * 1.02, length = passos))
    A_list      = Float64[]
    P_burn_list = Float64[]
    P_flux_list = Float64[]
    P_wall_list = Float64[]
    y_efetivo   = Float64[]

    A_ext_total = π * R_ext^2

    for y in y_vetor
        porto_expandido = LibGEOS.buffer(ponto_porto, R_core + y, 128)
        vazio           = LibGEOS.intersection(porto_expandido, carcaca)

        borda_vazio     = LibGEOS.boundary(vazio)
        contato_carcaca = LibGEOS.intersection(borda_vazio, borda_carcaca)

        P_flux = LibGEOS.geomLength(borda_vazio)
        P_wall = LibGEOS.geomLength(contato_carcaca)
        P_burn = max(0.0, P_flux - P_wall)
        A_p    = LibGEOS.area(vazio)

        push!(A_list,      A_p)
        push!(P_burn_list, P_burn)
        push!(P_flux_list, P_flux)
        push!(P_wall_list, P_wall)
        push!(y_efetivo,   y)

        # Para quando toda a seção transversal for vazio (grão consumido)
        if A_p >= A_ext_total * 0.999
            break
        end
    end

    # Garante ao menos 2 pontos para UniformInterp1D
    if length(y_efetivo) < 2
        push!(y_efetivo, y_max_lut)
        push!(A_list,      A_ext_total)
        push!(P_burn_list, 0.0)
        push!(P_flux_list, 2π * R_ext)
        push!(P_wall_list, 2π * R_ext)
    end

    interp_A  = UniformInterp1D(y_efetivo, A_list)
    interp_P  = UniformInterp1D(y_efetivo, P_burn_list)
    interp_PF = UniformInterp1D(y_efetivo, P_flux_list)
    interp_PW = UniformInterp1D(y_efetivo, P_wall_list)

    return interp_A, interp_P, interp_PF, interp_PW, y_max_lut
end

function build_geometry(spec::MoonBurnerSpec)
    R_ext  = spec.D_ext  / 2.0
    R_core = spec.D_core / 2.0
    e      = spec.eccentricity

    R_core > 0.0   || error("build_geometry(MoonBurnerSpec): D_core deve ser positivo.")
    e >= 0.0       || error("build_geometry(MoonBurnerSpec): eccentricity deve ser ≥ 0.")
    e + R_core < R_ext || error(
        "build_geometry(MoonBurnerSpec): porto não cabe no grão — " *
        "e=$(round(e*1e3,digits=2)) mm + R_core=$(round(R_core*1e3,digits=2)) mm " *
        "≥ R_ext=$(round(R_ext*1e3,digits=2)) mm.")

    interp_A, interp_P, interp_PF, interp_PW, y_max = gerar_interpoladores_moonburner(
        spec.D_ext, spec.D_core, spec.eccentricity; passos = spec.passos
    )

    return GrainGeometryLUT(
        interp_A     = interp_A,
        interp_Pburn = interp_P,
        interp_Pflux = interp_PF,
        interp_Pwall = interp_PW,
        y_max        = y_max,
        name         = :moonburner
    )
end

# ── Fase 1b: Funções de layout com tipos concretos ────────────────────────────
# Cada função agora retorna GrainLayout{S} com S concreto (ou Union pequena).
# Julia especializa obter_geom_em_xy para o tipo S exacto → zero boxing.

# CASO MAIS COMUM: 1 segmento, geometria homogénea → S = GrainSegment{T,T}
function single_segment_layout(geom::GrainGeometryLUT{T}, L_total::Float64) where T
    seg = GrainSegment(0.0, L_total, geom)          # GrainSegment{T,T}
    return GrainLayout{GrainSegment{T,T}}([seg], L_total)
end

# DOIS SEGMENTOS HOMOGÉNEOS (mesmo T) — ex: BATES com dois grãos de geometria igual
function two_segment_layout(
    geom1::GrainGeometryLUT{T1},
    x_split::Float64,
    geom2::GrainGeometryLUT{T2},
    L_total::Float64
) where {T1, T2}
    seg1 = GrainSegment(0.0,     x_split, geom1)    # GrainSegment{T1,T1}
    seg2 = GrainSegment(x_split, L_total, geom2)    # GrainSegment{T2,T2}
    if T1 === T2
        # Tipos iguais → vector homogéneo concreto (caso mais rápido)
        return GrainLayout{GrainSegment{T1,T1}}([seg1, seg2], L_total)
    else
        # Tipos diferentes → Union pequena: Julia faz union splitting (eficiente)
        S = Union{GrainSegment{T1,T1}, GrainSegment{T2,T2}}
        segs = S[seg1, seg2]
        return GrainLayout{S}(segs, L_total)
    end
end

"""
    two_segment_layout_with_transition(geom1, x_split, geom2, L_total; L_trans=0.0)

Igual a `two_segment_layout`, mas insere uma zona de transição cônica de comprimento
`L_trans` centrada em `x_split`. Dentro dessa zona, A_port e P_burn são interpolados
linearmente entre geom1 e geom2, simulando o chanfro real de fabricação.

- `L_trans = 0.0` → comportamento idêntico ao `two_segment_layout` (degrau abrupto).
- `L_trans = D_ext * 0.5` → transição de ~meio diâmetro externo (valor típico).
"""
function two_segment_layout_with_transition(
    geom1::GrainGeometryLUT{T1},
    x_split::Float64,
    geom2::GrainGeometryLUT{T2},
    L_total::Float64;
    L_trans::Float64 = 0.0
) where {T1, T2}
    if L_trans <= 0.0
        return two_segment_layout(geom1, x_split, geom2, L_total)
    end

    L_trans = min(L_trans, x_split * 0.9, (L_total - x_split) * 0.9)
    x_trans_start = x_split - L_trans / 2.0
    x_trans_end   = x_split + L_trans / 2.0

    # Tipos dos três segmentos possíveis:
    #   GrainSegment{T1,T1}  — puro geom1 (cilindro)
    #   GrainSegment{T1,T2}  — transição
    #   GrainSegment{T2,T2}  — puro geom2 (finocyl)
    # Union pequena (≤3 tipos) → Julia usa union splitting, sem heap allocation.
    S = Union{GrainSegment{T1,T1}, GrainSegment{T1,T2}, GrainSegment{T2,T2}}
    segs = S[]

    if x_trans_start > 1e-6
        push!(segs, GrainSegment(0.0, x_trans_start, geom1))           # puro T1
    end
    push!(segs, GrainSegment(x_trans_start, x_trans_end, geom1, geom2, true))  # transição
    if x_trans_end < L_total - 1e-6
        push!(segs, GrainSegment(x_trans_end, L_total, geom2))         # puro T2
    end

    return GrainLayout{S}(segs, L_total)
end

"""
    tapered_finocyl_layout(geom_cyl, geom_luts, L_cyl, L_total; L_trans=0.0) → GrainLayout

Monta layout aft-finocyl com **afilamento axial das aletas** (fin taper).

Em vez de usar um único LUT finocyl em toda a seção aft, divide-a em N sub-segmentos com
`fin_length` crescente da fronteira fore-aft (aletas curtas) até o extremo aft/bocal (aletas
plenas). Isso replica o chanfro 3D das extremidades das aletas e distribui no tempo a transição
Phase 1→Phase 2, eliminando o degrau abrupto de Pc observado na simulação 2D.

## Estrutura do layout resultante:

    [  fore cilíndrico  | L_trans | fin_1 | fin_2 | ... | fin_N ]
                                    ↑                        ↑
                              fin_length_min          fin_length_max

## Parâmetros:
- `geom_cyl`    : LUT da seção cilíndrica fore (tipo T_cyl)
- `geom_luts`   : vetor de N LUTs finocyl com fin_length crescente (tipo T_fin, todos iguais)
- `L_cyl`       : comprimento do segmento cilíndrico fore [m]
- `L_total`     : comprimento total do grão [m]
- `L_trans`     : comprimento da zona de transição cônica [m] (0 = degrau, default=0)

## Tipo de retorno:
`GrainLayout{Union{GrainSegment{T_cyl,T_cyl}, GrainSegment{T_cyl,T_fin}, GrainSegment{T_fin,T_fin}}}`
— apenas 3 tipos concretos → union splitting, zero heap allocation no hot loop.
"""
function tapered_finocyl_layout(
    geom_cyl  ::GrainGeometryLUT{T_cyl},
    geom_luts ::Vector{GrainGeometryLUT{T_fin}},
    L_cyl     ::Float64,
    L_total   ::Float64;
    L_trans   ::Float64 = 0.0
) where {T_cyl, T_fin}
    n_tap = length(geom_luts)
    n_tap >= 1 || error("tapered_finocyl_layout: geom_luts deve ter ≥ 1 elemento.")
    L_total > L_cyl || error("tapered_finocyl_layout: L_total ($L_total) deve ser > L_cyl ($L_cyl).")

    # Clamp da zona de transição para não ultrapassar os segmentos vizinhos
    L_trans = min(L_trans, L_cyl * 0.9, (L_total - L_cyl) * 0.9)

    x_trans_start = L_cyl - L_trans / 2.0
    x_trans_end   = L_cyl + L_trans / 2.0
    L_fin_real    = L_total - x_trans_end          # comprimento disponível para sub-segmentos

    # ── Tipos dos segmentos (≤ 3 tipos concretos → union splitting) ──────────
    S = Union{GrainSegment{T_cyl, T_cyl},
              GrainSegment{T_cyl, T_fin},
              GrainSegment{T_fin, T_fin}}
    segs = S[]

    # 1. Segmento cilíndrico fore (antes da zona de transição)
    if x_trans_start > 1e-9
        push!(segs, GrainSegment(0.0, x_trans_start, geom_cyl))
    end

    # 2. Zona de transição cônica cilindro → 1º sub-segmento finocyl
    if L_trans > 1e-9
        push!(segs, GrainSegment(x_trans_start, x_trans_end, geom_cyl, geom_luts[1], true))
    end

    # 3. Sub-segmentos finocyl com fin_length crescente
    L_seg = L_fin_real / n_tap
    x_cur = x_trans_end
    for k in 1:n_tap
        x_nxt = (k == n_tap) ? L_total : x_cur + L_seg
        push!(segs, GrainSegment(x_cur, x_nxt, geom_luts[k]))
        x_cur = x_nxt
    end

    return GrainLayout{S}(segs, L_total)
end

export tapered_finocyl_layout

"""
    repeat_layout(geom, N, L_grao) -> GrainLayout{GrainSegment{T,T}}

Repete a mesma geometria `N` vezes consecutivas ao longo do eixo axial.
Caso mais eficiente: vetor completamente homogéneo, zero boxing em acesso.

Uso típico: BATES com N grãos idênticos.
"""
function repeat_layout(geom::GrainGeometryLUT{T}, N::Int, L_grao::Float64) where T
    N >= 1     || error("repeat_layout: N deve ser >= 1 (recebido N=$N).")
    L_grao > 0 || error("repeat_layout: L_grao deve ser positivo (recebido L_grao=$L_grao).")

    # Vector{GrainSegment{T,T}} — tipo 100% concreto → acesso sem boxing
    segments = Vector{GrainSegment{T,T}}(undef, N)
    for i in 1:N
        segments[i] = GrainSegment((i-1)*L_grao, i*L_grao, geom)
    end
    return GrainLayout{GrainSegment{T,T}}(segments, N * L_grao)
end

"""
    repeat_alternating_layout(geom1, L1, geom2, L2, N) -> GrainLayout

Monta N repetições da sequência `(geom1, geom2)` com comprimentos `L1` e `L2`.
Retorna `GrainLayout` completamente type-stable (vetor concreto, zero boxing).

Uso típico: motor finocyl multi-grão — N segmentos cilíndricos (`geom1`) alternados
com N segmentos com aletas (`geom2`).
"""
function repeat_alternating_layout(
    geom1::GrainGeometryLUT{T}, L1::Float64,
    geom2::GrainGeometryLUT{T}, L2::Float64,
    N::Int
) where {T}
    N  >= 1 || error("repeat_alternating_layout: N deve ser ≥ 1 (recebido: $N).")
    L1 >  0 || error("repeat_alternating_layout: L1 deve ser positivo (recebido: $L1).")
    L2 >  0 || error("repeat_alternating_layout: L2 deve ser positivo (recebido: $L2).")

    S    = GrainSegment{T, T}
    segs = Vector{S}(undef, 2N)
    x    = 0.0
    for i in 1:N
        segs[2i-1] = GrainSegment(x,        x + L1, geom1)
        x += L1
        segs[2i]   = GrainSegment(x,        x + L2, geom2)
        x += L2
    end
    return GrainLayout{S}(segs, x)
end

"""
    multi_segment_layout(pairs) -> GrainLayout

Monta layout a partir de lista de tuplas `(comprimento, lut)`.
Nota: por usar GrainGeometryLUT sem parâmetro de tipo, o vector de segmentos
fica com elemento abstracto. Preferir single/two/repeat_layout quando possível.
"""
function multi_segment_layout(pairs::Vector{Tuple{Float64, GrainGeometryLUT}})
    isempty(pairs) && error("multi_segment_layout: a lista de pares não pode ser vazia.")

    # Fallback abstracto: mantido para compatibilidade com layouts heterogéneos complexos.
    # Para casos simples, usar repeat_layout ou single_segment_layout que são type-stable.
    segments = GrainSegment[]
    sizehint!(segments, length(pairs))
    x = 0.0
    for (i, (L, geom)) in enumerate(pairs)
        L > 0 || error("multi_segment_layout: comprimento do segmento $i deve ser positivo (L=$L).")
        push!(segments, GrainSegment(x, x + L, geom))
        x += L
    end
    return GrainLayout{eltype(segments)}(segments, x)
end

# =========================================================================
# GEOMETRIA ANALÍTICA 3D — ZONA DE CHANFRO CONTÍNUA
# =========================================================================

"""
    build_tapered_zone_lut(base_spec, fl_start; N_fl=15, N_y=300) → GrainGeometryLUT

Constrói uma LUT de geometria **efetiva** para a zona de chanfro 3D de um grão finocyl.

## Conceito
Na geometria 3D real, as aletas crescem de comprimento zero (ou `fl_start`) até o comprimento
pleno (`fin_length`) ao longo da zona de chanfro. Em vez de discretizar isso em N degraus
(que introduzem saltos na curva de pressão), esta função calcula a **integral espacial
contínua** da área de queima sobre toda a distribuição de comprimentos de aleta:

```
P_eff(y) = (1/(fl_max - fl_start)) ∫[fl_start..fl_max] P_burn(fl, y) dfl
         ≈ (1/N_fl) Σₖ P_burn(fl_k, y)    [regra do ponto médio, fl_k uniforme]
```

O resultado é uma única `GrainGeometryLUT` com transição Phase 1→2 **completamente suave**
(C¹ quase em todo ponto) sem nenhum artefato de degrau.

## Parâmetros
- `base_spec`  : `FinocylSpec` com os parâmetros da aleta plena (`fin_length` = fl_max)
- `fl_start`   : fração de `fin_length` no início do chanfro (0.0 = aleta zero; 0.3 = 30%)
- `N_fl`       : número de LUTs usados na integração numérica (≥ 3; padrão 15)
- `N_y`        : pontos de amostragem na direção y (padrão 300)

## Retorno
`GrainGeometryLUT{UniformInterp1D}` — mesmo tipo de todos os outros LUTs, type-stable.
Campo `y_max` = `web_max` = (D_ext − D_core) / 2, igual à geometria base.
"""
function build_tapered_zone_lut(
    base_spec  ::FinocylSpec,
    fl_start   ::Float64 = 0.0;
    N_fl       ::Int     = 15,
    N_y        ::Int     = 300,
)
    fl_max   = base_spec.fin_length
    fl_start = clamp(fl_start, 0.0, fl_max * 0.95)
    N_fl     = max(3, N_fl)
    web_max  = (base_spec.D_ext - base_spec.D_core) / 2.0

    # ── 1. Construir N_fl LUTs: fl_k no centro de cada intervalo (ponto médio) ──
    luts = Vector{GrainGeometryLUT{UniformInterp1D}}(undef, N_fl)
    for k in 1:N_fl
        α    = fl_start + (fl_max - fl_start) * ((k - 0.5) / N_fl)
        fl_k = max(1e-4, α)
        spec_k = FinocylSpec(
            D_ext               = base_spec.D_ext,
            D_core              = base_spec.D_core,
            n_fins              = base_spec.n_fins,
            fin_width           = base_spec.fin_width,
            fin_length          = fl_k,
            slot_shape          = base_spec.slot_shape,
            fin_tip_radius      = min(base_spec.fin_tip_radius, fl_k * 0.49),
            fin_root_radius     = base_spec.fin_root_radius,
            geometry_resolution = base_spec.geometry_resolution,
        )
        luts[k] = Base.invokelatest(build_geometry, spec_k)
        @printf("  [TaperZone3D] LUT %2d/%d:  fl = %5.1f mm  (y_max = %.1f mm)\n",
                k, N_fl, fl_k * 1e3, luts[k].y_max * 1e3)
    end

    # ── 2. Para cada y, integrar sobre todos os fl_k (regra do ponto médio) ──
    y_vec  = collect(range(0.0, stop = web_max, length = N_y))
    P_eff  = zeros(Float64, N_y)
    A_eff  = zeros(Float64, N_y)
    Pf_eff = zeros(Float64, N_y)
    Pw_eff = zeros(Float64, N_y)

    for k in 1:N_fl
        lut = luts[k]
        y_lut_max = lut.interp_Pburn.y_max   # fim dos dados do LUT k
        for j in 1:N_y
            yc = clamp(y_vec[j], 0.0, y_lut_max)
            P_eff[j]  += lut.interp_Pburn(yc)
            A_eff[j]  += lut.interp_A(yc)
            Pf_eff[j] += lut.interp_Pflux(yc)
            Pw_eff[j] += lut.interp_Pwall(yc)
        end
    end
    inv_N = 1.0 / Float64(N_fl)
    P_eff  .*= inv_N
    A_eff  .*= inv_N
    Pf_eff .*= inv_N
    Pw_eff .*= inv_N

    # ── 3. Montar LUT resultado (mesmo tipo que qualquer outro LUT) ──────────
    return GrainGeometryLUT(
        interp_A     = UniformInterp1D(y_vec, A_eff),
        interp_Pburn = UniformInterp1D(y_vec, P_eff),
        interp_Pflux = UniformInterp1D(y_vec, Pf_eff),
        interp_Pwall = UniformInterp1D(y_vec, Pw_eff),
        y_max        = web_max,
        name         = :finocyl_taper3d,
    )
end

"""
    build_conic_finocyl_segments(base_spec, D_core_head, D_core_aft, L_grao;
                                 fin_fraction = 1.0, N_seg = 12)
        → Vector{Tuple{Float64, GrainGeometryLUT}}

Finocyl CÔNICO: fatia o grão em `N_seg` fatias axiais e devolve os pares
(comprimento, LUT) prontos para [`multi_segment_layout`](@ref).

O furo varia linearmente de `D_core_head` (cabeça, x=0) a `D_core_aft` (bocal,
x=L_grao). A fração `fin_fraction` do comprimento — a de TRÁS, junto ao bocal —
é aletada; o trecho dianteiro é furo liso (mesma convenção do finocyl reto, que
usa cilíndrico à frente + aletado no aft).

Cada fatia carrega a sua própria teia `web = (D_ext − D(x))/2` e, portanto, o seu
próprio `y_max`. É daí que vem o PLATÔ NEUTRO: as fatias de furo maior esgotam a
teia primeiro e saem da conta de área de queima (o solver para a célula no `y_max`
local), contrabalançando o crescimento de perímetro das fatias que ainda queimam
— ideia do booster cônico, cf. Poppe et al., DLR/EUCASS.

Cada fatia é um segmento SIMPLES (não de transição) de propósito: em segmento de
transição o `y_max` vira o máximo dos dois LUTs, o que faria a fatia de teia menor
continuar queimando após o burnout (super-estimando área).
"""
function build_conic_finocyl_segments(
    base_spec   ::FinocylSpec,
    D_core_head ::Float64,
    D_core_aft  ::Float64,
    L_grao      ::Float64;
    fin_fraction::Float64 = 1.0,
    N_seg       ::Int     = 12,
    fl_ramp_zone::Float64 = 0.0,   # [m] rampa de profundidade da aleta (0 = aleta reta)
    fl_start_frac::Float64 = 0.3,  # profundidade inicial da rampa (fração de fin_length)
)
    D_ext = base_spec.D_ext
    dc_lo = min(D_core_head, D_core_aft)
    dc_hi = max(D_core_head, D_core_aft)
    dc_lo > 0.0   || error("build_conic_finocyl_segments: D_core deve ser > 0.")
    dc_hi < D_ext || error("build_conic_finocyl_segments: D_core ($dc_hi) deve ser < D_ext ($D_ext).")
    L_grao > 0.0  || error("build_conic_finocyl_segments: L_grao deve ser > 0.")

    ff    = clamp(fin_fraction, 0.0, 1.0)
    N_seg = max(4, N_seg)
    L_cyl = L_grao * (1.0 - ff)          # trecho liso (dianteiro, lado da cabeça)

    # Reparte as fatias entre os dois trechos na proporção dos comprimentos,
    # com pelo menos 1 fatia em cada trecho que de fato exista.
    N_cyl = L_cyl <= 1e-9        ? 0 : max(1, round(Int, N_seg * (1.0 - ff)))
    N_fin = (L_grao - L_cyl) <= 1e-9 ? 0 : max(1, N_seg - N_cyl)

    # Furo local no ponto médio de cada fatia
    Dx(x) = D_core_head + (D_core_aft - D_core_head) * clamp(x / L_grao, 0.0, 1.0)

    pares = Tuple{Float64, GrainGeometryLUT}[]
    sizehint!(pares, N_cyl + N_fin)

    # ── Trecho liso: cone sem aletas, da cabeça até L_cyl ─────────────────────
    for k in 1:N_cyl
        L_k   = L_cyl / N_cyl
        x_mid = (k - 0.5) * L_k
        lut   = Base.invokelatest(build_geometry,
                                  CylinderSpec(D_ext = D_ext, D_core = Dx(x_mid)))
        push!(pares, (L_k, lut))
    end

    # ── Trecho aletado: cone + aletas, de L_cyl até o bocal ───────────────────
    # Se fl_ramp_zone>0: a aleta RAMPA de fl_start_frac·fin_length (junto à junção com
    # o trecho cilíndrico) até a profundidade plena ao longo de fl_ramp_zone → transição
    # boost→sustain suave (aleta 3D). Sempre limitada pela teia local (web_k·0.9).
    L_zona_fin = L_grao - L_cyl
    fl_full    = base_spec.fin_length
    for k in 1:N_fin
        L_k   = L_zona_fin / N_fin
        x_mid = L_cyl + (k - 0.5) * L_k
        dc_k  = Dx(x_mid)
        web_k = (D_ext - dc_k) / 2.0
        dist  = x_mid - L_cyl                                # distância da JUNÇÃO
        fl_ramp = (fl_ramp_zone > 1e-6 && dist < fl_ramp_zone) ?
                  fl_full * (fl_start_frac + (1.0 - fl_start_frac) * (dist / fl_ramp_zone)) :
                  fl_full
        fl_k  = max(1e-4, min(fl_ramp, web_k * 0.9))          # rampa, limitada pela teia local
        spec_k = FinocylSpec(
            D_ext = D_ext, D_core = dc_k,
            n_fins = base_spec.n_fins, fin_width = base_spec.fin_width,
            fin_length = fl_k,
            slot_shape = base_spec.slot_shape,
            fin_tip_radius = min(base_spec.fin_tip_radius, fl_k * 0.49),
            fin_root_radius = base_spec.fin_root_radius,
            geometry_resolution = base_spec.geometry_resolution,
        )
        push!(pares, (L_k, Base.invokelatest(build_geometry, spec_k)))
    end

    isempty(pares) && error("build_conic_finocyl_segments: nenhuma fatia gerada.")
    return pares
end

"""
    build_cono_cil_finocyl_segments(base_spec, D_core, D_cil_ext, L_grao;
                                    fin_fraction=1.0, fin_no_bocal=true, N_seg=30)
        → Vector{Tuple{Float64, GrainGeometryLUT}}

Variante CONOCYL-FINOCYL modular: o trecho ALETADO tem furo CONSTANTE = `D_core`
(fatias idênticas), e o trecho CILÍNDRICO (liso) é um CONE cujo furo vai de `D_core`
na JUNÇÃO com as aletas até `D_cil_ext` na EXTREMIDADE livre. Assim o afilamento
fica SÓ no cilindro (que governa o sustain), desacoplado das aletas (que definem o
boost) — cf. grãos modulares cilíndrico+cônico+finocyl (literatura de boost-sustain).

`fin_no_bocal=true`  → aletado junto ao BOCAL (aft): [cilíndrico na cabeça … aletado no bocal].
`fin_no_bocal=false` → aletado na CABEÇA (fore): [aletado na cabeça … cilíndrico no bocal].

O achatamento do sustain vem do mesmo mecanismo do cone: as fatias cilíndricas de
furo MAIOR (extremidade, se `D_cil_ext>D_core`) esgotam a teia primeiro e saem da
conta de área. NOTA: no 0D (lumped) a orientação NÃO altera a área somada — só
importa no 1D / térmica / desenho.
"""
function build_cono_cil_finocyl_segments(
    base_spec   ::FinocylSpec,
    D_core      ::Float64,     # furo CONSTANTE do trecho aletado (= junção)
    D_cil_ext   ::Float64,     # furo na EXTREMIDADE livre do trecho cilíndrico
    L_grao      ::Float64;
    fin_fraction::Float64 = 1.0,
    fin_no_bocal::Bool    = true,
    N_seg       ::Int     = 30,
    fl_ramp_zone::Float64 = 0.0,   # [m] comprimento em que a ALETA RAMPA em profundidade
    fl_start_frac::Float64 = 0.3,  # profundidade inicial da rampa (fração de fin_length)
)
    D_ext = base_spec.D_ext
    max(D_core, D_cil_ext) < D_ext || error("build_cono_cil: D_core/D_cil_ext devem ser < D_ext.")
    (D_core > 0.0 && D_cil_ext > 0.0) || error("build_cono_cil: furos devem ser > 0.")
    L_grao > 0.0 || error("build_cono_cil: L_grao deve ser > 0.")

    ff    = clamp(fin_fraction, 0.0, 1.0)
    N_seg = max(4, N_seg)
    L_fin = L_grao * ff
    L_cyl = L_grao - L_fin
    N_fin = L_fin <= 1e-9 ? 0 : max(1, round(Int, N_seg * ff))
    N_cyl = L_cyl <= 1e-9 ? 0 : max(1, N_seg - N_fin)

    # ── Fatias ALETADAS (furo CONSTANTE D_core); ordem construída JUNÇÃO → EXTREMIDADE ──
    # Se fl_ramp_zone>0: a aleta RAMPA em profundidade de fl_start_frac·fin_length (junto à
    # junção) até fin_length plena ao longo de fl_ramp_zone → transição boost→sustain suave
    # (aleta 3D "deep finocyl"). fin_tip_radius/fin_root_radius arredondam a aleta.
    fin_pares = Tuple{Float64, GrainGeometryLUT}[]
    if N_fin > 0
        web_f  = (D_ext - D_core) / 2.0
        fl_max = min(base_spec.fin_length, web_f * 0.9)
        L_k    = L_fin / N_fin
        _lut_cache = Dict{Float64, GrainGeometryLUT}()   # reaproveita LUT p/ fl_k iguais
        for k in 1:N_fin
            dist = (k - 0.5) * L_k                         # distância da JUNÇÃO
            fl_k = (fl_ramp_zone > 1e-6 && dist < fl_ramp_zone) ?
                   fl_max * (fl_start_frac + (1.0 - fl_start_frac) * (dist / fl_ramp_zone)) :
                   fl_max
            fl_k = round(max(1e-4, fl_k), digits = 5)      # arredonda p/ cachear
            lut  = get!(_lut_cache, fl_k) do
                Base.invokelatest(build_geometry, FinocylSpec(
                    D_ext = D_ext, D_core = D_core,
                    n_fins = base_spec.n_fins, fin_width = base_spec.fin_width,
                    fin_length = fl_k, slot_shape = base_spec.slot_shape,
                    fin_tip_radius = min(base_spec.fin_tip_radius, fl_k * 0.49),
                    fin_root_radius = base_spec.fin_root_radius,
                    geometry_resolution = base_spec.geometry_resolution,
                ))
            end
            push!(fin_pares, (L_k, lut))
        end
    end

    # ── Fatias CILÍNDRICAS (cone: furo D_core na JUNÇÃO → D_cil_ext na EXTREMIDADE) ──
    cyl_pares = Tuple{Float64, GrainGeometryLUT}[]  # ordem: junção → extremidade
    if N_cyl > 0
        L_k = L_cyl / N_cyl
        for k in 1:N_cyl
            s   = (k - 0.5) / N_cyl                 # 0 na junção, 1 na extremidade
            D_k = D_core + (D_cil_ext - D_core) * s
            push!(cyl_pares, (L_k, Base.invokelatest(build_geometry,
                                                     CylinderSpec(D_ext = D_ext, D_core = D_k))))
        end
    end

    # ── Monta a ordem física cabeça → bocal conforme a orientação ─────────────
    # fin_pares está na ordem JUNÇÃO→EXTREMIDADE. Monta cabeça→bocal conforme a orientação:
    pares = fin_no_bocal ?
        vcat(reverse(cyl_pares), fin_pares) :          # cilíndrico(cabeça→junção) + aletado(junção→bocal)
        vcat(reverse(fin_pares), cyl_pares)            # aletado(cabeça→junção) + cilíndrico(junção→bocal)
    isempty(pares) && error("build_cono_cil_finocyl_segments: nenhuma fatia gerada.")
    return pares
end

"""
    taper3d_layout(geom_cyl, geom_taper, geom_fin, L_cyl, L_taper, L_total) → GrainLayout

Monta layout de **3 zonas** para grão finocyl com chanfro 3D analítico:

```
[  fore cilíndrico  |  zona chanfro 3D  |  finocyl aleta plena  ]
0                 L_cyl           L_cyl+L_taper              L_total
```

- `geom_cyl`   : LUT da seção cilíndrica dianteira
- `geom_taper` : LUT efetiva da zona de chanfro (`build_tapered_zone_lut`)
- `geom_fin`   : LUT da seção finocyl com aleta plena
- `L_cyl`      : comprimento da seção cilíndrica [m]
- `L_taper`    : comprimento da zona de chanfro [m]
- `L_total`    : comprimento total do grão [m]
"""
function taper3d_layout(
    geom_cyl  ::GrainGeometryLUT{T_cyl},
    geom_taper::GrainGeometryLUT{T_tap},
    geom_fin  ::GrainGeometryLUT{T_fin},
    L_cyl     ::Float64,
    L_taper   ::Float64,
    L_total   ::Float64,
) where {T_cyl, T_tap, T_fin}
    L_total > L_cyl + L_taper + 1e-9 ||
        error("taper3d_layout: L_total ($L_total) deve ser > L_cyl+L_taper ($(L_cyl+L_taper)).")
    L_taper > 1e-9 || error("taper3d_layout: L_taper deve ser > 0.")

    S = Union{GrainSegment{T_cyl, T_cyl},
              GrainSegment{T_tap, T_tap},
              GrainSegment{T_fin, T_fin}}
    segs = S[]

    # Zona 1: cilindro dianteiro (fore)
    if L_cyl > 1e-9
        push!(segs, GrainSegment(0.0, L_cyl, geom_cyl))
    end

    # Zona 2: chanfro 3D (LUT efetiva integrada)
    push!(segs, GrainSegment(L_cyl, L_cyl + L_taper, geom_taper))

    # Zona 3: finocyl com aleta plena
    push!(segs, GrainSegment(L_cyl + L_taper, L_total, geom_fin))

    return GrainLayout{S}(segs, L_total)
end

# =========================================================================
# UTILITÁRIOS DE GEOMETRIA
# =========================================================================

"""
    hydraulic_diameter(lut, y) -> Float64

Diâmetro hidráulico: D_h = 4·A_port / P_wetted [m]

Usa `interp_Pflux` como perímetro molhado (toda a borda do vazio é banhada
pelos gases de combustão, incluindo a parede da carcaça).
Retorna 0 se P_flux for nulo (burnout).
"""
@inline function hydraulic_diameter(lut::GrainGeometryLUT, y::Float64)
    Pf = lut.interp_Pflux(y)
    Pf < 1e-12 && return 0.0
    return 4.0 * lut.interp_A(y) / Pf
end

"""
    validar_lut(lut; tol_jump=0.5, throw_on_error=false) -> Bool

Verifica consistência da LUT de geometria após construção:
  1. Sem NaN/Inf em A_port, P_burn, P_flux, P_wall
  2. A_port monotonicamente não-decrescente (vazio só cresce com a queima)
  3. P_burn ≥ 0 e P_wall ≥ 0 em todos os pontos
  4. P_flux ≈ P_burn + P_wall (consistência interna, tolerância 1%)
  5. Sem saltos > `tol_jump`×100% consecutivos em P_burn
  6. y_max > 0

Parâmetros
----------
- `tol_jump`      : fração máxima de variação relativa entre pontos consecutivos em P_burn
- `throw_on_error`: se true lança erro; se false emite @warn e retorna false

Retorna `true` se tudo OK, `false` (ou lança) se houver anomalias.
"""
function validar_lut(
    lut            ::GrainGeometryLUT;
    tol_jump       ::Float64 = 0.5,
    throw_on_error ::Bool    = false,
)
    msgs = String[]

    # Acessar os vetores internos do interpolador
    As   = lut.interp_A.vals
    Ps   = lut.interp_Pburn.vals
    PFs  = lut.interp_Pflux.vals
    PWs  = lut.interp_Pwall.vals
    N    = length(As)
    y0   = lut.interp_A.y0
    ymax = lut.interp_A.y_max

    # 1. y_max positivo
    ymax > 0.0 ||
        push!(msgs, "y_max = $ymax deve ser positivo")

    # 2. NaN / Inf
    (any(isnan, As)  || any(isinf, As))  && push!(msgs, "NaN/Inf em A_port")
    (any(isnan, Ps)  || any(isinf, Ps))  && push!(msgs, "NaN/Inf em P_burn")
    (any(isnan, PFs) || any(isinf, PFs)) && push!(msgs, "NaN/Inf em P_flux")
    (any(isnan, PWs) || any(isinf, PWs)) && push!(msgs, "NaN/Inf em P_wall")

    # 3. Valores negativos
    any(x -> x < -1e-10, As)  && push!(msgs, "A_port negativo em algum ponto")
    any(x -> x < -1e-10, Ps)  && push!(msgs, "P_burn negativo em algum ponto")
    any(x -> x < -1e-10, PWs) && push!(msgs, "P_wall negativo em algum ponto")

    # 4. A_port monotonicamente não-decrescente
    for i in 2:N
        if As[i] < As[i-1] - 1e-9
            dy = (ymax - y0) / (N - 1)
            y_i = y0 + (i-1)*dy
            push!(msgs, @sprintf("A_port decresce em y≈%.4f m (%.4e → %.4e m²)",
                                 y_i, As[i-1], As[i]))
            break   # reportar só o primeiro para não poluir
        end
    end

    # 5. P_flux ≈ P_burn + P_wall (consistência, 1% de tolerância)
    for i in 1:N
        ref = PFs[i]
        ref < 1e-9 && continue
        err = abs(PFs[i] - (Ps[i] + PWs[i])) / ref
        if err > 0.01
            dy = (ymax - y0) / (N - 1)
            y_i = y0 + (i-1)*dy
            push!(msgs, @sprintf(
                "P_flux ≠ P_burn+P_wall em y≈%.4f m (Δ=%.1f %%)", y_i, err*100))
            break
        end
    end

    # 6. Saltos bruscos em P_burn — ignora a queda final (P→0 no burnout é esperado)
    for i in 2:N
        Ps[i-1] < 1e-9 && continue   # ponto anterior já é zero → burnout, ignorar
        Ps[i]   < 1e-9 && continue   # destino é zero → queda de burnout, esperada
        jump = abs(Ps[i] - Ps[i-1]) / Ps[i-1]
        if jump > tol_jump
            dy = (ymax - y0) / (N - 1)
            y_i = y0 + (i-1)*dy
            push!(msgs, @sprintf(
                "salto de %.0f%% em P_burn em y≈%.4f m", jump*100, y_i))
            break
        end
    end

    isempty(msgs) && return true

    msg = "validar_lut ($(lut.name)):\n" * join("  • " .* msgs, "\n")
    throw_on_error ? error(msg) : (@warn msg; return false)
end

function validar_layout(layout::GrainLayout)
    isempty(layout.segments) &&
        error("GrainLayout inválido: nenhum segmento definido.")

    x_prev = 0.0

    for (i, seg) in enumerate(layout.segments)
        seg.x_start > seg.x_end &&
            error("Segmento $i inválido: x_start ($(seg.x_start)) > x_end ($(seg.x_end)).")

        abs(seg.x_start - x_prev) > 1e-9 &&
            error("Segmento $i inválido: há buraco ou sobreposição no layout " *
                  "(x_start=$(seg.x_start), x_prev=$x_prev).")

        # Verificar y_max positivo em ambas as geometrias do segmento
        seg.geom_a.y_max > 0.0 ||
            error("Segmento $i: geom_a.y_max = $(seg.geom_a.y_max) deve ser positivo.")
        if seg.is_transition
            seg.geom_b.y_max > 0.0 ||
                error("Segmento $i (transição): geom_b.y_max = $(seg.geom_b.y_max) deve ser positivo.")
        end

        x_prev = seg.x_end
    end

    abs(x_prev - layout.L_total) > 1e-9 &&
        error("Layout inválido: último segmento termina em $x_prev, " *
              "mas L_total = $(layout.L_total).")
end

# =========================================================================
# CONTORNOS PARA CAD — polígono do vazio na mesma construção do solver
# =========================================================================
#
# Estas funções devolvem o MESMO polígono LibGEOS que alimenta as LUTs de
# A_port e P_burn. Exportar CAD a partir daqui garante que a figura do
# relatório é a geometria efectivamente simulada, e não um desenho paralelo
# que pode divergir dos números da tabela ao lado.
#
# Verificado: a área reconstruída do contorno bate com `LibGEOS.area` do
# polígono original com erro 0.0 (shoelace vs GEOS).

"""
    poligono_vazio(spec, y) -> LibGEOS.Geometry

Polígono do **vazio** (porto + fendas já queimadas) com web `y` [m], recortado
pela carcaça. `y = 0` dá a geometria inicial; `y = y_max` dá o fim da queima.

Reproduz exactamente a construção usada em `gerar_interpoladores_*`.
"""
function poligono_vazio end

function poligono_vazio(spec::CylinderSpec, y::Float64 = 0.0)
    centro  = LibGEOS.Point(0.0, 0.0)
    carcaca = LibGEOS.buffer(centro, spec.D_ext / 2.0, 128)
    base    = LibGEOS.buffer(centro, spec.D_core / 2.0, 128)
    return LibGEOS.intersection(LibGEOS.buffer(base, y, 128), carcaca)
end

poligono_vazio(spec::BatesSpec, y::Float64 = 0.0) =
    poligono_vazio(CylinderSpec(D_ext = spec.D_ext, D_core = spec.D_core), y)

function poligono_vazio(spec::FinocylSpec, y::Float64 = 0.0)
    R_ext  = spec.D_ext  / 2.0
    R_furo = spec.D_core / 2.0
    centro  = LibGEOS.Point(0.0, 0.0)
    carcaca = LibGEOS.buffer(centro, R_ext, 128)
    base    = LibGEOS.buffer(centro, R_furo, 128)
    for i in 1:spec.n_fins
        slot = criar_slot_finocyl(
            (i - 1) * (2π / spec.n_fins), R_furo, spec.fin_width, spec.fin_length;
            shape       = spec.slot_shape,
            tip_radius  = spec.fin_tip_radius,
            root_radius = spec.fin_root_radius,
            resolution  = spec.geometry_resolution,
        )
        base = LibGEOS.union(base, slot)
    end
    base = LibGEOS.intersection(base, carcaca)
    return LibGEOS.intersection(LibGEOS.buffer(base, y, 128), carcaca)
end

function poligono_vazio(spec::StarSpec, y::Float64 = 0.0)
    centro  = LibGEOS.Point(0.0, 0.0)
    carcaca = LibGEOS.buffer(centro, spec.D_ext / 2.0, 64)
    nucleo  = LibGEOS.Polygon(desenhar_estrela(spec.pontas, spec.raio_base,
                                               spec.raio_fenda, spec.angulo_fenda))
    # Mesmo fillet de 1.5 mm aplicado nas LUTs (abre e fecha o buffer)
    r_fil  = 0.0015
    nucleo = LibGEOS.buffer(LibGEOS.buffer(nucleo, r_fil, 64), -r_fil, 64)
    return LibGEOS.intersection(LibGEOS.buffer(nucleo, y, 64), carcaca)
end

function poligono_vazio(spec::MoonBurnerSpec, y::Float64 = 0.0)
    centro  = LibGEOS.Point(0.0, 0.0)
    carcaca = LibGEOS.buffer(centro, spec.D_ext / 2.0, 128)
    porto   = LibGEOS.buffer(LibGEOS.Point(spec.eccentricity, 0.0),
                             spec.D_core / 2.0, 128)
    return LibGEOS.intersection(LibGEOS.buffer(porto, y, 128), carcaca)
end

# End-burner queima pela FACE: não há vazio radial. Devolve polígono vazio para
# o chamador poder tratar o caso sem `if` espalhado.
# `readgeom("POLYGON EMPTY")` é a forma canónica — construir a partir de um
# vector de coordenadas vazio faz o LibGEOS estourar no índice.
poligono_vazio(::EndBurnerSpec, ::Float64 = 0.0) =
    LibGEOS.readgeom("POLYGON EMPTY")

"""
    aneis_de_wkt(wkt) -> Vector{Vector{Tuple{Float64,Float64}}}

Extrai os anéis de coordenadas de um WKT de POLYGON/MULTIPOLYGON.

Vai por WKT de propósito: os acessores de coordenadas do LibGEOS mudaram de
nome entre versões, enquanto o texto WKT é estável. Cada anel devolvido já vem
fechado (último vértice = primeiro), como o GEOS escreve.
"""
function aneis_de_wkt(wkt::AbstractString)
    aneis = Vector{Vector{Tuple{Float64,Float64}}}()
    for m in eachmatch(r"\(([-0-9eE\.\s,]+)\)", wkt)
        pts = Tuple{Float64,Float64}[]
        for par in split(m.captures[1], ',')
            v = split(strip(par))
            length(v) >= 2 && push!(pts, (parse(Float64, v[1]), parse(Float64, v[2])))
        end
        length(pts) > 3 && push!(aneis, pts)
    end
    return aneis
end

"""
    contorno_vazio(spec, y; escala=1.0) -> Vector{Vector{Tuple{Float64,Float64}}}

Anéis do vazio como listas de pontos. `escala = 1000.0` converte m → mm.

O primeiro anel é o contorno externo do vazio; anéis seguintes, se houver, são
ilhas de propelente dentro do porto (raro, mas possível em estrela com fendas
que se fecham).
"""
function contorno_vazio(spec, y::Float64 = 0.0; escala::Float64 = 1.0)
    poli = poligono_vazio(spec, y)
    LibGEOS.isEmpty(poli) && return Vector{Vector{Tuple{Float64,Float64}}}()
    aneis = aneis_de_wkt(LibGEOS.writegeom(poli))
    escala == 1.0 && return aneis
    return [[(p[1]*escala, p[2]*escala) for p in anel] for anel in aneis]
end

"""
    contorno_circulo(D, n=180; escala=1.0) -> Vector{Tuple{Float64,Float64}}

Círculo fechado de diâmetro `D`, para o contorno externo do grão.
"""
function contorno_circulo(D::Float64, n::Int = 180; escala::Float64 = 1.0)
    R = D / 2.0 * escala
    return [(R*cos(θ), R*sin(θ)) for θ in range(0, 2π; length = n + 1)]
end

export poligono_vazio, contorno_vazio, contorno_circulo, aneis_de_wkt

end