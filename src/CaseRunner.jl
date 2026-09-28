# ==============================================================================
# CaseRunner.jl — Funções de construção e execução de casos de simulação
# ==============================================================================
# Extraído de main.jl para ser reutilizável por gui.jl (e futuros scripts).
# Contém: build_case_geometry, build_propellant_from_case, build_domain_geometry,
#         validar_case_input, validar_geometria_caso, simular_caso
# ==============================================================================

# Validação do stack (estilo OpenMotor): propelente + cada grão + tubeira (Σ L).
function _validar_case_stack(inp::CaseInput)
    (0.0 < inp.n < 1.0) || error("Expoente de queima n=$(inp.n) fora de (0,1).")
    inp.a     > 0.0 || error("Coeficiente de queima a=$(inp.a) deve ser > 0.")
    inp.rho_p > 0.0 || error("Densidade ρ_p=$(inp.rho_p) deve ser > 0.")
    inp.Tc    > 0.0 || error("Temperatura de chama Tc=$(inp.Tc) deve ser > 0.")
    inp.R     > 0.0 || error("Constante do gás R=$(inp.R) deve ser > 0.")

    inp.D_ext > 0.0 || error("D_ext (carcaça) deve ser > 0.")
    for (i, g) in enumerate(inp.grains)
        g.L > 0.0 || error("Grão $i do stack: L=$(g.L) deve ser > 0.")
        g.tipo in (:bates, :finocyl, :star, :moonburner) ||
            error("Grão $i: tipo $(g.tipo) não suportado no stack " *
                  "(use :bates/:finocyl/:star/:moonburner; end-burner só fora de stack).")
        if g.tipo in (:bates, :finocyl, :moonburner)
            (0.0 < g.D_core < inp.D_ext) ||
                error("Grão $i ($(g.tipo)): precisa 0 < D_core ($(round(g.D_core*1e3))mm) " *
                      "< D_ext da carcaça ($(round(inp.D_ext*1e3))mm).")
        end
        if g.tipo == :finocyl
            (g.n_fins > 0 && g.fin_width > 0 && g.fin_length > 0) ||
                error("Grão $i (finocyl): n_fins, fin_width e fin_length devem ser > 0.")
            web = (inp.D_ext - g.D_core) / 2.0
            g.fin_length < web ||
                error("Grão $i (finocyl): fin_length=$(round(g.fin_length*1e3))mm " *
                      "deve ser < web=$(round(web*1e3))mm (carcaça $(round(inp.D_ext*1e3))mm).")
        end
    end

    L_cam = sum(g.L for g in inp.grains)
    inp.x_garganta > L_cam ||
        error("x_garganta=$(inp.x_garganta) deve ser > comprimento total do stack ($(round(L_cam,digits=3)) m).")
    inp.L_total > inp.x_garganta || error("L_total deve ser maior que x_garganta.")
end

function validar_case_input(inp::CaseInput)
    if !isempty(inp.grains)
        return _validar_case_stack(inp)
    end
    if inp.geometry_type ∉ (:bates, :star, :finocyl, :finocyl_conico, :finocyl_cono_cil, :end_burner, :moonburner, :wagon_wheel, :multiperf)
        error("geometry_type inválido: $(inp.geometry_type). " *
              "Use :bates, :star, :finocyl, :finocyl_conico, :finocyl_cono_cil, :end_burner, :moonburner, :wagon_wheel ou :multiperf.")
    end
    if inp.D_ext <= 0 || inp.L_grao <= 0
        error("D_ext e L_grao devem ser positivos.")
    end

    # ── Validação de propelente / termodinâmica (segurança) ───────────────────
    # Inputs de propelente/termo NÃO eram validados — um typo (n=3.21 em vez de
    # 0.321) ou erro de unidade em 'a' produzia resultado silenciosamente errado.
    # n fora de (0,1): o expoente 1/(1−n) da pressão de equilíbrio diverge/inverte.
    (0.0 < inp.n < 1.0) ||
        error("Expoente de queima n = $(inp.n) fora de (0, 1). " *
              "n ≥ 1 → realimentação de pressão INSTÁVEL (pode divergir → ruptura); " *
              "n ≤ 0 → fisicamente inválido. Verifique valor/unidade.")
    inp.n > 0.8 &&
        @warn "n = $(inp.n) > 0.8 — margem de estabilidade baixa " *
              "(SM = $(round((1.0-inp.n)*100, digits=1))%). Risco de instabilidade de combustão."
    inp.a     > 0.0 || error("Coeficiente de queima a = $(inp.a) deve ser > 0.")
    inp.rho_p > 0.0 || error("Densidade ρ_p = $(inp.rho_p) deve ser > 0.")
    inp.Tc    > 0.0 || error("Temperatura de chama Tc = $(inp.Tc) deve ser > 0.")
    inp.R     > 0.0 || error("Constante do gás R = $(inp.R) deve ser > 0.")
    (1.0 < inp.gamma < 1.4) ||
        @warn "γ = $(inp.gamma) fora da faixa física típica (1.0, 1.4) para gases de combustão."
    (0.0 < inp.eta_cstar <= 1.0) ||
        @warn "η_c* = $(inp.eta_cstar) fora de (0, 1] — eficiência > 1 é não-física."
    # Sanidade da taxa de queima — PEGA erro de UNIDADE de 'a' (que depende de n!).
    # a deve estar em SI (m/s/Paⁿ). r(7 MPa) fora de 0.1–100 mm/s indica unidade errada.
    let r_7MPa_mm = inp.a * (7.0e6 ^ inp.n) * 1.0e3
        (0.05 < r_7MPa_mm < 200.0) ||
            @warn "Taxa de queima r(7 MPa) = $(round(r_7MPa_mm, digits=3)) mm/s fora da faixa " *
                  "típica (0.1–100 mm/s). VERIFIQUE A UNIDADE de 'a' — deve ser m/s/Paⁿ (SI). " *
                  "Ex.: a_SI ≈ 7e-5 para HTPB/AP, não 0.07 nem 7."
    end

    # end_burner não usa D_core (queima apenas pela face)
    if inp.geometry_type in (:bates, :finocyl, :finocyl_conico, :finocyl_cono_cil, :moonburner, :wagon_wheel, :multiperf) && inp.D_core <= 0
        error("D_core deve ser positivo para $(inp.geometry_type).")
    end
    # B1 FIX: end_burner é sempre um único bloco — N_graos > 1 não tem sentido físico
    # e causaria faces intermediárias entre segmentos com inhibited_ends=3 mal definidas.
    if inp.geometry_type == :end_burner && inp.N_graos > 1
        error("Para :end_burner, N_graos deve ser 1. " *
              "Um grão end-burner é sempre um único bloco sólido.")
    end
    # B3 FIX: moonburner — alinhar tolerância com gerar_interpoladores_moonburner.
    # LibGEOS requer folga mínima de 1e-4 m (0.1 mm) entre porto e carcaça para
    # evitar erros numéricos na interseção de geometrias quase-tangentes.
    if inp.geometry_type == :moonburner
        R_ext  = inp.D_ext  / 2.0
        R_core = inp.D_core / 2.0
        e      = inp.eccentricity
        if e < 0.0
            error("Para :moonburner, eccentricity deve ser ≥ 0.")
        end
        folga_minima = 1e-4   # 0.1 mm — tolerância numérica do LibGEOS
        if e + R_core >= R_ext - folga_minima
            error("Para :moonburner: eccentricity ($(round(e*1e3,digits=2)) mm) + " *
                  "R_core ($(round(R_core*1e3,digits=2)) mm) deve ser < R_ext " *
                  "($(round(R_ext*1e3,digits=2)) mm) − 0.1 mm de folga mínima. " *
                  "Aumente R_ext ou reduza e/R_core.")
        end
    end
    if inp.geometry_type in (:finocyl, :finocyl_conico, :finocyl_cono_cil)
        if inp.n_fins <= 0 || inp.fin_width <= 0 || inp.fin_length <= 0
            error("Para $(inp.geometry_type), informe n_fins, fin_width e fin_length válidos.")
        end
        if inp.D_core <= 0
            error("Para $(inp.geometry_type), D_core deve ser positivo.")
        end
        # Cônico: checa o PIOR caso entre o furo da cabeça (D_core) e o do bocal (D_core_aft).
        dc_aft = (inp.geometry_type in (:finocyl_conico, :finocyl_cono_cil) && inp.D_core_aft > 1e-6) ?
                 inp.D_core_aft : inp.D_core
        dc_min = min(inp.D_core, dc_aft)          # menor furo → menor perímetro (sobreposição)
        dc_max = max(inp.D_core, dc_aft)          # maior furo → menor teia
        if (inp.D_ext - dc_max) / 2.0 <= 0
            error("Para $(inp.geometry_type), D_core (e D_core_aft) devem ser < D_ext.")
        end
        # No cônico a aleta é recortada por seção (fl_k = min(fin_length, web_k·0.9)),
        # então só é ERRO se ela não couber nem na seção de teia mais folgada.
        lim_fin = (inp.D_ext - dc_min) / 2.0
        if inp.fin_length >= lim_fin
            error("Para $(inp.geometry_type): fin_length ($(round(inp.fin_length*1e3, digits=2)) mm) " *
                  "deve ser < espessura da teia ($(round(lim_fin*1e3, digits=2)) mm).")
        end
        perim_core = π * dc_min
        slot_total = Float64(inp.n_fins) * inp.fin_width
        if slot_total >= perim_core
            error("Para $(inp.geometry_type): $(inp.n_fins) × fin_width = $(round(slot_total*1e3, digits=2)) mm " *
                  "deve ser < π·D_core = $(round(perim_core*1e3, digits=2)) mm " *
                  "(aletas se sobreporiam).")
        end
    end
    if inp.geometry_type == :star
        if inp.pontas <= 0 || inp.raio_base <= 0 || inp.raio_fenda <= 0
            error("Para :star, informe pontas, raio_base e raio_fenda válidos.")
        end
    end
    if inp.x_garganta <= inp.L_grao * inp.N_graos
        error("x_garganta deve ser maior que o comprimento total da câmara.")
    end
    if inp.L_total <= inp.x_garganta
        error("L_total deve ser maior que x_garganta.")
    end
    if !(0.0 < inp.eta_tubeira <= 1.0)
        error("eta_tubeira deve estar em (0, 1] — recebido $(inp.eta_tubeira). " *
              "Use 1.0 para desligar o derating de empuxo.")
    end
end

# Constrói a Spec de geometria (build_geometry) a partir de um GrainSpec do stack.
function _spec_from_grain(g::GrainSpec, D_ext::Float64)
    if g.tipo == :bates
        return BatesSpec(D_ext = D_ext, D_core = g.D_core)
    elseif g.tipo == :finocyl
        return FinocylSpec(
            D_ext = D_ext, D_core = g.D_core, n_fins = g.n_fins,
            fin_width = g.fin_width, fin_length = g.fin_length,
            fin_fraction = g.fin_fraction, slot_shape = g.slot_shape,
            fin_tip_radius = g.fin_tip_radius, fin_root_radius = g.fin_root_radius,
            geometry_resolution = g.geometry_resolution,
        )
    elseif g.tipo == :star
        return StarSpec(
            D_ext = D_ext, pontas = g.pontas, raio_base = g.raio_base,
            raio_fenda = g.raio_fenda, angulo_fenda = g.angulo_fenda,
            passos = g.passos_geometria,
        )
    elseif g.tipo == :moonburner
        return MoonBurnerSpec(
            D_ext = D_ext, D_core = g.D_core,
            eccentricity = g.eccentricity, passos = g.passos_geometria,
        )
    else
        error("_spec_from_grain: tipo de grão não suportado no stack: $(g.tipo)")
    end
end

# Monta a geometria de um stack heterogêneo: LUT por grão + multi_segment_layout.
# Retorna a MESMA tupla que build_case_geometry (spec=nothing; não há tipo único).
function _build_stack_geometry(inp::CaseInput)
    pares = Tuple{Float64, ModuloGeometria.GrainGeometryLUT}[]
    for g in inp.grains
        if g.tipo == :finocyl && g.fin_fraction < 1.0 - 1e-9
            # Finocyl parcialmente aletado: divide em (cilíndrico fore) + (aletado aft),
            # como o caminho de tipo único. As faces ficam nas pontas do GRÃO (não na
            # transição) porque grain_boundaries trata isto como 1 grão físico só.
            ff = clamp(g.fin_fraction, 1e-3, 1.0 - 1e-9)
            geom_cyl = Base.invokelatest(ModuloGeometria.build_geometry,
                                         CylinderSpec(D_ext = inp.D_ext, D_core = g.D_core))
            geom_fin = Base.invokelatest(ModuloGeometria.build_geometry, _spec_from_grain(g, inp.D_ext))
            Base.invokelatest(ModuloGeometria.validar_lut, geom_cyl; throw_on_error = false)
            Base.invokelatest(ModuloGeometria.validar_lut, geom_fin; throw_on_error = false)
            push!(pares, (g.L * (1.0 - ff), geom_cyl))   # parte cilíndrica
            push!(pares, (g.L * ff,         geom_fin))   # parte aletada
        else
            lut = Base.invokelatest(ModuloGeometria.build_geometry, _spec_from_grain(g, inp.D_ext))
            Base.invokelatest(ModuloGeometria.validar_lut, lut; throw_on_error = false)
            push!(pares, (g.L, lut))
        end
    end

    layout = ModuloGeometria.multi_segment_layout(pares)
    ModuloGeometria.validar_layout(layout)

    y_burn_max = maximum(
        seg.is_transition ? max(seg.geom_a.y_max, seg.geom_b.y_max) : seg.geom_a.y_max
        for seg in layout.segments
    )
    A0 = layout.segments[1].geom_a.interp_A(0.0)
    D_port_inicial = sqrt(4.0 * A0 / pi)

    return nothing, pares[1][2], layout, y_burn_max, D_port_inicial
end

# Nº de fatias axiais do furo cônico (:finocyl_conico). Compromisso: mais fatias =
# cone melhor resolvido e burnout mais escalonado, porém uma LUT a mais por fatia
# (custo de build) e segmentos mais curtos que a célula da malha 1D.
const _N_FATIAS_CONICO = 30   # 30 (era 12): tail do bore cônico mais LISA (menos "escadinha" de
                              # esgotamento das fatias). Custo ~2.5× por config cônica. Para curva
                              # perfeitamente contínua (sem degrau), o certo seria integrar A_b(w)
                              # analiticamente sobre o cone — reescrita do build_conic_finocyl_segments.

function build_case_geometry(inp::CaseInput)
    validar_case_input(inp)

    if !isempty(inp.grains)
        return _build_stack_geometry(inp)
    end

    # ── :finocyl_conico — furo CÔNICO (D_core na cabeça → D_core_aft no bocal) ────
    # O grão é FATIADO ao longo de x: cada fatia tem o furo local (e a teia local)
    # da sua posição. As fatias de furo MAIOR esgotam a teia antes → saem da conta
    # de área → contrabalançam o crescimento do perímetro das demais → platô mais
    # NEUTRO (ideia do booster cônico, DLR/EUCASS). O trecho dianteiro fica liso e
    # o aft aletado, na fração inp.fin_fraction — igual ao finocyl reto.
    if inp.geometry_type == :finocyl_conico
        dc_aft = inp.D_core_aft > 1e-6 ? inp.D_core_aft : inp.D_core
        base_spec = FinocylSpec(
            D_ext = inp.D_ext, D_core = inp.D_core,
            n_fins = inp.n_fins, fin_width = inp.fin_width,
            fin_length = inp.fin_length, fin_fraction = inp.fin_fraction,
            slot_shape          = inp.slot_shape,
            fin_tip_radius      = inp.fin_tip_radius,
            fin_root_radius     = inp.fin_root_radius,
            geometry_resolution = inp.geometry_resolution,
        )
        pares_c = Base.invokelatest(
            ModuloGeometria.build_conic_finocyl_segments,
            base_spec, inp.D_core, dc_aft, inp.L_grao;
            fin_fraction  = inp.fin_fraction,
            N_seg         = _N_FATIAS_CONICO,
            fl_ramp_zone  = inp.fin_taper_zone_length,   # rampa da aleta (0 = reta)
            fl_start_frac = inp.fin_taper_start_frac,
        )
        for (_, lut) in pares_c
            Base.invokelatest(ModuloGeometria.validar_lut, lut; throw_on_error = false)
        end
        # N_graos > 1: repete o mesmo grão em série (faces vêm de N_graos, não do layout)
        pares_all = inp.N_graos > 1 ? repeat(pares_c, inp.N_graos) : pares_c
        layout_c  = ModuloGeometria.multi_segment_layout(pares_all)
        ModuloGeometria.validar_layout(layout_c)

        y_max_c = maximum(seg.geom_a.y_max for seg in layout_c.segments)
        # Porto inicial EQUIVALENTE: média de A(0) ponderada pelo comprimento da fatia
        # (num cone o porto varia com x; o D_port_ini só alimenta o L* da estabilidade).
        L_tot_c = sum(L for (L, _) in pares_c)
        A0_c    = sum(L * lut.interp_A(0.0) for (L, lut) in pares_c) / L_tot_c
        return base_spec, pares_c[1][2], layout_c, y_max_c, sqrt(4.0 * A0_c / pi)
    end

    # ── :finocyl_cono_cil — aletado com furo CONSTANTE + trecho cilíndrico CÔNICO ──
    # Só o cilindro afila (D_core → D_core_aft); as aletas ficam em D_core constante.
    # `finocyl_no_bocal` escolhe se o aletado fica no bocal (aft) ou na cabeça (fore).
    if inp.geometry_type == :finocyl_cono_cil
        d_cil_ext = inp.D_core_aft > 1e-6 ? inp.D_core_aft : inp.D_core
        base_spec = FinocylSpec(
            D_ext = inp.D_ext, D_core = inp.D_core,
            n_fins = inp.n_fins, fin_width = inp.fin_width,
            fin_length = inp.fin_length, fin_fraction = inp.fin_fraction,
            slot_shape          = inp.slot_shape,
            fin_tip_radius      = inp.fin_tip_radius,
            fin_root_radius     = inp.fin_root_radius,
            geometry_resolution = inp.geometry_resolution,
        )
        pares_c = Base.invokelatest(
            ModuloGeometria.build_cono_cil_finocyl_segments,
            base_spec, inp.D_core, d_cil_ext, inp.L_grao;
            fin_fraction  = inp.fin_fraction,
            fin_no_bocal  = inp.finocyl_no_bocal,
            N_seg         = _N_FATIAS_CONICO,
            fl_ramp_zone  = inp.fin_taper_zone_length,   # rampa da aleta (0 = reta)
            fl_start_frac = inp.fin_taper_start_frac,
        )
        for (_, lut) in pares_c
            Base.invokelatest(ModuloGeometria.validar_lut, lut; throw_on_error = false)
        end
        pares_all = inp.N_graos > 1 ? repeat(pares_c, inp.N_graos) : pares_c
        layout_c  = ModuloGeometria.multi_segment_layout(pares_all)
        ModuloGeometria.validar_layout(layout_c)
        y_max_c = maximum(seg.geom_a.y_max for seg in layout_c.segments)
        L_tot_c = sum(L for (L, _) in pares_c)
        A0_c    = sum(L * lut.interp_A(0.0) for (L, lut) in pares_c) / L_tot_c
        return base_spec, pares_c[1][2], layout_c, y_max_c, sqrt(4.0 * A0_c / pi)
    end

    spec =
        inp.geometry_type == :bates ? BatesSpec(
            D_ext = inp.D_ext, D_core = inp.D_core,
            inhibited_ends = inp.inhibited_ends
        ) :
        inp.geometry_type == :star ? StarSpec(
            D_ext = inp.D_ext, pontas = inp.pontas,
            raio_base = inp.raio_base, raio_fenda = inp.raio_fenda,
            angulo_fenda = inp.angulo_fenda, passos = inp.passos_geometria
        ) :
        inp.geometry_type == :finocyl ? FinocylSpec(
            D_ext = inp.D_ext, D_core = inp.D_core,
            n_fins = inp.n_fins, fin_width = inp.fin_width,
            fin_length = inp.fin_length, fin_fraction = inp.fin_fraction,
            slot_shape          = inp.slot_shape,
            fin_tip_radius      = inp.fin_tip_radius,
            fin_root_radius     = inp.fin_root_radius,
            geometry_resolution = inp.geometry_resolution,
        ) :
        inp.geometry_type == :end_burner ? EndBurnerSpec(
            D_ext = inp.D_ext
        ) :
        inp.geometry_type == :moonburner ? MoonBurnerSpec(
            D_ext        = inp.D_ext,
            D_core       = inp.D_core,
            eccentricity = inp.eccentricity,
        ) :
        inp.geometry_type == :wagon_wheel ? WagonWheelSpec(
            D_ext       = inp.D_ext, D_core = inp.D_core,
            n_spokes    = Int(round(inp.n_fins)),
            slot_width  = inp.fin_width, slot_length = inp.fin_length,
            slot_shape  = inp.slot_shape,
            geometry_resolution = inp.geometry_resolution,
        ) :
        inp.geometry_type == :multiperf ? MultiPerfSpec(
            D_ext  = inp.D_ext, n_perf = Int(round(inp.n_fins)),
            D_perf = inp.D_core, pitch_frac = 0.5,
        ) :
        error("geometry_type não suportado: $(inp.geometry_type)")

    geom_lut = Base.invokelatest(ModuloGeometria.build_geometry, spec)

    # Validação geométrica da LUT (estava dormente): checa NaN/Inf, A_port
    # monotônica, P_flux≈P_burn+P_wall, saltos bruscos. Modo aviso (não bloqueia
    # geometrias válidas que apenas tropecem numa heurística); para release
    # comercial pode-se passar throw_on_error=true.
    Base.invokelatest(ModuloGeometria.validar_lut, geom_lut; throw_on_error=false)

    layout = if inp.geometry_type == :finocyl && inp.fin_fraction < 1.0 - 1e-9
        fin_frac = clamp(inp.fin_fraction, 1e-3, 1.0 - 1e-9)
        L_cyl    = inp.L_grao * (1.0 - fin_frac)
        L_fin    = inp.L_grao * fin_frac
        L_trans  = inp.L_grao * clamp(inp.L_trans_frac, 0.0, 0.3)
        cyl_spec = CylinderSpec(D_ext = inp.D_ext, D_core = inp.D_core)
        geom_cyl = Base.invokelatest(ModuloGeometria.build_geometry, cyl_spec)

        if inp.N_graos == 1 && inp.fin_taper_zone_length > 1e-4
            # ── MODO B: Geometria analítica 3D — zona de chanfro contínua ────
            # A zona de chanfro é modelada como uma LUT efetiva que integra
            # P_burn sobre toda a distribuição de fin_length [fl_start, fl_max].
            # Resultado: transição Phase 1→2 suave (C¹), sem degraus discretos.
            L_taper = clamp(inp.fin_taper_zone_length, 0.0, (L_fin - 1e-3) * 0.8)
            n_fl    = max(5, inp.fin_taper_n_segs)   # n_segs → nº de LUTs para integração

            @printf(
                "  [TaperZone3D] zona chanfro = %.0f mm  (%d LUTs, fl: %.0f%%→%.0f%% de %.0f mm)\n",
                L_taper * 1e3, n_fl,
                inp.fin_taper_start_frac * 100.0, 100.0,
                inp.fin_length * 1e3
            )

            geom_taper = ModuloGeometria.build_tapered_zone_lut(
                spec,
                inp.fin_taper_start_frac * inp.fin_length;   # fl_start em metros
                N_fl = n_fl,
            )

            ModuloGeometria.taper3d_layout(
                geom_cyl, geom_taper, geom_lut,
                L_cyl, L_taper, inp.L_grao
            )

        elseif inp.N_graos == 1 && inp.fin_taper_n_segs > 1
            # ── MODO A: sub-segmentos discretos (comportamento original) ──────
            # Divide a seção aft inteira em N segmentos com fin_length crescente.
            # Útil para análise de sensibilidade; pode criar degraus visíveis na curva.
            n_tap   = max(2, inp.fin_taper_n_segs)
            f_start = clamp(inp.fin_taper_start_frac, 0.0, 1.0 - 1/n_tap)

            @printf(
                "  [FinTaper-A] %d sub-segmentos, fl = %.0f → %.0f mm\n",
                n_tap,
                f_start * inp.fin_length * 1e3,
                inp.fin_length * 1e3
            )

            T_fin = typeof(geom_lut)
            luts_fin = Vector{T_fin}(undef, n_tap)
            for k in 1:n_tap
                α       = f_start + (1.0 - f_start) * ((k - 0.5) / n_tap)
                fl_k    = max(1e-4, α * inp.fin_length)
                spec_k  = FinocylSpec(
                    D_ext               = inp.D_ext,
                    D_core              = inp.D_core,
                    n_fins              = inp.n_fins,
                    fin_width           = inp.fin_width,
                    fin_length          = fl_k,
                    slot_shape          = inp.slot_shape,
                    fin_tip_radius      = min(inp.fin_tip_radius, fl_k * 0.49),
                    fin_root_radius     = inp.fin_root_radius,
                    geometry_resolution = inp.geometry_resolution,
                )
                luts_fin[k] = Base.invokelatest(ModuloGeometria.build_geometry, spec_k)
                @printf("    seg %d/%d: fl = %.1f mm  y_max = %.1f mm\n",
                        k, n_tap, fl_k * 1e3, luts_fin[k].y_max * 1e3)
            end

            ModuloGeometria.tapered_finocyl_layout(
                geom_cyl, luts_fin, L_cyl, inp.L_grao; L_trans = L_trans
            )

        elseif inp.N_graos == 1
            ModuloGeometria.two_segment_layout_with_transition(
                geom_cyl, L_cyl, geom_lut, inp.L_grao; L_trans = L_trans
            )
        else
            # Multi-grain: repeat_alternating_layout (type-stable, zero boxing).
            ModuloGeometria.repeat_alternating_layout(
                geom_cyl, L_cyl, geom_lut, L_fin, inp.N_graos
            )
        end
    elseif inp.N_graos == 1
        ModuloGeometria.single_segment_layout(geom_lut, inp.L_grao)
    else
        ModuloGeometria.repeat_layout(geom_lut, inp.N_graos, inp.L_grao)
    end

    ModuloGeometria.validar_layout(layout)

    # prop.y_max = teto global de y_queima: deve ser o MÁXIMO entre todos os segmentos.
    # O controle per-célula (cada célula para no y_max do seu próprio segmento) é feito
    # por obter_ymax_local() em physics.jl, que retorna o y_max correto para cada x.
    # Usar minimum aqui cortaria prematuramente células com aletas mais longas (taper).
    y_burn_max = maximum(
        seg.is_transition ?
            max(seg.geom_a.y_max, seg.geom_b.y_max) :
            seg.geom_a.y_max
        for seg in layout.segments
    )

    A0 = layout.segments[1].geom_a.interp_A(0.0)
    D_port_inicial = sqrt(4.0 * A0 / pi)

    return spec, geom_lut, layout, y_burn_max, D_port_inicial
end

function build_propellant_from_case(
    inp           ::CaseInput,
    layout        ::GrainLayout,
    y_burn_max    ::Float64,
    D_port_inicial::Float64
)
    # End-burner: forçar inhibited_ends=3 (só face traseira queima) independente
    # do que o utilizador definiu na GUI.
    inh_efetivo = (inp.geometry_type == :end_burner && isempty(inp.grains)) ?
        3 : inp.inhibited_ends

    # grain_type: sinaliza lógica especial de queima em physics.jl
    gtype = (inp.geometry_type == :end_burner && isempty(inp.grains)) ?
        :end_burner : :standard

    # Fronteiras físicas dos grãos do stack (vazio = tipo único → fallback uniforme).
    grain_bounds = Float64[]
    grain_inh    = Int[]
    if !isempty(inp.grains)
        push!(grain_bounds, 0.0)
        xacc = 0.0
        for g in inp.grains
            xacc += g.L
            push!(grain_bounds, xacc)
            push!(grain_inh, g.inhibited_ends)   # faces inibidas DESTE grão
        end
    end

    return Propelente{typeof(layout)}(
        rho_p = inp.rho_p, a = inp.a, n = inp.n,
        Tc = inp.Tc, gamma = inp.gamma, R = inp.R,
        eta_cstar = inp.eta_cstar, eta_tubeira = inp.eta_tubeira,
        k_p = inp.k_p, cp_p = inp.cp_p,
        frac_alumina = inp.frac_alumina, T_ignicao = inp.T_ignicao,
        D_ext = inp.D_ext, D_port_ini = D_port_inicial,
        L_grao = inp.L_grao, N_graos = inp.N_graos,
        y_max = y_burn_max, layout = layout,
        grain_boundaries = grain_bounds,
        grain_inhibited  = grain_inh,
        inhibited_ends = inh_efetivo,
        grain_type     = gtype,
        alpha_e = inp.alpha_e, beta_e = inp.beta_e,
        sigma_p = inp.sigma_p, T_ref = inp.T_ref, T_grain = inp.T_grain
    )
end

function build_domain_geometry(inp::CaseInput)
    return Dict(
        "x_garganta"       => inp.x_garganta,
        "D_garganta_ini"   => inp.D_garganta_ini,
        "D_saida"          => inp.D_saida,
        "L_total"          => inp.L_total,
        "alpha_divergencia" => inp.alpha_divergencia,
        # Parâmetros de erosão da garganta
        "erosao_r_dot_ref" => inp.erosao_r_dot_ref * 1e-3,  # mm/s → m/s
        "erosao_P_ref"     => inp.erosao_P_ref_MPa  * 1e6,  # MPa → Pa
        "erosao_n_exp"     => inp.erosao_n_exp,
    )
end

# ==============================================================================
# Retenção de escória (slag) de Al₂O₃
# ==============================================================================
"""
    _eta_slag(frac_slag, frac_alumina, m_prop_kg) -> (η_slag, slag_kg)

Penalidade de impulso pela Al₂O₃ RETIDA no motor (não sai pela tubeira). A massa
de Al₂O₃ produzida ≈ `frac_alumina·(102/54)·m_prop` (2Al+1.5O₂→Al₂O₃). Uma fração
`frac_slag` dela vira escória retida (domo aft/parede) → não gera empuxo:

    slag_kg = frac_slag·frac_alumina·(102/54)·m_prop
    η_slag  = 1 − slag_kg/m_prop = 1 − frac_slag·frac_alumina·(102/54)

`frac_slag = 0` (default) → `(1.0, 0.0)`, sem efeito. Empírico e dependente de
geometria (submersa/spin ↑ slag) — ver ConfigModelo.frac_slag.
"""
function _eta_slag(frac_slag::Real, frac_alumina::Real, m_prop_kg::Real)
    fs = clamp(Float64(frac_slag), 0.0, 1.0)
    (fs <= 0.0 || frac_alumina <= 0.0) && return (1.0, 0.0)
    frac_al2o3 = Float64(frac_alumina) * (102.0 / 54.0)   # Al → Al₂O₃ em massa
    slag_frac  = clamp(fs * frac_al2o3, 0.0, 0.99)        # fração da massa total retida
    slag_kg    = slag_frac * Float64(m_prop_kg)
    return (1.0 - slag_frac, slag_kg)
end

"""
    _eta_2ph_final(η_escalar, cfg, inp, d43_um, P_avg_MPa, silencioso) -> η_2ph

Se `cfg.usar_2fases_acoplado`, recalcula η_2ph integrando o escoamento bifásico
ACOPLADO (TwoPhaseNozzle.eta_2ph_acoplado) no divergente real; senão devolve o
η escalar. Faz fallback ao escalar se o acoplado falhar ou sair fora de (0,1].
"""
function _eta_2ph_final(η_escalar::Real, cfg, inp, d43_um::Real,
                        P_avg_MPa::Real, silencioso::Bool)
    (cfg.usar_2fases_acoplado && inp.frac_alumina > 0.0 && d43_um > 0.0) ||
        return Float64(η_escalar)
    P0 = max(Float64(P_avg_MPa), 0.1) * 1e6
    try
        if cfg.usar_distribuicao_particula
            η_ac, info = eta_2ph_distribuicao(inp, d43_um, P0;
                                              sigma_g = cfg.sigma_g_particula)
            (isfinite(η_ac) && 0.0 < η_ac <= 1.0) || return Float64(η_escalar)
            silencioso || @printf(
                "  [2ph DISTRIB σ_g=%.2f] η=%.4f  (mono d43 era %.4f, escalar %.4f, D43=%.1f µm)\n",
                cfg.sigma_g_particula, η_ac, info.eta_mono, η_escalar, d43_um)
            return η_ac
        end
        η_ac, _ = eta_2ph_acoplado(inp, d43_um, P0)
        (isfinite(η_ac) && 0.0 < η_ac <= 1.0) || return Float64(η_escalar)
        silencioso || @printf(
            "  [2ph ACOPLADO] η=%.4f  (escalar era %.4f, D43=%.1f µm — TwoPhaseNozzle)\n",
            η_ac, η_escalar, d43_um)
        return η_ac
    catch e
        silencioso || @printf("  [2ph ACOPLADO] falhou → mantém escalar (%s)\n",
                              first(sprint(showerror, e), 60))
        return Float64(η_escalar)
    end
end

# ==============================================================================
# Porteira de MEOP — verificação de sobrepressão (segurança)
# ==============================================================================
"""
    _verificar_meop(P_meop_MPa, P_max_MPa, silencioso) -> (P_meop, FS_meop)

Compara a pressão máxima simulada contra a MEOP especificada pelo usuário.
- `P_meop_MPa = 0` → verificação desativada, retorna `(0.0, Inf)`.
- Senão retorna `(P_meop_MPa, MEOP/P_max)` e emite aviso de SOBREPRESSÃO se FS < 1.
O aviso respeita `modo_silencioso` (Monte Carlo); o campo `FS_meop` fica sempre
disponível no `SimulationResult` para inspeção programática.
"""
function _verificar_meop(P_meop_MPa::Float64, P_max_MPa::Float64, silencioso::Bool)
    P_meop_MPa > 0.0 || return (0.0, Inf)
    FS = P_meop_MPa / max(P_max_MPa, 1e-9)
    if FS < 1.0 && !silencioso
        @warn @sprintf("⛔ SOBREPRESSÃO: P_max = %.2f MPa EXCEDE a MEOP = %.2f MPa (FS = %.2f < 1.0). Risco de RUPTURA da carcaça — revise geometria/garganta/propelente.",
                       P_max_MPa, P_meop_MPa, FS)
    end
    return (P_meop_MPa, FS)
end

# ==============================================================================
# Auto-ignitor — escalonamento automático ao tamanho do motor
# ==============================================================================
"""
    _auto_ignitor(inp, layout, geom_lut) → Ignitor

Cria um `Ignitor` com `m_dot_ign` escalonado às condições de equilíbrio
inicial do motor, evitando sobrepressão durante o arranque.

**Estratégia (balística 0-D):**
1. Calcula a área de queima inicial `A_burn₀` a partir do layout (y = 0).
2. Kn₀ = A_burn₀ / A_throat.
3. c* = √(R · Tc · η²) / Γ(γ)   (Vandenkerckhove).
4. P_eq₀ = (ρ_p · a · c* · Kn₀)^(1/(1-n)).
5. ṁ_eq = ρ_p · a · P_eq₀ⁿ · A_burn₀   (fluxo de equilíbrio do grão).
6. ṁ_ign = ṁ_eq × 0.10  → ~10 % do fluxo de equilíbrio.

Usar 10 % mantém a sobrepressão transitória abaixo de ~15 % mesmo
com o grão já a queimar durante a fase de ignição (análise 0-D, n ≈ 0.35).
Um mínimo de 1 × 10⁻³ kg/s garante ignição em motores muito pequenos.
"""
function _auto_ignitor(
    inp      ::CaseInput,
    layout,          # GrainLayout
    geom_lut         # GeomLUT / resultado de build_case_geometry
)
    # ── c* teórico (fonte única: calcular_cstar_teorico) ────────────────────
    cstar_0D = calcular_cstar_teorico(inp.R, inp.Tc, inp.gamma, inp.eta_cstar)

    A_throat = π / 4.0 * inp.D_garganta_ini^2
    A_ext    = π / 4.0 * inp.D_ext^2

    # ── Área de queima inicial (y = 0) ──────────────────────────────────────
    A_burn_ini = if inp.geometry_type == :end_burner
        # End-burner: só a face traseira queima (área transversal total)
        A_ext
    else
        # Geometrias com porto radial: lateral + faces axiais
        A_lat = 0.0
        for seg in layout.segments
            L_seg = seg.x_end - seg.x_start
            if seg.is_transition
                P_a  = seg.geom_a.interp_Pburn(0.0)
                P_b  = seg.geom_b.interp_Pburn(0.0)
                A_lat += 0.5 * (P_a + P_b) * L_seg
            else
                A_lat += seg.geom_a.interp_Pburn(0.0) * L_seg
            end
        end
        # Faces axiais livres
        inh = inp.geometry_type == :end_burner ? 3 : inp.inhibited_ends
        n_g = isempty(inp.grains) ? inp.N_graos : length(inp.grains)
        n_faces = (inh == 0 ? 2 : inh == 2 ? 0 : 1) * n_g
        A_port  = geom_lut.interp_A(0.0)
        A_lat + n_faces * max(A_ext - A_port, 0.0)
    end

    # Proteção: área mínima para evitar divisão por zero em configurações
    # ainda não completamente construídas (ex.: geometria degradada).
    A_burn_ini = max(A_burn_ini, 1e-6)

    # ── Kn e P_eq iniciais (0-D) ────────────────────────────────────────────
    Kn_ini  = A_burn_ini / A_throat
    exp_p   = 1.0 / (1.0 - inp.n)
    P_eq_0  = (inp.rho_p * inp.a * cstar_0D * Kn_ini)^exp_p
    P_eq_0  = max(P_eq_0, 101325.0)          # mínimo 1 atm

    # ── Fluxo de massa de equilíbrio do grão ────────────────────────────────
    r_dot_eq  = inp.a * P_eq_0^inp.n
    m_dot_eq  = inp.rho_p * r_dot_eq * A_burn_ini

    # ── m_dot_ign = 10 % do equilíbrio, mínimo 1 g/s ───────────────────────
    # NOTA (Fase 0.4): m_dot_ign é o PICO do perfil senoidal (ver struct Ignitor),
    # não a média — o fluxo médio efetivo do ignitor é 2/π·m_dot_ign ≈ 6,4% do eq.
    m_dot_ign = max(0.10 * m_dot_eq, 1e-3)

    return Ignitor(posicao_final = 0.15, m_dot_ign = m_dot_ign)
end

function validar_geometria_caso(prop::Propelente{L}, geom::Dict) where {L}
    L_camara   = comprimento_camara(prop)
    x_garganta = geom["x_garganta"]
    L_total    = geom["L_total"]
    if x_garganta <= L_camara
        error("Caso inválido: x_garganta=$(x_garganta) deve ser > L_camara=$(L_camara).")
    end
    if L_total <= x_garganta
        error("Caso inválido: L_total=$(L_total) deve ser > x_garganta=$(x_garganta).")
    end
end

# ==============================================================================
# Diagnóstico de geometria — Etapa 6
# ==============================================================================

"""
    diagnostico_geometria(inp; N_amostra=100, plotar=true, mostrar_tabela=true)

Diagnóstico rápido de geometria sem executar a simulação completa.

Computa, para cada fração de regressão y/y_max:
  • Kn(y)    = A_burn(y) / A_throat        (Klemmung)
  • P_eq(y)  = (ρ_p · a · c* · Kn)^(1/(1−n))  (equilíbrio balístico 0D)

Imprime uma tabela resumo e gera gráfico 2-painéis Kn / P_eq.

Parâmetros
----------
- `N_amostra`      : pontos amostrados ao longo da regressão (padrão: 100)
- `plotar`         : gera gráfico 2-painéis (requer Plots.jl carregado)
- `mostrar_tabela` : imprime tabela e alertas no terminal

Retorna
-------
`(ys, Kns, Peqs, cstar)` — vetores de amostragem + c* teórico [m/s]
"""
function diagnostico_geometria(inp::CaseInput;
                                N_amostra    ::Int  = 100,
                                plotar       ::Bool = true,
                                mostrar_tabela::Bool = true)

    # ── 1. Construir geometria ──────────────────────────────────────────────
    spec, geom_lut, layout, y_burn_max, D_port_ini = build_case_geometry(inp)

    # ── 2. Parâmetros da tubeira ────────────────────────────────────────────
    A_throat = π / 4.0 * inp.D_garganta_ini^2
    A_saida  = π / 4.0 * inp.D_saida^2
    ε        = A_saida / A_throat

    # ── 3. c* teórico (Vandenkerckhove; fonte única) ───────────────────────
    cstar   = calcular_cstar_teorico(inp.R, inp.Tc, inp.gamma, inp.eta_cstar)

    # ── 4. Amostrar Kn(y) e P_eq(y) ao longo da regressão ──────────────────
    ys     = LinRange(0.0, y_burn_max, N_amostra)
    Kns    = Vector{Float64}(undef, N_amostra)
    Peqs   = Vector{Float64}(undef, N_amostra)

    A_ext          = π / 4.0 * inp.D_ext^2
    exp_p          = 1.0 / (1.0 - inp.n)

    # Fronteiras dos grãos (cabeça→tubeira) e faces inibidas por grão — para a
    # recessão axial ser aplicada por SEGMENTO (ver bug do finocyl abaixo).
    if isempty(inp.grains)
        gbounds = collect(range(0.0, inp.L_grao * inp.N_graos, length = inp.N_graos + 1))
        ginh    = fill(inp.inhibited_ends, inp.N_graos)
    else
        gbounds = zeros(Float64, length(inp.grains) + 1)
        @inbounds for k in eachindex(inp.grains); gbounds[k+1] = gbounds[k] + inp.grains[k].L; end
        ginh    = [g.inhibited_ends for g in inp.grains]
    end

    # Área de porto LOCAL em x — coroa da face de cada grão. Num STACK, a face de
    # cada grão está na SUA geometria (ex.: a face traseira do finocyl usa o porto
    # do FINOCYL, não o geom_lut global — que no stack aponta para o 1º grão/BATES).
    # Sem isto a coroa usava o porto errado (BATES em vez de finocyl) → área de face
    # grande demais e o Kn no burnout estourava (~+4%). Para geometria uniforme
    # (BATES) devolve o mesmo geom_lut → V&V bit-idêntico.
    Aport_local(x, yq) = begin
        for seg in layout.segments
            if seg.x_start - 1e-9 <= x <= seg.x_end + 1e-9
                if seg.is_transition
                    f  = clamp((x - seg.x_start) / max(seg.x_end - seg.x_start, 1e-12), 0.0, 1.0)
                    Aa = seg.geom_a.interp_A(clamp(yq, 0.0, seg.geom_a.y_max))
                    Ab = seg.geom_b.interp_A(clamp(yq, 0.0, seg.geom_b.y_max))
                    return Aa + f * (Ab - Aa)
                else
                    return seg.geom_a.interp_A(clamp(yq, 0.0, seg.geom_a.y_max))
                end
            end
        end
        return geom_lut.interp_A(clamp(yq, 0.0, geom_lut.y_max))
    end

    for (i, y) in enumerate(ys)
        # ── Área lateral com recessão axial das faces INTEGRADA POR SEGMENTO ──
        # Cada face livre recua `y` (mesma taxa radial). A área lateral de cada
        # segmento é integrada só sobre sua porção ATIVA — o que a face já
        # consumiu (do topo/base de cada grão) é excluído. Para um BATES uniforme
        # isto é idêntico ao antigo fator global f_len=(L−n·y)/L (V&V preservado).
        #
        # CORREÇÃO (bug finocyl, subpressurização): o f_len GLOBAL encurtava o grão
        # INTEIRO uniformemente. Num finocyl com aletas no aft (inh=3), a face aft
        # recua para dentro do trecho aletado — que JÁ queimou por completo cedo
        # (P_burn≈0). A face recuava numa região vazia mas o f_len ainda encurtava o
        # trecho PLANO → área lateral e P_max subestimadas ~7% (não-conservador).
        # Integrando por segmento, a face "come" só o aletado (que contribui ~0) e
        # o trecho plano fica com seu comprimento cheio.
        A_lateral = 0.0
        A_faces   = 0.0
        @inbounds for gi in eachindex(ginh)
            inh_g = ginh[gi]
            front_free = (inh_g == 0 || inh_g == 1)             # face dianteira livre
            aft_free   = (inh_g == 0 || inh_g == 3)             # face traseira livre
            front_rec  = front_free ? y : 0.0
            aft_rec    = aft_free   ? y : 0.0
            a_lo = gbounds[gi]   + front_rec
            a_hi = gbounds[gi+1] - aft_rec
            a_hi <= a_lo && continue
            # Área lateral: só a porção ATIVA de cada segmento dentro do grão.
            for seg in layout.segments
                x_a = max(seg.x_start, a_lo)
                x_b = min(seg.x_end,   a_hi)
                L_act = x_b - x_a
                L_act <= 0.0 && continue
                if seg.is_transition
                    P_a = seg.geom_a.interp_Pburn(clamp(y, 0.0, seg.geom_a.y_max))
                    P_b = seg.geom_b.interp_Pburn(clamp(y, 0.0, seg.geom_b.y_max))
                    A_lateral += 0.5 * (P_a + P_b) * L_act
                else
                    P = seg.geom_a.interp_Pburn(clamp(y, 0.0, seg.geom_a.y_max))
                    A_lateral += P * L_act
                end
            end
            # Faces axiais: coroa (A_ext − A_port LOCAL) na posição de cada face
            # livre — usa o porto da geometria naquele x (corrige a coroa no stack).
            front_free && (A_faces += max(A_ext - Aport_local(a_lo, y), 0.0))
            aft_free   && (A_faces += max(A_ext - Aport_local(a_hi, y), 0.0))
        end
        A_burn  = A_lateral + A_faces
        Kns[i]  = A_burn / A_throat
        Peqs[i] = (inp.rho_p * inp.a * cstar * Kns[i])^exp_p
    end

    # ── 5. Estatísticas resumidas ───────────────────────────────────────────
    Kn_max   = maximum(Kns)
    Peq_max  = maximum(Peqs)
    Peq_min  = minimum(Peqs)
    Peq_med  = sum(Peqs) / N_amostra
    variacao = (Peq_max - Peq_min) / max(Peq_med, 1e-12) * 100.0

    # ── 6. Tabela resumo ────────────────────────────────────────────────────
    if mostrar_tabela
        println()
        println("=" ^ 62)
        println("  DIAGNÓSTICO DE GEOMETRIA — ", inp.name)
        println("=" ^ 62)
        println("  Geometria do grão")
        println("  ", "─" ^ 58)
        @printf("  %-30s : %s\n",     "Tipo",                    String(inp.geometry_type))
        @printf("  %-30s : %.1f mm\n","D_ext",                   inp.D_ext   * 1e3)
        @printf("  %-30s : %.1f mm\n","D_core / porto ini equiv.",D_port_ini  * 1e3)
        @printf("  %-30s : %.1f mm\n","L_grão",                  inp.L_grao  * 1e3)
        @printf("  %-30s : %d\n",     "N_grãos",                 inp.N_graos)
        @printf("  %-30s : %.2f mm\n","Espessura web (y_max)",   y_burn_max  * 1e3)
        @printf("  %-30s : %d / 2\n", "Faces inibidas / grão",   inp.inhibited_ends)
        println()
        println("  Tubeira")
        println("  ", "─" ^ 58)
        @printf("  %-30s : %.2f mm\n","D_garganta",              inp.D_garganta_ini * 1e3)
        @printf("  %-30s : %.2f mm\n","D_saída",                 inp.D_saida        * 1e3)
        @printf("  %-30s : %.4f cm²\n","A_throat",               A_throat * 1e4)
        @printf("  %-30s : %.3f\n",   "ε (razão de expansão)",   ε)
        println()
        println("  Propelente / balística 0D")
        println("  ", "─" ^ 58)
        @printf("  %-30s : %.0f kg/m³\n",     "ρ_p",         inp.rho_p)
        @printf("  %-30s : %.4e m/s/Paⁿ\n",   "a",           inp.a)
        @printf("  %-30s : %.4f\n",            "n",           inp.n)
        @printf("  %-30s : %.1f m/s  (η=%.2f)\n","c* teórico",cstar, inp.eta_cstar)
        println()
        println("  Curva Kn / P_eq ao longo da regressão")
        println("  ", "─" ^ 58)
        @printf("  %-14s  %-8s  %-10s  %-10s\n",
                "Frac. y/y_max", "y [mm]", "Kn [-]", "P_eq [MPa]")
        println("  ", "─" ^ 58)
        for frac in (0.0, 0.25, 0.50, 0.75, 1.0)
            idx = clamp(round(Int, frac * (N_amostra - 1)) + 1, 1, N_amostra)
            @printf("  %-14.2f  %-8.2f  %-10.1f  %-10.3f\n",
                    frac, ys[idx]*1e3, Kns[idx], Peqs[idx]/1e6)
        end
        println("  ", "─" ^ 58)
        @printf("  %-14s  %-8s  %-10.1f  %-10.3f\n",
                "Máximo", "—", Kn_max, Peq_max/1e6)
        println("=" ^ 62)
        println()
    end

    # ── 7. Alertas ──────────────────────────────────────────────────────────
    alertas = String[]

    if Kn_max > 600.0
        push!(alertas, "AVISO   Kn_max = $(round(Kn_max, digits=1)) > 600 — risco de instabilidade de pressão.")
    elseif Kn_max > 400.0
        push!(alertas, "INFO    Kn_max = $(round(Kn_max, digits=1)) > 400 — zona de atenção (limite típico: 600).")
    end

    if Peq_max > 20e6
        push!(alertas, "AVISO   P_eq_max = $(round(Peq_max/1e6, digits=2)) MPa — excede limite típico de carcaça (20 MPa).")
    elseif Peq_max > 10e6
        push!(alertas, "INFO    P_eq_max = $(round(Peq_max/1e6, digits=2)) MPa — pressão elevada (> 10 MPa).")
    end

    if variacao > 30.0
        idx_mid = clamp(round(Int, 0.5 * (N_amostra - 1)) + 1, 1, N_amostra)
        tipo_queima = Kns[idx_mid] > Kns[1] ? "progressivo" : "regressivo"
        push!(alertas, "INFO    Variação de P_eq = $(round(variacao, digits=1))% — perfil $(tipo_queima).")
    end

    if inp.D_core > 0.0 && inp.D_core / inp.D_ext > 0.70
        push!(alertas, "AVISO   D_core/D_ext = $(round(inp.D_core/inp.D_ext, digits=2)) > 0.70 — teia radial muito fina.")
    end

    if mostrar_tabela && !isempty(alertas)
        println("  Diagnósticos:")
        for alerta in alertas
            println("  >> ", alerta)
        end
        println()
    end

    # ── 8. Gráfico 2-painéis ────────────────────────────────────────────────
    if plotar
        xs_norm = collect(ys) ./ y_burn_max

        p1 = Plots.plot(xs_norm, Kns;
            xlabel = "Regressão y/y_max",
            ylabel = "Kn = A_b/A_t",
            title  = "Kn — $(inp.name)",
            lw = 2, color = :royalblue, label = "Kn",
            legend = :topright, grid = true)
        if Kn_max > 200.0
            Plots.hline!(p1, [400.0]; ls = :dash, color = :orange, lw = 1, label = "400")
        end
        if Kn_max > 400.0
            Plots.hline!(p1, [600.0]; ls = :dash, color = :red,    lw = 1, label = "600")
        end

        p2 = Plots.plot(xs_norm, Peqs ./ 1e6;
            xlabel = "Regressão y/y_max",
            ylabel = "P_eq [MPa]",
            title  = "Pressão de equilíbrio 0D",
            lw = 2, color = :firebrick, label = "P_eq",
            legend = :topright, grid = true)
        if Peq_max > 5e6
            Plots.hline!(p2, [10.0]; ls = :dash, color = :orange, lw = 1, label = "10 MPa")
        end
        if Peq_max > 15e6
            Plots.hline!(p2, [20.0]; ls = :dash, color = :red,    lw = 1, label = "20 MPa")
        end

        plt = Plots.plot(p1, p2;
            layout     = (1, 2),
            size       = (900, 420),
            plot_title = "Diagnóstico — $(inp.name)",
            margin     = 5Plots.mm)
        display(plt)
    end

    return collect(ys), Kns, Peqs, cstar
end

"""
    _cfg_com_erosao(inp, cfg) -> ConfigModelo

Traduz `inp.erosao_ativa` (a intenção do utilizador, vinda do formulário) para
`cfg.usar_erosao_garganta` (a chave que os solvers testam), preservando todos os
outros campos por keyword-reconstruction.

Existe porque as duas informações vivem em objetos diferentes: a GUI só liga o
flag no `CaseInput`, enquanto `Solver0D`, `SimulationCore` e `calcular_taxa_erosao`
consultam o `ConfigModelo`. Sem esta ponte, marcar "Erosão da garganta" no
formulário não tem efeito nenhum.

**Aplicar em TODOS os caminhos de simulação** — 1D e 0D. Foi exatamente o
esquecimento do caminho 0D que fez a opção ser ignorada em silêncio: o dispatch
do 0D acontece antes do ponto onde o 1D fazia a tradução.

Se `cfg.usar_erosao_garganta` já estiver ligado, devolve o `cfg` inalterado
(chamada explícita por script tem precedência e não precisa do flag em `inp`).
"""
function _cfg_com_erosao(inp::CaseInput, cfg::ConfigModelo) ::ConfigModelo
    (inp.erosao_ativa && !cfg.usar_erosao_garganta) || return cfg
    return ConfigModelo(
        usar_queima_somente_camara     = cfg.usar_queima_somente_camara,
        usar_faces_axiais              = cfg.usar_faces_axiais,
        f_faces_axiais                 = cfg.f_faces_axiais,
        f_regressao_axial              = cfg.f_regressao_axial,
        ganho_termico_ignitor          = cfg.ganho_termico_ignitor,
        coef_convectivo_base           = cfg.coef_convectivo_base,
        t_ramp_ignicao                 = cfg.t_ramp_ignicao,
        dt_max_startup                 = cfg.dt_max_startup,
        usar_erosiva                   = cfg.usar_erosiva,
        usar_erosao_garganta           = true,   # ← ativado
        cfl                            = cfg.cfl,
        usar_limitador_minmod          = cfg.usar_limitador_minmod,
        n_geom_skip                    = cfg.n_geom_skip,
        n_hist_skip                    = cfg.n_hist_skip,
        P_tailoff                      = cfg.P_tailoff,
        t_min_tailoff                  = cfg.t_min_tailoff,
        limiar_burnout_area_rel        = cfg.limiar_burnout_area_rel,
        limiar_burnout_mdot_rel        = cfg.limiar_burnout_mdot_rel,
        limiar_burnout_empuxo_rel      = cfg.limiar_burnout_empuxo_rel,
        tempo_confirmacao_burnout      = cfg.tempo_confirmacao_burnout,
        limiar_fallback_gradp          = cfg.limiar_fallback_gradp,
        limiar_fallback_temp           = cfg.limiar_fallback_temp,
        limiar_fallback_p              = cfg.limiar_fallback_p,
        modo_simulacao                 = cfg.modo_simulacao,
        modo_silencioso                = cfg.modo_silencioso,
    )
end

"""
    simular_caso(inp; cfg, salvar_csv=true) → SimulationResult

Executa a simulação 1D completa para o caso definido em `inp`.

# Argumentos obrigatórios
- `inp::CaseInput`      : parâmetros do caso (geometria, propelente, tubeira, malha)
- `cfg::ConfigModelo`   : configuração numérica (CFL, frequências de gravação, etc.)

# Argumentos opcionais

# Retorno
[`SimulationResult`](@ref) com históricos temporais + métricas integradas.
Suporta desestruturação retroativa:
```julia
res = simular_caso(inp; cfg=cfg)
tempos, pressoes, empuxos = res   # compatibilidade com scripts anteriores
res.I_total                       # acesso direto a métricas
```

# Exemplo
```julia
res = simular_caso(inp; cfg=ConfigModelo(cfl=0.25, modo_silencioso=true))
println("Isp = \$(res.Isp) s  |  I = \$(res.I_total) N·s")
```
"""
function simular_caso(inp::CaseInput;
                      cfg             ::ConfigModelo,
                      salvar_csv      ::Bool    = true)   # false → pula CSV/plots/análises (rápido p/ MC/V&V)

    # Checado AQUI e não só em validar_case_input porque o caminho 0D envolve o
    # diagnóstico de geometria num try/catch — um erro lançado lá vira aviso e a
    # simulação seguiria com um valor inválido.
    if !(0.0 < inp.eta_tubeira <= 1.0)
        error("eta_tubeira deve estar em (0, 1] — recebido $(inp.eta_tubeira). " *
              "Use 1.0 para desligar o derating de empuxo.")
    end

    # ── Dispatch: modo 0D (quasi-estático OU unsteady/transiente) ────────────
    # Retorna antecipadamente; as correções η_div / η_2ph são aplicadas em
    # _simular_0d_com_correcoes, com lógica idêntica ao bloco de correções 1D.
    # :zero_d          → P_eq algébrico (quasi-estático)
    # :zero_d_unsteady → EDO dP/dt (mesmo modelo do OpenMotor; ver Solver0D)
    #
    # ATENÇÃO: o cfg tem de passar por `_cfg_com_erosao` ANTES do dispatch. A GUI
    # só liga `inp.erosao_ativa`; quem traduz isso para `cfg.usar_erosao_garganta`
    # é este helper. Sem ele, o 0D recebia o cfg cru e a erosão da garganta era
    # silenciosamente ignorada (o caminho 1D já fazia a tradução mais abaixo).
    if cfg.modo_simulacao == :zero_d || cfg.modo_simulacao == :zero_d_unsteady
        return _simular_0d_com_correcoes(inp, _cfg_com_erosao(inp, cfg);
                                         salvar_csv = salvar_csv)
    end

    # ── Modo 1D (solver CFD MUSCL-HLLC) — código original ────────────────────
    spec, geom_lut, layout, y_burn_max, D_port_inicial = build_case_geometry(inp)

    if !cfg.modo_silencioso
        println("------------------------------------------------------")
        println("Caso: ", inp.name)
        println("Geometria: ", geom_lut.name)
        println("Kernel: SimulationCore (type-stable)")
        println("y_max: ", round(y_burn_max, digits=6), " m")
        println("D_port_ini equivalente: ", round(D_port_inicial, digits=6), " m")
        println("------------------------------------------------------")
    end

    prop = build_propellant_from_case(inp, layout, y_burn_max, D_port_inicial)
    ig   = _auto_ignitor(inp, layout, geom_lut)   # escalonado ao motor
    geom = build_domain_geometry(inp)

    cfg.modo_silencioso || @printf("  Ignitor auto: ṁ_ign = %.4f kg/s\n", ig.m_dot_ign)

    validar_geometria_caso(prop, geom)

    # Ativa erosão da garganta se solicitado em inp (ver `_cfg_com_erosao`).
    cfg_efetivo = _cfg_com_erosao(inp, cfg)

    resultado_sim = executar_simulacao_v2(inp.name, prop, ig, cfg_efetivo, geom;
                              N_malha         = inp.N_malha,
                              t_maximo        = inp.t_maximo,
                              salvar_csv      = salvar_csv)

    if resultado_sim === nothing
        cfg.modo_silencioso || println("Aviso: a simulação não retornou dados (possível divergência).")
        return nothing
    end

    # ── Avisos de segurança (revisão Fase 1: C1 malha, I2 teto de pressão) ─────
    # C1: o aviso antigo ("~3-7 % baixo, use N>=160-240") vinha de estudos de
    # convergência contaminados por dois erros corrigidos em 2026-09: o termo de
    # área não bem-balanceado e a área de queima perdida nas células de borda.
    # Com eles corrigidos, P_máx variou 0,00003 % entre N=80 e 120 (V3, BATES) e
    # a área de queima do motor de 3 m ficou em ±0,13 % com N=100/200/400. Fica só
    # a recomendação genérica para malhas realmente grossas.
    if !cfg.modo_silencioso && inp.N_malha < 80
        @warn @sprintf(
            "[Malha] N=%d é uma malha grossa: confirme P_máx repetindo com uma malha mais fina (ex.: 2N) antes de fechar a MEOP.",
            inp.N_malha)
    end
    # I2: pico próximo do teto de taxa de queima (clamp 50 MPa em physics.jl) trunca.
    if !cfg.modo_silencioso && resultado_sim.P_max > 40.0
        @warn @sprintf(
            "[Pressão] P_máx = %.1f MPa próximo do teto de taxa de queima (50 MPa) — o pico real pode estar TRUNCADO (subestimado). Não-conservador para dimensionamento.",
            resultado_sim.P_max)
    end

    # ── Fatores de correção de desempenho ─────────────────────────────────────
    # 1. Eficiência de divergência cônica  λ = (1 + cos α) / 2
    η_div = (1.0 + cosd(Float64(inp.alpha_divergencia))) / 2.0

    # 2. Eficiência de duas fases (partículas Al₂O₃ no escoamento da tubeira)
    #
    # Prioridade de modelo:
    #   (a) d_p_alumina_um > 0  → Stokes com diâmetro fornecido pelo usuário
    #   (b) D_core = 0          → escalar clássico (star/end_burner sem porto cil.)
    #   (c) default             → auto-predição D₄₃ via Hermsen (1981) + Stokes
    #   frac_alumina = 0        → η_2ph = 1 (sem alumínio)

    d43_predito_um = 0.0   # D₄₃ efetivo usado no cálculo de η_2ph [µm]; 0 = N/A

    η_2ph = if inp.frac_alumina > 0.0
        if inp.d_p_alumina_um > 0.0
            # (a) Diâmetro fornecido pelo usuário → modelo físico de Stokes + escala Dt
            d43_predito_um = inp.d_p_alumina_um
            calcular_2fases_fisica(inp.frac_alumina, inp.d_p_alumina_um,
                                   inp.Tc, inp.gamma, inp.R;
                                   Dt_m = inp.D_garganta_ini).eta_Isp

        elseif inp.D_core <= 0.0
            # (b) Grão sem porto cilíndrico (star, end_burner).
            # Hermsen requer τ = P·V_porto/(R·Tc·ṁ) — sem D_core, V_porto = 0.
            # Fallback: modelo escalar 1 − 0.14·ξ_ox (conservador).
            cfg.modo_silencioso || @printf(
                "  [η_2ph] D_core=0 (geometria %s) → modelo escalar\n",
                string(inp.geometry_type))
            calcular_correcao_duas_fases(inp.frac_alumina).eta_Isp

        else
            # (c) Auto-predição D₄₃ via Hermsen+porto (1981/Salita 1995) — modelo N9A corrigido
            ṁ_avg  = resultado_sim.m_consumida / max(resultado_sim.t_burn, 1e-9)

            # Volume inicial do porto:
            #   cilindro central + slots das aletas (finocyl) — ambos em m³
            # Cônico: tronco de cone, V = π/12·L·(D₁² + D₁D₂ + D₂²)
            V_cil = if inp.geometry_type == :finocyl_conico && inp.D_core_aft > 1e-6
                d1, d2 = inp.D_core, inp.D_core_aft
                (π / 12.0) * (d1^2 + d1 * d2 + d2^2) * inp.L_grao * Float64(inp.N_graos)
            elseif inp.geometry_type == :finocyl_cono_cil && inp.D_core_aft > 1e-6
                # aletado: cilindro D_core sobre L·ff ; cilíndrico: tronco D_core→D_core_aft sobre L·(1−ff)
                d1, d2 = inp.D_core, inp.D_core_aft
                ffv  = clamp(inp.fin_fraction, 0.0, 1.0)
                Vfin = (π / 4.0)  * d1^2 * inp.L_grao * ffv
                Vcyl = (π / 12.0) * (d1^2 + d1 * d2 + d2^2) * inp.L_grao * (1.0 - ffv)
                (Vfin + Vcyl) * Float64(inp.N_graos)
            else
                (π / 4.0) * inp.D_core^2 * inp.L_grao * Float64(inp.N_graos)
            end
            V_fins = if inp.geometry_type in (:finocyl, :finocyl_conico, :finocyl_cono_cil) && inp.n_fins > 0
                # Cada slot ≈ retângulo: largura × profundidade radial × comprimento axial
                Float64(inp.n_fins) * inp.fin_width * inp.fin_length *
                inp.L_grao * inp.fin_fraction * Float64(inp.N_graos)
            else
                0.0
            end
            V_porto = V_cil + V_fins

            # Tempo de residência: τ = ρ_c × V_porto / ṁ, ρ_c = P̄/(R·Tc) [s → ms]
            # P_avg está em MPa → converter para Pa (* 1e6) para τ e Hermsen
            tau_ms = 1000.0 * resultado_sim.P_avg * 1e6 * V_porto /
                     (inp.R * inp.Tc * max(ṁ_avg, 1e-9))

            # Hermsen corrigido: inclui fator de comprimento de porto (Salita 1995)
            # Importante para motores grandes onde L_grain/Dt >> 1
            L_grain_total = inp.L_grao * Float64(inp.N_graos)
            D43_um = calcular_d43_hermsen_corrigido(inp.D_garganta_ini, inp.frac_alumina,
                                                    resultado_sim.P_avg * 1e6, tau_ms,
                                                    L_grain_total; cap_f_porto = cfg.cap_f_porto)
            d43_predito_um = D43_um

            r2f = calcular_2fases_fisica(inp.frac_alumina, D43_um,
                                         inp.Tc, inp.gamma, inp.R;
                                         Dt_m = inp.D_garganta_ini)

            cfg.modo_silencioso || @printf(
                "  [Hermsen+porto] D₄₃ = %.1f µm  (Dt=%.1f mm, L/Dt=%.1f, τ=%.2f ms, P̄=%.2f MPa)\n",
                D43_um, inp.D_garganta_ini * 1e3, L_grain_total / inp.D_garganta_ini,
                tau_ms, resultado_sim.P_avg)
            cfg.modo_silencioso || @printf(
                "  [Stokes] f_geo=%.2f×  ψ_v=%.3f  ψ_T=%.3f  τ_nozzle=%.1f µs\n",
                r2f.f_geo, r2f.psi_v, r2f.psi_T, r2f.tau_nozzle * 1e6)

            r2f.eta_Isp
        end
    else
        1.0
    end

    # η_div (λ = (1+cosα)/2) já é aplicado dentro do SimulationCore:
    #   empuxo_cfd = F_puro * lambda_div * eta_bl   (linha ~512 de SimulationCore.jl)
    # Reaplicar aqui causaria dupla contagem da perda de divergência e subestimaria
    # o Isp em ~1.7% (α=15°) ou ~3% (α=20°).  Mantemos η_div apenas informativo.
    # Duas fases ACOPLADO (opt-in) — sobrepõe o η escalar quando ligado.
    η_2ph = _eta_2ph_final(η_2ph, cfg, inp, d43_predito_um, resultado_sim.P_avg, cfg.modo_silencioso)

    # Retenção de escória (slag): fração da Al₂O₃ que fica retida no motor e não
    # gera empuxo (ver ConfigModelo.frac_slag). Massa de Al₂O₃ = frac_alumina·(102/54)·m_prop.
    η_slag, slag_kg = _eta_slag(cfg.frac_slag, inp.frac_alumina, resultado_sim.m_consumida)
    η_tot  = η_2ph * η_slag   # duas fases × retenção de escória
    Isp_c  = resultado_sim.Isp     * η_tot
    Itot_c = resultado_sim.I_total * η_tot

    if !cfg.modo_silencioso && (η_div < 0.9999 || η_tot < 0.9999)
        @printf("  Correções: η_div=%.4f (α=%.1f°, já no CFD)  η_2F=%.4f (Al=%.0f%%)  → η_2ph=%.4f\n",
                η_div, inp.alpha_divergencia, η_2ph, inp.frac_alumina * 100.0, η_tot)
        η_slag < 0.9999 && @printf("  Escória (slag): frac_slag=%.0f%% → %.1f kg retidos, η_slag=%.4f\n",
                cfg.frac_slag * 100.0, slag_kg, η_slag)
        @printf("  Isp corrigido = %.1f s  (1D = %.1f s)\n",
                Isp_c, resultado_sim.Isp)
    end

    # ── Porteira de MEOP (segurança) ──────────────────────────────────────────
    P_meop_val, FS_meop_val = _verificar_meop(inp.P_meop_MPa, resultado_sim.P_max,
                                              cfg.modo_silencioso)

    # Reconstrói o SimulationResult com os fatores de correção preenchidos
    # (preserva os campos de erosão já calculados em _calcular_metricas_e_salvar)
    resultado_sim = SimulationResult(
        resultado_sim.caso,
        resultado_sim.tempos, resultado_sim.pressoes, resultado_sim.empuxos,
        resultado_sim.t_burn, resultado_sim.I_total, resultado_sim.Isp,
        resultado_sim.m_consumida,
        resultado_sim.P_max, resultado_sim.P_avg,
        resultado_sim.F_max, resultado_sim.F_avg, resultado_sim.CF_avg,
        resultado_sim.Kn_ini, resultado_sim.Kn_max,
        resultado_sim.n_fallbacks, resultado_sim.csv_path,
        η_div, η_2ph, η_tot, Isp_c, Itot_c,
        d43_predito_um,                       # D₄₃ efetivo [µm]
        resultado_sim.D_garganta_final,
        resultado_sim.erosao_radial_mm,
        resultado_sim.hist_D_garganta,
        resultado_sim.kn_hist,
        P_meop_val, FS_meop_val,              # MEOP (segurança)
    )


    return resultado_sim   # SimulationResult — suporta desestruturação retroativa
end

# ==============================================================================
# Helper: 0D + correções η (lógica idêntica ao bloco de pós-processamento 1D)
# ==============================================================================

"""
    _simular_0d_com_correcoes(inp, cfg) -> Union{SimulationResult, Nothing}

Executa o solver 0D quasi-estático e aplica as mesmas correções de eficiência
(η_div, η_2ph) usadas no modo 1D, para garantir resultados comparáveis.

Chamado automaticamente por `simular_caso` quando `cfg.modo_simulacao == :zero_d`.
"""
function _simular_0d_com_correcoes(inp::CaseInput, cfg::ConfigModelo;
                                   salvar_csv::Bool = false) ::Union{SimulationResult, Nothing}

    resultado_sim = cfg.modo_simulacao == :zero_d_unsteady ?
                    simular_0d_unsteady(inp, cfg) : simular_0d(inp, cfg)
    resultado_sim === nothing && return nothing

    # ── Fatores de correção de desempenho (mesma lógica do modo 1D) ──────────
    # 1. Eficiência de divergência cônica  λ = (1 + cos α) / 2
    #    Já embutida no Cf dentro de simular_0d (igual ao SimulationCore no modo 1D).
    #    Mantemos η_div apenas como campo informativo no SimulationResult.
    η_div = (1.0 + cosd(Float64(inp.alpha_divergencia))) / 2.0

    # 2. Eficiência de duas fases (partículas Al₂O₃)
    d43_predito_um = 0.0

    η_2ph = if inp.frac_alumina > 0.0
        if inp.d_p_alumina_um > 0.0
            # (a) Diâmetro fornecido pelo usuário → modelo físico de Stokes + escala Dt
            d43_predito_um = inp.d_p_alumina_um
            calcular_2fases_fisica(inp.frac_alumina, inp.d_p_alumina_um,
                                   inp.Tc, inp.gamma, inp.R;
                                   Dt_m = inp.D_garganta_ini).eta_Isp

        elseif inp.D_core <= 0.0
            # (b) Grão sem porto cilíndrico (star, end_burner) → modelo escalar
            cfg.modo_silencioso || @printf(
                "  [η_2ph 0D] D_core=0 (geometria %s) → modelo escalar\n",
                string(inp.geometry_type))
            calcular_correcao_duas_fases(inp.frac_alumina).eta_Isp

        else
            # (c) Auto-predição D₄₃ via Hermsen+porto (1981/Salita 1995) — modelo N9A corrigido
            ṁ_avg  = resultado_sim.m_consumida / max(resultado_sim.t_burn, 1e-9)

            V_cil  = (π / 4.0) * inp.D_core^2 * inp.L_grao * Float64(inp.N_graos)
            V_fins = if inp.geometry_type == :finocyl && inp.n_fins > 0
                Float64(inp.n_fins) * inp.fin_width * inp.fin_length *
                inp.L_grao * inp.fin_fraction * Float64(inp.N_graos)
            else
                0.0
            end
            V_porto = V_cil + V_fins

            # P_avg está em MPa → converter para Pa (* 1e6) para τ e Hermsen
            tau_ms = 1000.0 * resultado_sim.P_avg * 1e6 * V_porto /
                     (inp.R * inp.Tc * max(ṁ_avg, 1e-9))

            # Hermsen corrigido: inclui fator de comprimento de porto (Salita 1995)
            L_grain_total = inp.L_grao * Float64(inp.N_graos)
            D43_um = calcular_d43_hermsen_corrigido(inp.D_garganta_ini, inp.frac_alumina,
                                                    resultado_sim.P_avg * 1e6, tau_ms,
                                                    L_grain_total; cap_f_porto = cfg.cap_f_porto)
            d43_predito_um = D43_um

            r2f = calcular_2fases_fisica(inp.frac_alumina, D43_um,
                                         inp.Tc, inp.gamma, inp.R;
                                         Dt_m = inp.D_garganta_ini)

            cfg.modo_silencioso || @printf(
                "  [Hermsen+porto] D₄₃ = %.1f µm  (Dt=%.1f mm, L/Dt=%.1f, τ=%.2f ms, P̄=%.2f MPa)\n",
                D43_um, inp.D_garganta_ini * 1e3, L_grain_total / inp.D_garganta_ini,
                tau_ms, resultado_sim.P_avg)
            cfg.modo_silencioso || @printf(
                "  [Stokes] f_geo=%.2f×  ψ_v=%.3f  ψ_T=%.3f  τ_nozzle=%.1f µs\n",
                r2f.f_geo, r2f.psi_v, r2f.psi_T, r2f.tau_nozzle * 1e6)

            r2f.eta_Isp
        end
    else
        1.0
    end

    # Duas fases ACOPLADO (opt-in) — sobrepõe o η escalar quando ligado.
    η_2ph = _eta_2ph_final(η_2ph, cfg, inp, d43_predito_um, resultado_sim.P_avg, cfg.modo_silencioso)

    # η_div já está embutido no Cf do solver 0D — η_2ph × η_slag aplicados aqui
    η_slag, slag_kg = _eta_slag(cfg.frac_slag, inp.frac_alumina, resultado_sim.m_consumida)
    η_tot  = η_2ph * η_slag
    Isp_c  = resultado_sim.Isp     * η_tot
    Itot_c = resultado_sim.I_total * η_tot

    if !cfg.modo_silencioso && (η_div < 0.9999 || η_tot < 0.9999)
        @printf("  Correções: η_div=%.4f (α=%.1f°, já no Cf 0D)  η_2F=%.4f (Al=%.0f%%)  → η_2ph=%.4f\n",
                η_div, inp.alpha_divergencia, η_2ph, inp.frac_alumina * 100.0, η_tot)
        η_slag < 0.9999 && @printf("  Escória (slag): frac_slag=%.0f%% → %.1f kg retidos, η_slag=%.4f\n",
                cfg.frac_slag * 100.0, slag_kg, η_slag)
        @printf("  Isp corrigido = %.1f s  (0D = %.1f s)\n",
                Isp_c, resultado_sim.Isp)
    end

    P_meop_val, FS_meop_val = _verificar_meop(inp.P_meop_MPa, resultado_sim.P_max,
                                              cfg.modo_silencioso)

    # ── CSV das séries temporais ─────────────────────────────────────────────
    # O 1D grava resultados_v2_*.csv SEM cabeçalho (layout posicional herdado).
    # Aqui gravamos COM cabeçalho: o 0D não tem perfis 1D para casar, e o nome
    # das colunas evita ambiguidade em quem for ler (Excel, CompararCSV, etc).
    csv_path_0d = resultado_sim.csv_path
    if salvar_csv
        csv_path_0d = "resultados_0d_" * resultado_sim.caso * ".csv"
        try
            open(csv_path_0d, "w") do io
                println(io, "t_s,P_MPa,F_N,Kn")
                kn = resultado_sim.kn_hist
                for k in eachindex(resultado_sim.tempos)
                    knv = k <= length(kn) ? kn[k] : NaN
                    @printf(io, "%.6f,%.6f,%.4f,%.4f\n",
                            resultado_sim.tempos[k], resultado_sim.pressoes[k],
                            resultado_sim.empuxos[k], knv)
                end
            end
            cfg.modo_silencioso || println("Dados salvos em: $csv_path_0d")
        catch e
            @warn "Não consegui gravar o CSV do 0D" exception = e
            csv_path_0d = resultado_sim.csv_path
        end
    end

    return SimulationResult(
        resultado_sim.caso,
        resultado_sim.tempos, resultado_sim.pressoes, resultado_sim.empuxos,
        resultado_sim.t_burn, resultado_sim.I_total, resultado_sim.Isp,
        resultado_sim.m_consumida,
        resultado_sim.P_max, resultado_sim.P_avg,
        resultado_sim.F_max, resultado_sim.F_avg, resultado_sim.CF_avg,
        resultado_sim.Kn_ini, resultado_sim.Kn_max,
        resultado_sim.n_fallbacks, csv_path_0d,
        η_div, η_2ph, η_tot, Isp_c, Itot_c,
        d43_predito_um,
        resultado_sim.D_garganta_final,
        resultado_sim.erosao_radial_mm,
        resultado_sim.hist_D_garganta,
        resultado_sim.kn_hist,
        P_meop_val, FS_meop_val,              # MEOP (segurança)
    )
end
