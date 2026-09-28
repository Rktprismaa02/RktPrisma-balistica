# ==============================================================================
# ErosiveAnalysis.jl — Diagnóstico Espacial de Queima Erosiva
# ==============================================================================
# Calcula e visualiza o fluxo de massa axial G(x,t) e o aumento da taxa de queima
# por erosão a partir dos snapshots CFD — com o MESMO modelo erosivo que a
# simulação aplicou (`_taxa_erosiva`, cfg.modelo_erosivo) e SÓ na região do grão.
#
# Correções (2026-09):
#   • antes o laço cobria a malha inteira, incluindo a tubeira: o pico de G e a
#     "augmentação" de ~170% vinham da GARGANTA, onde não há propelente;
#   • antes usava sempre Lenoir-Robillard (α_e padrão), mesmo com a simulação em
#     Mukunda-Paul → o gráfico mostrava uma erosão que não foi aplicada;
#   • a linha fixa G_thr = 200 kg/(m²·s) foi trocada pelo limiar do modelo
#     (Mukunda: g_th = 35 no parâmetro adimensional g; Lenoir: cfg.G_erosao_lim);
#   • ρ do gás usa R cheio (EOS do núcleo, ver calcular_propriedades_escoamento_local).
#
# Uso (automático via SimulationCore._calcular_metricas_e_salvar):
#   res = analisar_erosao_espacial(prof_snaps, grain_snaps, prop, x_cells; cfg)
#   plotar_analise_erosiva(res; salvar=true)
# ==============================================================================

using Plots
using Printf
using Statistics

# ──────────────────────────────────────────────────────────────────────────────
# 1. ANÁLISE PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

"""
    analisar_erosao_espacial(prof_snaps, grain_snaps, prop, x_cells; cfg, ...) -> Dict

Perfis axiais, restritos ao grão (x ≤ comprimento da câmara), de:
- G(x,t) = ρ·|u|                                   [kg/(m²·s)]
- aumento erosivo aplicado = (r/r_base − 1)·100    [%], via `_taxa_erosiva`
- parâmetro de Mukunda g(x,t) = g₀·(Re₀/1000)^(−1/8) (só modelos Mukunda)

Sem `cfg`, ou com `cfg.usar_erosiva = false`, o aumento é 0 (a simulação não
aplicou erosiva) e o gráfico diz isso.
"""
function analisar_erosao_espacial(
    prof_snaps  ::Vector{SpatialSnapshot},
    grain_snaps ::Vector{GrainRegSnapshot},
    prop,
    x_cells     ::Vector{Float64};
    cfg         ::Union{Nothing, ConfigModelo} = nothing,
    nome_caso   ::String  = "motor"
) :: Dict{String,Any}

    isempty(prof_snaps)  && return Dict{String,Any}("erro" => "prof_snaps vazio")
    isempty(grain_snaps) && return Dict{String,Any}("erro" => "grain_snaps vazio")

    ns     = length(prof_snaps)
    rho_p  = prop.rho_p
    f_term = exp(prop.sigma_p * (prop.T_grain - prop.T_ref))

    erosiva_ativa = cfg !== nothing && cfg.usar_erosiva
    modelo  = cfg === nothing ? :nenhum : cfg.modelo_erosivo
    mukunda = modelo === :mukunda || modelo === :mukunda_std
    mu_gas  = cfg === nothing ? 9.0e-5 : cfg.mu_gas
    g_th    = 35.0
    G_lim   = cfg === nothing ? 0.0 : cfg.G_erosao_lim

    # Só as células do grão: na tubeira não há propelente (e a garganta tem o
    # maior G do domínio, o que dominava o gráfico antigo).
    L_cam  = comprimento_camara(prop)
    idx    = findall(x -> x <= L_cam, x_cells)
    x_grao = x_cells[idx]

    G_profiles   = Vector{Vector{Float64}}(undef, ns)
    aug_profiles = Vector{Vector{Float64}}(undef, ns)
    g_profiles   = Vector{Vector{Float64}}(undef, ns)
    G_max_global = 0.0

    for k in 1:ns
        snap_p = prof_snaps[k]
        snap_g = grain_snaps[k]
        nc     = length(idx)
        G_k    = zeros(nc); aug_k = zeros(nc); g_k = zeros(nc)

        for (j, i) in enumerate(idx)
            P_Pa  = max(snap_p.P[i] * 1e6, 1e3)     # MPa → Pa
            T_i   = max(snap_p.T[i], 100.0)
            rho_i = max(P_Pa / (prop.R * T_i), 1e-6)
            G_i   = rho_i * abs(snap_p.u[i])
            G_k[j] = G_i
            G_max_global = max(G_max_global, G_i)

            # Geometria local como na simulação (perímetro de queima da LUT)
            x_i  = x_cells[i]
            y_i  = clamp(snap_g.y_reg[i], 0.0, obter_ymax_local(x_i, prop))
            A_l, P_l = obter_geometria_local(x_i, y_i, prop)
            A_l  = max(A_l, 1e-8); P_l = max(P_l, 1e-6)
            D_h  = 4.0 * A_l / P_l
            d0   = modelo === :mukunda ? P_l / π : D_h

            r_base = max(prop.a * P_Pa^prop.n * f_term, 1e-12)
            if erosiva_ativa && y_i < obter_ymax_local(x_i, prop)
                r_tot    = _taxa_erosiva(r_base, max(G_i, 1e-6), d0, prop, cfg)
                aug_k[j] = (r_tot / r_base - 1.0) * 100.0
            end
            if mukunda
                Re0    = rho_p * r_base * d0 / mu_gas
                g_k[j] = G_i / (rho_p * r_base) * (Re0 / 1000.0)^(-0.125)
            end
        end
        G_profiles[k] = G_k; aug_profiles[k] = aug_k; g_profiles[k] = g_k
    end

    tempos      = [s.t for s in prof_snaps]
    G_max_snap  = [isempty(G) ? 0.0 : maximum(G) for G in G_profiles]
    G_mean_snap = [isempty(G) ? 0.0 : mean(G)    for G in G_profiles]
    g_max_snap  = [isempty(g) ? 0.0 : maximum(g) for g in g_profiles]
    aug_max     = erosiva_ativa ? maximum(maximum.(aug_profiles; init=0.0)) : 0.0
    nao_unif    = G_max_snap ./ max.(G_mean_snap, 1.0)

    metricas = Dict{String,Any}(
        "G_max_geral"   => G_max_global,
        "G_max_snap"    => G_max_snap,
        "G_mean_snap"   => G_mean_snap,
        "g_max_snap"    => g_max_snap,
        "nao_unif"      => nao_unif,
        "aug_max_pct"   => aug_max,
        "alpha_e"       => prop.alpha_e,
        "alpha_e_ativo" => erosiva_ativa,   # nome legado: "erosiva aplicada na simulação"
    )

    return Dict{String,Any}(
        "G_profiles"    => G_profiles,
        "aug_profiles"  => aug_profiles,
        "g_profiles"    => g_profiles,
        "tempos"        => tempos,
        "x_cells"       => x_grao,
        "modelo"        => modelo,
        "mukunda"       => mukunda,
        "g_th"          => g_th,
        "G_lim"         => G_lim,
        "alpha_e_ativo" => erosiva_ativa,
        "metricas"      => metricas,
        "nome_caso"     => nome_caso,
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# 2. VISUALIZAÇÃO — PAINEL 2×2
# ──────────────────────────────────────────────────────────────────────────────

"""
    plotar_analise_erosiva(resultado; salvar, dpi)

Painel 2×2 (só a região do grão):
- (1,1) Perfis G(x,t)
- (1,2) Aumento erosivo APLICADO pela simulação (mesmo modelo de physics.jl)
- (2,1) Evolução temporal de G_max e G_média no grão
- (2,2) Mukunda: parâmetro g(x,t) contra o limiar g_th; outros: não-uniformidade
"""
function plotar_analise_erosiva(
    resultado ::Dict{String,Any};
    salvar    ::Bool = true,
    dpi       ::Int  = 150
)
    haskey(resultado, "erro") && return nothing

    G_profs  = resultado["G_profiles"]
    aug_profs= resultado["aug_profiles"]
    g_profs  = resultado["g_profiles"]
    tempos   = resultado["tempos"]
    x_mm     = resultado["x_cells"] .* 1000.0
    ativo    = resultado["alpha_e_ativo"]
    mukunda  = resultado["mukunda"]
    modelo   = resultado["modelo"]
    g_th     = resultado["g_th"]
    G_lim    = resultado["G_lim"]
    m        = resultado["metricas"]
    nome     = resultado["nome_caso"]
    ns       = length(tempos)

    (isempty(G_profs) || ns == 0 || isempty(x_mm)) && return nothing

    nome_modelo = modelo === :mukunda     ? "Mukunda-Paul (d₀ = P/π)" :
                  modelo === :mukunda_std ? "Mukunda-Paul (d₀ = D_h)" :
                  modelo === :lenoir      ? "Lenoir-Robillard" : "—"
    pal = cgrad(:plasma, max(ns, 2); categorical=true)

    function _base(yl, tit)
        plot(; xlabel="Posição axial no grão  x  [mm]", ylabel=yl, title=tit,
               legend=:topleft, grid=true, gridalpha=0.3,
               framestyle=:box, titlefontsize=10)
    end

    # ── (1,1) G(x,t) no grão ─────────────────────────────────────────────────
    p1 = _base("G  [kg/(m²·s)]", "Fluxo de Massa Axial G(x,t) — grão")
    if modelo === :lenoir && G_lim > 0
        hline!(p1, [G_lim]; ls=:dash, lw=1.2, lc=:red, alpha=0.75,
               label=@sprintf("G_lim = %.0f kg/(m²·s)", G_lim))
    end
    for k in 1:ns
        plot!(p1, x_mm, G_profs[k]; lw=2, lc=pal[k], label=@sprintf("t = %.1f s", tempos[k]))
    end

    # ── (1,2) aumento erosivo APLICADO pela simulação ───────────────────────
    p2 = _base("(r / r_base − 1) × 100  [%]", "Aumento erosivo aplicado — $(nome_modelo)")
    hline!(p2, [0.0]; ls=:dot, lw=0.8, lc=:gray60, label="")
    for k in 1:ns
        plot!(p2, x_mm, aug_profs[k]; lw=2, lc=pal[k], label=@sprintf("t = %.1f s", tempos[k]))
    end
    if !ativo
        annotate!(p2, x_mm[max(1, end÷2)], 0.5,
                  text("Queima erosiva desligada na simulação", :center, 8, :gray60))
    end

    # ── (2,1) G_max e G_média no grão por snapshot ──────────────────────────
    p3 = plot(tempos, m["G_max_snap"];
        lw=2, lc=:tomato, label="G_max(t)",
        xlabel="Tempo  [s]", ylabel="G  [kg/(m²·s)]",
        title ="Evolução Temporal — G no grão",
        legend=:topright, grid=true, gridalpha=0.3,
        framestyle=:box, titlefontsize=10)
    plot!(p3, tempos, m["G_mean_snap"]; lw=2, lc=:steelblue, ls=:dash, label="G_média(t)")

    # ── (2,2) Mukunda: g(x,t) vs limiar g_th; senão não-uniformidade ─────────
    if mukunda
        p4 = _base("g = g₀·(Re₀/1000)^(−1/8)  [-]", "Parâmetro de Mukunda g(x,t)")
        hline!(p4, [g_th]; ls=:dash, lw=1.2, lc=:red, alpha=0.75,
               label=@sprintf("g_th = %.0f (limiar)", g_th))
        for k in 1:ns
            plot!(p4, x_mm, g_profs[k]; lw=2, lc=pal[k], label=@sprintf("t = %.1f s", tempos[k]))
        end
    else
        p4 = plot(tempos, m["nao_unif"];
            lw=2, lc=:purple, label="G_max / G_média",
            xlabel="Tempo  [s]", ylabel="Não-uniformidade  [-]",
            title ="Não-Uniformidade Axial de G (grão)",
            legend=:topleft, grid=true, gridalpha=0.3,
            framestyle=:box, titlefontsize=10)
        hline!(p4, [1.0]; ls=:dot, lw=0.8, lc=:gray60, label="perfil uniforme")
    end

    subtitle = ativo ?
        @sprintf("%s  |  aumento máx. aplicado no grão = %.1f %%", nome_modelo, m["aug_max_pct"]) :
        "Queima erosiva desligada na simulação"

    fig = plot(p1, p2, p3, p4;
        layout      = (2, 2),
        size        = (1100, 780),
        plot_title  = "Análise de Queima Erosiva — $(nome)\n$(subtitle)",
        plot_titlefontsize = 12,
        margin      = 8Plots.mm,
        top_margin  = 12Plots.mm,
        dpi         = dpi,
    )

    display(fig)
    if salvar
        fname = "erosao_espacial_$(nome).png"
        savefig(fig, fname)
        println("  ✅ Análise erosiva: $(fname)")
    end
    return fig
end
