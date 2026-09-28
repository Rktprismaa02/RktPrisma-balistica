# ==============================================================================
# ChamberProfile.jl — Perfis Espaciais 1D do Escoamento Interno
# ==============================================================================
# Gera snapshots temporais do campo de escoamento ao longo do eixo do motor,
# revelando o que simuladores 0D (OpenMotor, BurnSim) são incapazes de mostrar:
# a distribuição espacial real de Mach, pressão, velocidade e temperatura.
#
# Diferencial vs simuladores gratuitos:
#   • OpenMotor / BurnSim: P(t) escalar — sem informação espacial
#   • SPP v5.0: M(x,t), P(x,t), u(x,t), T(x,t) — distribuição quasi-1D real
#
# Uso (automático via SimulationCore.jl):
#   Snapshots capturados automaticamente em 5%/20%/40%/60%/80% de t_maximo.
#   Painel 2×2 gerado automaticamente no pós-processamento.
# ==============================================================================

using Plots
using Printf
using Statistics

# ──────────────────────────────────────────────────────────────────────────────
# 1. ESTRUTURA DE DADOS
# ──────────────────────────────────────────────────────────────────────────────

"""
    SpatialSnapshot

Perfil espacial completo da solução CFD num instante `t`.
Contém as quatro variáveis de diagnóstico ao longo do eixo do motor.
"""
struct SpatialSnapshot
    t ::Float64              # instante de tempo [s]
    M ::Vector{Float64}      # número de Mach local [-]
    P ::Vector{Float64}      # pressão estática [MPa]
    u ::Vector{Float64}      # velocidade axial [m/s]
    T ::Vector{Float64}      # temperatura estática [K]
    G ::Vector{Float64}      # fluxo de massa por área ρ|u| [kg/(m²·s)]
    mdot ::Vector{Float64}   # vazão mássica ρ·u·A [kg/s] (na garganta = vazão do motor)
end

# ──────────────────────────────────────────────────────────────────────────────
# 2. EXTRACÇÃO DO SNAPSHOT A PARTIR DA SOLUÇÃO CFD
# ──────────────────────────────────────────────────────────────────────────────

"""
    calcular_snapshot(W, malha, gamma, R, t) -> SpatialSnapshot

Calcula o perfil espacial a partir do vetor de variáveis primitivas `W`.

# Argumentos
- `W`    : vetor de `SVector{3}(ρ, u, P)` por célula
- `malha`: `Malha1D`
- `gamma`: razão de calores específicos [–]
- `R`    : constante específica do gás [J/(kg·K)]
- `t`    : tempo actual [s]
"""
function calcular_snapshot(
    W     ::Vector{SVector{3,Float64}},
    malha ::Malha1D,
    gamma ::Float64,
    R     ::Float64,
    t     ::Float64
)::SpatialSnapshot
    N = malha.N
    M_v = Vector{Float64}(undef, N)
    P_v = Vector{Float64}(undef, N)
    u_v = Vector{Float64}(undef, N)
    T_v = Vector{Float64}(undef, N)
    G_v = Vector{Float64}(undef, N)
    m_v = Vector{Float64}(undef, N)

    @inbounds for i in 1:N
        rho_i = max(W[i][1], 1e-6)
        u_i   = W[i][2]
        P_i   = max(W[i][3], 1e3)          # mínimo 1 kPa (evita raiz negativa)
        T_i   = P_i / (rho_i * R)
        c_i   = sqrt(gamma * P_i / rho_i)  # velocidade do som local
        M_v[i] = abs(u_i) / max(c_i, 1.0)
        P_v[i] = P_i * 1e-6                # Pa → MPa
        u_v[i] = u_i
        T_v[i] = T_i
    end

    @inbounds for i in 1:N
        G_v[i] = W[i][1] * abs(W[i][2])
        m_v[i] = W[i][1] * W[i][2] * malha.A_centros[i]
    end
    return SpatialSnapshot(t, M_v, P_v, u_v, T_v, G_v, m_v)
end

# ──────────────────────────────────────────────────────────────────────────────
# 3. VISUALIZAÇÃO — PAINEL 2×2 DE PERFIS ESPACIAIS
# ──────────────────────────────────────────────────────────────────────────────

"""
    plotar_perfis_camara(snaps, x, x_garganta, nome_caso; salvar, dpi)

Gera painel 2×2 com perfis espaciais em múltiplos instantes temporais.

Codificação de cor: azul (início da queima) → vermelho (fim da queima).
Linha vertical tracejada marca a garganta. Linha pontilhada marca M = 1.

# Painéis
- (1,1) Número de Mach M(x)
- (1,2) Pressão P(x) [MPa]
- (2,1) Velocidade axial u(x) [m/s]
- (2,2) Temperatura estática T(x) [K]
"""
function plotar_perfis_camara(
    snaps      ::Vector{SpatialSnapshot},
    x          ::Vector{Float64},
    x_garganta ::Float64,
    nome_caso  ::String;
    salvar     ::Bool = true,
    dpi        ::Int  = 150
)
    isempty(snaps) && return nothing

    ns   = length(snaps)
    x_mm = x .* 1000.0        # m → mm para legibilidade
    xt   = x_garganta * 1000.0 # posição da garganta em mm

    # Paleta plasma: violeta (cedo) → amarelo (tarde)
    pal = cgrad(:plasma, ns; categorical = true)

    # Função auxiliar para criar sub-plot base
    function _base_plot(ylabel_str, title_str)
        plot(;
            xlabel      = "Posição axial  x  [mm]",
            ylabel      = ylabel_str,
            title       = title_str,
            legend      = :topright,
            grid        = true,
            gridalpha   = 0.3,
            framestyle  = :box,
            titlefontsize = 10,
        )
    end

    # ── Panel (1,1) — Número de Mach ─────────────────────────────────────────
    p1 = _base_plot("Mach  M  [–]", "Perfil de Número de Mach")
    vline!(p1, [xt];  ls = :dash, lw = 1.2, lc = :black,  label = "Garganta")
    hline!(p1, [1.0]; ls = :dot,  lw = 1.0, lc = :gray40, label = "M = 1")
    for (k, s) in enumerate(snaps)
        plot!(p1, x_mm, s.M; lw = 2, lc = pal[k],
              label = @sprintf("t = %.1f s", s.t))
    end

    # ── Panel (1,2) — Pressão ─────────────────────────────────────────────────
    p2 = _base_plot("Pressão  P  [MPa]", "Perfil de Pressão Estática")
    vline!(p2, [xt]; ls = :dash, lw = 1.2, lc = :black, label = "Garganta")
    for (k, s) in enumerate(snaps)
        plot!(p2, x_mm, s.P; lw = 2, lc = pal[k],
              label = @sprintf("t = %.1f s", s.t))
    end

    # ── Panel (2,1) — Velocidade axial ────────────────────────────────────────
    p3 = _base_plot("Velocidade  u  [m/s]", "Perfil de Velocidade Axial")
    vline!(p3, [xt]; ls = :dash, lw = 1.2, lc = :black, label = "Garganta")
    hline!(p3, [0.0]; ls = :dot, lw = 0.8, lc = :gray40, label = "")
    for (k, s) in enumerate(snaps)
        plot!(p3, x_mm, s.u; lw = 2, lc = pal[k],
              label = @sprintf("t = %.1f s", s.t))
    end

    # ── Panel (2,2) — Temperatura estática ───────────────────────────────────
    p4 = _base_plot("Temperatura  T  [K]", "Perfil de Temperatura Estática")
    vline!(p4, [xt]; ls = :dash, lw = 1.2, lc = :black, label = "Garganta")
    for (k, s) in enumerate(snaps)
        plot!(p4, x_mm, s.T; lw = 2, lc = pal[k],
              label = @sprintf("t = %.1f s", s.t))
    end

    # ── Compor painel 2×2 ────────────────────────────────────────────────────
    fig = plot(p1, p2, p3, p4;
        layout      = (2, 2),
        size        = (1100, 780),
        plot_title  = "Perfis Espaciais 1D — $(nome_caso)",
        margin      = 8Plots.mm,
        dpi         = dpi,
    )

    display(fig)
    if salvar
        fname = "perfis_camara_$(nome_caso).png"
        savefig(fig, fname)
        println("  ✅ Perfis espaciais: $(fname)")
    end

    return fig
end

# ──────────────────────────────────────────────────────────────────────────────
# 4. EXPORTAÇÃO CSV
# ──────────────────────────────────────────────────────────────────────────────

"""
    salvar_perfis_csv(snaps, x, nome_caso)

Exporta todos os snapshots para um único CSV.

Colunas: `x_m, M_t<t>, P_MPa_t<t>, u_ms_t<t>, T_K_t<t>, M_t<t+1>, ...`
"""
function salvar_perfis_csv(
    snaps     ::Vector{SpatialSnapshot},
    x         ::Vector{Float64},
    nome_caso ::String
)
    isempty(snaps) && return

    fname = "perfis_camara_$(nome_caso).csv"
    open(fname, "w") do io
        # Cabeçalho
        print(io, "x_m")
        for s in snaps
            ts = @sprintf("%.2f", s.t)
            print(io, ",M_t$(ts),P_MPa_t$(ts),u_ms_t$(ts),T_K_t$(ts),G_kg_m2s_t$(ts),mdot_kg_s_t$(ts)")
        end
        println(io)
        # Dados por célula
        for i in eachindex(x)
            @printf(io, "%.6f", x[i])
            for s in snaps
                @printf(io, ",%.6f,%.6f,%.4f,%.4f",
                        s.M[i], s.P[i], s.u[i], s.T[i])
                @printf(io, ",%.3f,%.4f", s.G[i], s.mdot[i])
            end
            println(io)
        end
    end
    println("  ✅ Perfis CSV: $(fname)")
end

# ──────────────────────────────────────────────────────────────────────────────
# 5. REGRESSÃO DO GRÃO — snapshot e visualização
# ──────────────────────────────────────────────────────────────────────────────

"""
    GrainRegSnapshot

Snapshot da regressão radial do grão num instante `t`.
Capturado automaticamente nos mesmos instantes dos `SpatialSnapshot`.
"""
struct GrainRegSnapshot
    t      ::Float64
    y_reg  ::Vector{Float64}   # regressão acumulada por célula [m]
    A_port ::Vector{Float64}   # área do porto por célula [m²]
end

"""
    plotar_regressao_grao(snaps, x, x_garganta, y_max, D_ext, At, nome_caso; salvar, dpi)

Gera painel 2×2 da evolução do grão ao longo do eixo axial em múltiplos instantes.

# Painéis
- (1,1) Fração de burnout β(x,t) = y(x,t)/y_max  [0→1]
- (1,2) Diâmetro equivalente do porto D_p(x,t) [mm]
- (2,1) A_port(x,t)/A_garganta — contribuição local ao Kn
- (2,2) Variação Δβ(x) entre 1º e último snapshot — mapa de não-uniformidade
"""
function plotar_regressao_grao(
    snaps      ::Vector{GrainRegSnapshot},
    x          ::Vector{Float64},
    x_garganta ::Float64,
    y_max      ::Float64,
    D_ext      ::Float64,
    At         ::Float64,
    nome_caso  ::String;
    salvar     ::Bool = true,
    dpi        ::Int  = 150
)
    (isempty(snaps) || y_max <= 0.0) && return nothing

    ns    = length(snaps)
    x_mm  = x .* 1000.0
    D_ext_mm = D_ext * 1000.0

    # Limita ao domínio da câmara (x ≤ x_garganta)
    idx_cam = findall(xi -> xi <= x_garganta * 1.01, x)
    isempty(idx_cam) && return nothing
    x_cam = x_mm[idx_cam]

    pal = cgrad(:viridis, max(ns, 2); categorical=true)

    function _base(ylabel_str, title_str)
        plot(;
            xlabel      = "Posição axial  x  [mm]",
            ylabel      = ylabel_str,
            title       = title_str,
            legend      = :best,
            grid        = true,
            gridalpha   = 0.3,
            framestyle  = :box,
            titlefontsize = 10,
        )
    end

    # ── (1,1) Fração de burnout β = y/y_max ─────────────────────────────────
    p1 = _base("Fração queimada  β  [–]", "Regressão Normalizada β(x,t)")
    ylims!(p1, (-0.03, 1.12))
    hline!(p1, [1.0]; ls=:dash, lw=1.2, lc=:red, label="burnout completo")
    for (k, s) in enumerate(snaps)
        β = s.y_reg[idx_cam] ./ y_max
        plot!(p1, x_cam, clamp.(β, 0.0, 1.0); lw=2, lc=pal[k],
              label=@sprintf("t = %.1f s", s.t))
    end

    # ── (1,2) Diâmetro equivalente do porto D_p(x,t) [mm] ──────────────────
    p2 = _base("D_porto efetivo  [mm]", "Evolução do Diâmetro do Porto")
    hline!(p2, [D_ext_mm]; ls=:dash, lw=1.2, lc=:black, label="D externo")
    for (k, s) in enumerate(snaps)
        Dp = 2.0 .* sqrt.(max.(s.A_port[idx_cam], 1e-12) ./ π) .* 1000.0
        plot!(p2, x_cam, Dp; lw=2, lc=pal[k],
              label=@sprintf("t = %.1f s", s.t))
    end

    # ── (2,1) A_port/A_garganta — perfil de Kn axial ────────────────────────
    p3 = _base("A_porto / A_garganta  [–]", "Razão de Área Local  (proxy do Kn axial)")
    for (k, s) in enumerate(snaps)
        kn_local = s.A_port[idx_cam] ./ max(At, 1e-10)
        plot!(p3, x_cam, kn_local; lw=2, lc=pal[k],
              label=@sprintf("t = %.1f s", s.t))
    end

    # ── (2,2) Não-uniformidade de queima Δβ(x) ──────────────────────────────
    p4 = _base("Variação  Δβ  [–]", "Não-uniformidade de Queima  (último − 1º snapshot)")
    if ns >= 2
        y1  = snaps[1].y_reg[idx_cam]
        yn  = snaps[end].y_reg[idx_cam]
        Δβ  = clamp.((yn .- y1) ./ y_max, 0.0, 1.0)
        Δt  = snaps[end].t - snaps[1].t
        avg = mean(Δβ)
        plot!(p4, x_cam, Δβ; lw=2, lc=:steelblue,
              label=@sprintf("Δt = %.1f s", Δt))
        hline!(p4, [avg]; ls=:dash, lw=1, lc=:orange,
               label=@sprintf("média = %.3f", avg))
    else
        plot!(p4, [0.0], [0.0]; label="(necessário ≥ 2 snapshots)")
    end

    fig = plot(p1, p2, p3, p4;
        layout      = (2, 2),
        size        = (1100, 780),
        plot_title  = "Regressão do Grão — $(nome_caso)",
        margin      = 8Plots.mm,
        dpi         = dpi,
    )

    display(fig)
    if salvar
        fname = "regressao_grao_$(nome_caso).png"
        savefig(fig, fname)
        println("  ✅ Regressão do grão: $(fname)")
    end
    return fig
end
