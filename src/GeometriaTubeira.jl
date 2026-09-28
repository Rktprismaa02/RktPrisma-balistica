# ==============================================================================
# GeometriaTubeira.jl — gás quente e contorno da tubeira
# ==============================================================================
# Extraído de ThermalNozzle.jl do RktPrisma: só as definições usadas pelo modelo
# de duas fases acoplado (TwoPhaseNozzle.jl). A condução térmica transiente
# daquele arquivo não faz parte deste repositório.
# ==============================================================================

# ==============================================================================
# 1. ESTADO DO GÁS QUENTE
# ==============================================================================

"""
    GasQuente

Propriedades do gás de combustão necessárias à análise térmica, em números
puros — sem depender do tipo `Propelente`. É isto que permite executar a
análise em modo autónomo.

# Campos
| Campo          | Unidade   | Descrição |
|:---------------|:----------|:----------|
| `T0`           | K         | Temperatura de estagnação (câmara) |
| `gamma`        | —         | Razão de calores específicos |
| `R`            | J/(kg·K)  | Constante específica do gás |
| `mu`           | Pa·s      | Viscosidade dinâmica (0 = Sutherland a T₀) |
| `Pr`           | —         | Número de Prandtl (0 = Chapman-Enskog a partir de γ) |
| `frac_alumina` | —         | Fracção mássica de Al no propelente (correcção de duas fases) |
| `eta_cstar`    | —         | Eficiência de c* (T₀ efectiva = T₀·η²) |
"""
struct GasQuente
    T0           ::Float64
    gamma        ::Float64
    R            ::Float64
    mu           ::Float64
    Pr           ::Float64
    frac_alumina ::Float64
    eta_cstar    ::Float64
end

"""
    GasQuente(; T0, gamma, R, mu=0.0, Pr=0.0, frac_alumina=0.0, eta_cstar=1.0)

Construtor por palavra-chave. Com `mu = 0` a viscosidade é estimada pela lei de
Sutherland avaliada à temperatura efectiva de câmara; com `Pr = 0` o Prandtl é
estimado por `Pr ≈ 4γ/(9γ−5)`.
"""
function GasQuente(; T0::Real, gamma::Real, R::Real,
                     mu::Real = 0.0, Pr::Real = 0.0,
                     frac_alumina::Real = 0.0, eta_cstar::Real = 1.0)
    T0 > 0.0    || error("GasQuente: T0 deve ser > 0.")
    gamma > 1.0 || error("GasQuente: gamma deve ser > 1.")
    R > 0.0     || error("GasQuente: R deve ser > 0.")
    return GasQuente(Float64(T0), Float64(gamma), Float64(R), Float64(mu),
                     Float64(Pr), Float64(frac_alumina), Float64(eta_cstar))
end

"""
    GasQuente(prop::Propelente) -> GasQuente

Extrai as propriedades do gás a partir de um `Propelente` do simulador
(modo acoplado).
"""
GasQuente(prop) = GasQuente(; T0 = prop.Tc, gamma = prop.gamma, R = prop.R,
                              frac_alumina = prop.frac_alumina,
                              eta_cstar = prop.eta_cstar)

# ==============================================================================
# 2. GEOMETRIA DA TUBEIRA
# ==============================================================================

"""
    GeometriaTubeira

Perfil interno da tubeira discretizado em estações axiais.

# Campos
- `x`      : posição axial [m]
- `R`      : raio da superfície quente [m]
- `A`      : área da secção [m²]
- `regiao` : `:convergente`, `:garganta` ou `:divergente`
- `x_t`, `R_t`, `A_t` : parâmetros da garganta
"""
struct GeometriaTubeira
    x      ::Vector{Float64}
    R      ::Vector{Float64}
    A      ::Vector{Float64}
    regiao ::Vector{Symbol}
    x_t    ::Float64
    R_t    ::Float64
    A_t    ::Float64
    L_conv ::Float64
    L_div  ::Float64
end

"""
    tubeira_conica(; D_entrada, D_garganta, D_saida, alpha_conv_deg,
                     alpha_div_deg, L_garganta=0.0, N_x=101) -> GeometriaTubeira

Reconstrói o perfil interno de uma tubeira cónica a partir de diâmetros e
semiângulos — **modo autónomo**, sem precisar de simulação.

Os comprimentos são calculados pelos ângulos:

    L_conv = (R_entrada − R_garganta)/tan(α_conv)
    L_div  = (R_saída   − R_garganta)/tan(α_div)

Todos os diâmetros são **internos**, isto é, da superfície quente do isolante.
"""
function tubeira_conica(; D_entrada::Real, D_garganta::Real, D_saida::Real,
                          alpha_conv_deg::Real = 45.0,
                          alpha_div_deg ::Real = 15.0,
                          L_garganta    ::Real = 0.0,
                          N_x           ::Int  = 101)

    D_garganta > 0.0 || error("tubeira_conica: D_garganta deve ser > 0.")
    D_entrada > D_garganta ||
        error("tubeira_conica: D_entrada ($D_entrada m) deve ser maior que D_garganta ($D_garganta m).")
    D_saida > D_garganta ||
        error("tubeira_conica: D_saida ($D_saida m) deve ser maior que D_garganta ($D_garganta m).")
    N_x >= 5 || error("tubeira_conica: N_x deve ser >= 5.")

    a_conv = deg2rad(Float64(alpha_conv_deg))
    a_div  = deg2rad(Float64(alpha_div_deg))
    (0.0 < a_conv < π/2) || error("tubeira_conica: alpha_conv_deg deve estar em (0, 90).")
    (0.0 < a_div  < π/2) || error("tubeira_conica: alpha_div_deg deve estar em (0, 90).")

    R_ent = D_entrada  / 2.0
    R_t   = D_garganta / 2.0
    R_sai = D_saida    / 2.0
    L_g   = Float64(L_garganta)

    L_conv = (R_ent - R_t) / tan(a_conv)
    L_div  = (R_sai - R_t) / tan(a_div)
    L_tot  = L_conv + L_g + L_div

    x_ini_g = L_conv
    x_fim_g = L_conv + L_g

    x      = collect(range(0.0, L_tot; length = N_x))
    R      = zeros(N_x)
    regiao = Vector{Symbol}(undef, N_x)

    @inbounds for i in 1:N_x
        xi = x[i]
        if xi < x_ini_g
            R[i]      = R_ent - xi * tan(a_conv)
            regiao[i] = :convergente
        elseif xi <= x_fim_g
            R[i]      = R_t
            regiao[i] = :garganta
        else
            R[i]      = R_t + (xi - x_fim_g) * tan(a_div)
            regiao[i] = :divergente
        end
    end

    A = @. π * R^2
    return GeometriaTubeira(x, R, A, regiao, x_ini_g, R_t, π * R_t^2, L_conv, L_div)
end

"""
    tubeira_de_simulacao(prop, geom; N_x=101) -> GeometriaTubeira

Constrói a geometria da tubeira a partir do `Dict` de geometria do simulador
(**modo acoplado**), usando o mesmo modelo cónico de `atualizar_geometria_tubeira!`.

`geom` deve conter `x_garganta`, `D_garganta_ini`, `D_saida` e `L_total`.
"""
function tubeira_de_simulacao(prop, geom::Dict; N_x::Int = 101)
    L_cam   = comprimento_camara(prop)
    x_t     = Float64(geom["x_garganta"])
    D_t     = Float64(geom["D_garganta_ini"])
    D_saida = Float64(geom["D_saida"])
    L_total = Float64(geom["L_total"])

    R_cam = prop.D_ext / 2.0
    R_t   = D_t / 2.0
    R_sai = D_saida / 2.0

    L_conv = x_t - L_cam
    L_div  = L_total - x_t
    L_conv > 0.0 || error("tubeira_de_simulacao: L_conv <= 0 — verifique x_garganta e o comprimento do grão.")
    L_div  > 0.0 || error("tubeira_de_simulacao: L_div <= 0 — verifique L_total e x_garganta.")

    x      = collect(range(L_cam, L_total; length = N_x))
    R      = zeros(N_x)
    regiao = Vector{Symbol}(undef, N_x)

    @inbounds for i in 1:N_x
        xi = x[i]
        if xi <= x_t
            f         = (xi - L_cam) / L_conv
            R[i]      = R_cam + f * (R_t - R_cam)
            regiao[i] = f >= 1.0 - 1e-9 ? :garganta : :convergente
        else
            f         = (xi - x_t) / L_div
            R[i]      = R_t + f * (R_sai - R_t)
            regiao[i] = :divergente
        end
    end

    A = @. π * R^2
    return GeometriaTubeira(x, R, A, regiao, x_t, R_t, π * R_t^2, L_conv, L_div)
end

"""
    tubeira_por_dimensoes(; D_camara, D_garganta, D_saida, L_camara, x_garganta,
                            L_total, N_x=101) -> GeometriaTubeira

Constrói a geometria a partir de comprimentos e diâmetros explícitos, sem
depender de nenhum tipo do simulador. É a via usada pela GUI no modo "última
simulação", onde os valores vêm do `CaseInput` já validado.

Convergente de `D_camara` a `D_garganta` entre `L_camara` e `x_garganta`;
divergente de `D_garganta` a `D_saida` entre `x_garganta` e `L_total`.
"""
function tubeira_por_dimensoes(; D_camara::Real, D_garganta::Real, D_saida::Real,
                                 L_camara::Real, x_garganta::Real, L_total::Real,
                                 N_x::Int = 101)
    L_conv = Float64(x_garganta) - Float64(L_camara)
    L_div  = Float64(L_total)    - Float64(x_garganta)
    L_conv > 0.0 || error("tubeira_por_dimensoes: L_conv <= 0 — x_garganta " *
                          "($x_garganta m) deve ser maior que o fim da câmara ($L_camara m).")
    L_div  > 0.0 || error("tubeira_por_dimensoes: L_div <= 0 — L_total " *
                          "($L_total m) deve ser maior que x_garganta ($x_garganta m).")
    D_garganta > 0.0 || error("tubeira_por_dimensoes: D_garganta deve ser > 0.")

    R_cam = Float64(D_camara)/2.0
    R_t   = Float64(D_garganta)/2.0
    R_sai = Float64(D_saida)/2.0
    x_t   = Float64(x_garganta)

    x      = collect(range(Float64(L_camara), Float64(L_total); length = N_x))
    R      = zeros(N_x)
    regiao = Vector{Symbol}(undef, N_x)

    @inbounds for i in 1:N_x
        xi = x[i]
        if xi <= x_t
            f         = (xi - Float64(L_camara)) / L_conv
            R[i]      = R_cam + f*(R_t - R_cam)
            regiao[i] = f >= 1.0 - 1e-9 ? :garganta : :convergente
        else
            f         = (xi - x_t) / L_div
            R[i]      = R_t + f*(R_sai - R_t)
            regiao[i] = :divergente
        end
    end
    A = @. π * R^2
    return GeometriaTubeira(x, R, A, regiao, x_t, R_t, π*R_t^2, L_conv, L_div)
end

# ==============================================================================
# 3. UTILITÁRIOS
# ==============================================================================

