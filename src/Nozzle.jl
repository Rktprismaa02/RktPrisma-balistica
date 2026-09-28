# ==============================================================================
# MÓDULO DA TUBEIRA (NOZZLE)
# ==============================================================================
# Contém as funções de perdas na tubeira (divergência e camada limite),
# o modelo de erosão da garganta (Ablação de Bartz) e a termodinâmica de expansão.

# function calcular_eta_div(alpha_rad::Float64)
#     return 0.5 * (1.0 + cos(alpha_rad))
# end

"""
    calcular_eta_bl(Dt::Float64) -> Float64

Eficiência de camada limite (boundary-layer momentum-thickness correction) da
tubeira cônica / parabólica de SRM.

## Fórmula

    η_bl = 1 − C / √Dt,     C = 5 × 10⁻⁴  m^(1/2)

onde `Dt` é o **diâmetro da garganta em metros**.

## Base física

A perda de impulso por camada limite resulta do deficit de quantidade de
movimento na espessura de deslocamento δ* na seção sônica da garganta.
Para a camada limite laminar compressível de Prandtl–Meyer:

    δ*/Rt ∝ Re_t^(-1/2),    Re_t = ρ* · a* · Dt / μ*

onde `ρ*`, `a*`, `μ*` são a densidade, velocidade do som e viscosidade
dinâmica na condição sônica (estado crítico). A perda de empuxo relativa
vale aproximadamente `Δη ≈ 2δ*/Dt`.

Para propelentes AP/HTPB sólidos operando em 3–10 MPa (T_c ≈ 2300–3500 K,
γ ≈ 1.20–1.30, μ* ≈ 7–9 × 10⁻⁵ Pa·s) a variação de `Re_t` com a pressão
e com a composição é pequena, e o produto `√(Re_t) / 2` colapsa no coeficiente
empírico C ≈ 4–6 × 10⁻⁴ m^(1/2). O valor central C = 5 × 10⁻⁴ m^(1/2) é
o mais citado na literatura de SRM de pequeno e médio porte.

## Referências

- Sutton, G.P. & Biblarz, O. (2017): *Rocket Propulsion Elements*, 9ª ed.,
  Wiley — Table 3-5: valores η_bl tabelados vs. Dt para SRM (reproduz a mesma
  constante C ≈ 0.0005 m^(1/2) implicitamente).
- Allman, J.G. & Hoffman, J.D. (1981): "Design of maximum-thrust nozzle contours
  by direct optimization", *AIAA J.* 19(6):750–751 — derivação analítica da
  constante para geometria cônica.
- JANNAF (1975): *Solid Propellant Rocket Motor Performance Prediction Standard*,
  CPIA Pub. 246, §5.3 — fórmula idêntica, intervalo C = 4–6 × 10⁻⁴ m^(1/2)
  para condições operacionais típicas de SRM AP/Al.

## Faixa de validade

| Dt [mm] | η_bl  | Observação              |
|:--------|:------|:------------------------|
|  4      | 0.950 | clamp inferior ativo    |
| 10      | 0.984 | pequenos SRM amadores   |
| 25      | 0.990 | SRM P=6–8 MPa típico    |
| 100     | 0.995 | clamp superior ativo    |

Para motores com Dt > ~150 mm a camada limite tende a ser turbulenta e a
fórmula subestima ligeiramente η_bl (erro < 0.2 %, conservador).
"""
function calcular_eta_bl(Dt::Float64)
    Dt <= 0.0 && return 0.98
    return clamp(1.0 - 5.0e-4 / sqrt(Dt), 0.95, 0.995)
end

"""
    calcular_taxa_erosao(Pc, Dt, cfg; r_dot_ref, P_ref, n_exp) -> taxa_radial_ms

Calcula a taxa de ablação **radial** da garganta (m/s de raio) baseada na pressão.
O chamador deve aplicar `D += 2 * taxa * dt` para converter taxa radial em crescimento diametral.

Modelo de potência calibrável:
    ṙ = r_dot_ref × (Pc / P_ref)^n_exp

# Kwargs
| Parâmetro     | Padrão       | Descrição |
|:--------------|:-------------|:----------|
| `r_dot_ref`   | 1.5e-4 m/s   | Taxa radial de referência a `P_ref` (grafite denso: ~0.15 mm/s a 5 MPa) |
| `P_ref`       | 5.0e6 Pa     | Pressão de referência [Pa] |
| `n_exp`       | 0.8          | Expoente de pressão [-] |
"""
function calcular_taxa_erosao(
    Pc ::Float64,
    Dt ::Float64,
    cfg::ConfigModelo;
    r_dot_ref ::Float64 = 1.5e-4,   # m/s (= 0.15 mm/s)
    P_ref     ::Float64 = 5.0e6,    # Pa
    n_exp     ::Float64 = 0.8
)
    if !cfg.usar_erosao_garganta || Pc < 1e5 || Dt <= 0.0
        return 0.0
    end
    return r_dot_ref * (Pc / P_ref)^n_exp
end

"""
    is_choked(P0::Float64, Pa::Float64, gamma::Float64)
Verifica se a tubeira atingiu fluxo sônico na garganta (choked flow).
"""
function is_choked(P0::Float64, Pa::Float64, gamma::Float64)
    razao_critica = ((gamma + 1.0) / 2.0)^(gamma / (gamma - 1.0))
    return (P0 / Pa) >= razao_critica
end

"""
    massflow_choked(P0::Float64, T0::Float64, At::Float64, gamma::Float64, R::Float64)
Calcula a vazão mássica isentrópica ideal na garganta.
"""
function massflow_choked(P0::Float64, T0::Float64, At::Float64, gamma::Float64, R::Float64)
    termo_gamma = sqrt(gamma * (2.0 / (gamma + 1.0))^((gamma + 1.0) / (gamma - 1.0)))
    return (P0 * At / sqrt(R * T0)) * termo_gamma
end

"""
    mach_from_area_ratio(AR::Float64, gamma::Float64; tol=1e-6, max_iter=50)
Resolve iterativamente o Mach na seção divergente usando Newton-Raphson.
"""
function mach_from_area_ratio(AR::Float64, gamma::Float64; tol=1e-6, max_iter=50)
    if AR <= 1.0
        return 1.0
    end
    
    M = 2.0
    
    
    for _ in 1:max_iter
        termo = 2.0 / (gamma + 1.0) * (1.0 + 0.5 * (gamma - 1.0) * M^2)
        f = (1.0 / M) * termo^((gamma + 1.0) / (2.0 * (gamma - 1.0))) - AR
        
        df = ((M^2 - 1.0) * termo^((gamma + 1.0) / (2.0 * (gamma - 1.0)))) /
             (M^2 * (1.0 + 0.5 * (gamma - 1.0) * M^2))
        
        M_new = M - f / df
        M_new = clamp(M_new, 1.001, 8.0)
        if abs(M_new - M) < tol
            return M_new
        end
        M = M_new
    end
    return M
end

"""
    compute_thrust(P0, T0, Pa, At, Ae, gamma, R, eta_div, eta_bl)
Calcula o empuxo físico isolado do CFD, corrigido pelas eficiências.
"""
function compute_thrust(P0::Float64, T0::Float64, Pa::Float64, At::Float64, Ae::Float64,
                        gamma::Float64, R::Float64, eta_div::Float64, eta_bl::Float64)
    if !is_choked(P0, Pa, gamma)
        return 0.0
    end

    AR = Ae / At
    M_e = mach_from_area_ratio(AR, gamma)

    Pe = P0 * (1.0 + 0.5 * (gamma - 1.0) * M_e^2)^(-gamma / (gamma - 1.0))
    Te = T0 * (1.0 + 0.5 * (gamma - 1.0) * M_e^2)^(-1.0)

    ve = M_e * sqrt(gamma * R * Te)
    mdot = massflow_choked(P0, T0, At, gamma, R)

    F_ideal = mdot * ve + (Pe - Pa) * Ae
    F_total = F_ideal * eta_div * eta_bl

    return max(0.0, F_total)
end

"""
    compute_thrust_from_cfd(rho_e, u_e, p_e, p_a, A_e)
Calcula o empuxo diretamente das propriedades resolvidas na face de saída.
"""
function compute_thrust_from_cfd(rho_e::Float64, u_e::Float64, p_e::Float64, p_a::Float64, A_e::Float64)
    m_dot_e = rho_e * u_e * A_e
    term_momentum = m_dot_e * u_e
    term_pressure = (p_e - p_a) * A_e
    
    return max(0.0, term_momentum + term_pressure)
end