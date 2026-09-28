# ==============================================================================
# Termoquimica.jl — função de Vandenkerckhove, c* e propriedades do gás
# ==============================================================================
# Extraído de ThermalAnalysis.jl do RktPrisma (seção 1): usado pelo modelo 0D,
# pelo CaseRunner e pela correção de duas fases.
# ==============================================================================

"""
    calcular_gamma_funcao(gamma) -> Γ

Função adimensional de Vandenkerckhove:
    Γ = √γ · [2/(γ+1)]^[(γ+1)/(2(γ-1))]

Usada no cálculo de c* teórico via c* = √(R·Tc) / Γ.
"""
@inline function calcular_gamma_funcao(gamma::Float64)
    exp_g = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    return sqrt(gamma) * (2.0 / (gamma + 1.0))^exp_g
end

"""
    calcular_cstar_teorico(prop) -> c* [m/s]

Velocidade característica teórica do propelente (com eficiência de c*):
    c* = √(R · Tc_eff) / Γ    onde  Tc_eff = Tc · η_c*²
"""
@inline calcular_cstar_teorico(R::Float64, Tc::Float64, gamma::Float64, eta_cstar::Float64) =
    sqrt(R * Tc * eta_cstar^2) / calcular_gamma_funcao(gamma)

calcular_cstar_teorico(prop::Propelente{L}) where {L} =
    calcular_cstar_teorico(prop.R, prop.Tc, prop.gamma, prop.eta_cstar)

"""
    calcular_prandtl_combustao(gamma) -> Pr

Estima o número de Prandtl dos gases de combustão APCP.
Correlação de Chapman-Enskog para gases poliatômicos [Ref. 3]:
    Pr ≈ 4γ / (9γ − 5)

Faixa típica: 0.70 – 0.85 para produtos de combustão HTPB/AP.
"""
@inline function calcular_prandtl_combustao(gamma::Float64)
    return 4.0 * gamma / (9.0 * gamma - 5.0)
end

"""
    calcular_viscosidade_combustao(T) -> μ [Pa·s]

Viscosidade dinâmica dos gases de combustão via lei de Sutherland.
Baseada em dados de N₂/CO₂ (produtos predominantes de HTPB/AP sem alumínio):
    μ = 1.458×10⁻⁶ · T^1.5 / (T + 110.4)  [Pa·s]
"""
@inline function calcular_viscosidade_combustao(T::Float64)
    return 1.458e-6 * T^1.5 / (T + 110.4)
end

