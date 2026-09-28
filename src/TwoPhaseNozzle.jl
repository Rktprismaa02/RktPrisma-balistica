# ==============================================================================
# TwoPhaseNozzle.jl — Escoamento bifásico ACOPLADO gás–partícula na tubeira
# ==============================================================================
#
# Substitui a eficiência escalar `η_2ph` por uma perda CALCULADA: integra as
# equações acopladas de gás e partículas condensadas (Al₂O₃) ao longo do
# divergente e mede o défice de impulso que a defasagem produz.
#
# ── O que muda face ao modelo escalar ────────────────────────────────────────
#
# O modelo anterior (`calcular_2fases_fisica`) resolve `τ_nozzle` de trás para
# frente, de modo a que uma partícula de 3 µm reproduza `η = 1 − 0.14·ξ_ox`, e
# depois escala por `clamp(Dt/25mm, 1, 5)`. O tempo de residência não vem da
# tubeira: vem da calibração. Aqui ele é consequência de integrar a partícula
# pela geometria real.
#
# ── Física ───────────────────────────────────────────────────────────────────
#
# Partícula (Lagrangiana ao longo da sua trajectória, regime estacionário):
#
#     u_p du_p/dx = (u_g − u_p)/τ_v         τ_v = ρ_s d²/(18 μ_g f_D)
#     u_p dT_p/dx = (T_g − T_p)/τ_T         τ_T = ρ_s c_s d²/(6 Nu k_g)
#
# com correcção de Schiller–Naumann no arrasto e Ranz–Marshall no Nusselt.
#
# Gás: continuidade e energia da mistura dão ρ_g, T_g e p de forma explícita.
# A quantidade de movimento da mistura, depois de eliminar dp/dx, dá
#
#     (ṁ_g/γ)(1 − 1/M²) du_g/dx = −ṁ_p du_p/dx
#                                  + (A p Y)/(T_g cp_g)·(c_s dT_p/dx + u_p du_p/dx)
#                                  + p dA/dx
#
# ── Onde está a singularidade sónica (leia isto antes de mexer) ──────────────
#
# A forma acima, com du_p/dx explícito, engana. `du_p/dx` NÃO é independente de
# `du_g/dx`: quanto mais acoplada a partícula, mais ela segue o gás. Escrevendo
# a relaxação sobre o passo obtém-se a forma afim du_p/dx = A_c + B_c·du_g/dx, e
# passando B_c·du_g/dx para a esquerda o coeficiente efectivo fica
#
#     (ṁ_g/γ)(1 − 1/M_gás²) + ṁ_p·B_c − (A p Y u_p/(T_g cp_g))·B_c
#
# que se anula em sítios DIFERENTES conforme o acoplamento:
#   • B_c → 1 (partícula colada): anula-se no ponto sónico da MISTURA, onde
#     M_gás = 1/√(1+Y) < 1. A mistura tem som mais lento que o gás.
#   • B_c → 0 (partícula congelada): anula-se em M_gás = 1, o sónico do gás.
#
# Sobre um passo de integração infinitesimal tem-se sempre Δt/τ_v → 0, logo
# B_c → 0: a singularidade que a integração vê LOCALMENTE é a do gás. Por isso
# se arranca no ponto sónico do gás — que fica a jusante da garganta, tanto mais
# quanto maior a carga de partículas — e não numa razão de áreas fixa.
#
# Errar isto não dá um erro pequeno: dá η > 1, ou seja, o modelo a prever LUCRO
# de impulso por haver alumina no escoamento. Já aconteceu duas vezes durante o
# desenvolvimento. Há uma guarda dura sobre M_gás no arranque, e o testset 14
# varre a carga de 2 % a 30 % de Al justamente para apanhar isto.
#
# O convergente não é resolvido: lá as partículas estão em equilíbrio
# (velocidades baixas, tempo de trânsito longo) e é no divergente que a
# defasagem nasce.
#
# ── Limites de referência ────────────────────────────────────────────────────
#
# O resultado é enquadrado por dois limites analíticos, que servem de teste:
#   • EQUILÍBRIO — partículas sempre coladas ao gás (d → 0). Máximo Isp.
#     A mistura comporta-se como um gás único com cp e R efectivos.
#   • CONGELADO — partículas nunca aceleram (d → ∞). Mínimo Isp.
# A solução acoplada tem de cair entre os dois, e tender a cada um deles nos
# limites de tamanho de partícula. É isso que os testes verificam.
#
# Referências:
#   [1] Kliegel, J.R. & Nickerson, G.R. (1962). "Flow of Gas-Particle Mixtures
#       in Axially Symmetric Nozzles." ARS Progress in Astronautics.
#   [2] Crowe, C.T. (1967). "Drag Coefficient of Particles in a Rocket Nozzle."
#       AIAA Journal 5(5):1021-1022.
#   [3] Hermsen, R.W. (1981). "Aluminum Oxide Particle Size for Solid Rocket
#       Motor Performance Prediction." J. Spacecraft 18(6):483-490.
#   [4] Ranz, W.E. & Marshall, W.R. (1952). Chem. Eng. Prog. 48:141-146.
#
# Dependências: ThermalNozzle.jl (GeometriaTubeira, GasQuente).
# ==============================================================================

using Printf

# ==============================================================================
# 1. PROPRIEDADES
# ==============================================================================

"""
    PropriedadesParticula

Fase condensada. Os valores por omissão são de Al₂O₃ líquido/sólido à
temperatura de câmara típica de APCP aluminizado.

| Campo | Unidade | Descrição |
|:--|:--|:--|
| `rho_s` | kg/m³ | densidade do material da partícula |
| `cp_s`  | J/(kg·K) | calor específico |
| `d43`   | m | diâmetro médio de Sauter/De Brouckere |
| `T_fusao`  | K | temperatura de fusão |
| `dH_fusao` | J/kg | calor latente de fusão; `0` desliga a mudança de fase |

# Mudança de fase
A Al₂O₃ sai da câmara LÍQUIDA (T₀ ≈ 3100 K contra fusão a 2327 K) e solidifica
dentro da tubeira, libertando 1.07 MJ/kg. Com 30 % de fase condensada isso são
~320 kJ/kg de mistura, ou ~5 % da entalpia total — não é desprezável.

Quan & Kliegel tratam-no (Eqs. 2-10/2-11 ramificam a entalpia da partícula em
`T_p ≷ T_pm`, com o termo `ΔH`); uma versão anterior deste módulo não, e isso
era parte do défice de Isp face a referências experimentais.
"""
Base.@kwdef struct PropriedadesParticula
    rho_s    ::Float64 = 3990.0    # Al₂O₃
    cp_s     ::Float64 = 1260.0    # ~1500 K
    d43      ::Float64 = 3.0e-6
    T_fusao  ::Float64 = 2327.0    # Al₂O₃ funde a 2054 °C
    dH_fusao ::Float64 = 1.07e6
end

"""
    _avancar_fase(Tp0, fliq0, Tp_relax, part) -> (T_p, f_liq)

Aplica a parada de solidificação sobre a temperatura que a relaxação daria.

Enquanto houver líquido, a partícula não pode arrefecer abaixo da fusão: o calor
que sairia converte-se em solidificação a temperatura constante. `f_liq` é a
fracção ainda líquida. Só quando ela chega a zero é que a temperatura volta a
descer, e o excedente do passo é aplicado então — o que evita que um passo largo
"salte" a parada e perca o calor latente.
"""
@inline function _avancar_fase(Tp0::Float64, fliq0::Float64, Tp_relax::Float64,
                               part::PropriedadesParticula)
    (part.dH_fusao <= 0.0 || fliq0 <= 0.0) && return (Tp_relax, 0.0)
    Tp_relax >= part.T_fusao && return (Tp_relax, fliq0)          # ainda líquida

    # A relaxação tentou descer abaixo da fusão. A parte abaixo dela paga-se em
    # solidificação: Δf = c_s·ΔT_excesso / ΔH.
    Δf = part.cp_s * (part.T_fusao - Tp_relax) / part.dH_fusao
    if Δf >= fliq0
        sobra = (Δf - fliq0) * part.dH_fusao / part.cp_s   # já não há o que solidificar
        return (part.T_fusao - sobra, 0.0)
    end
    return (part.T_fusao, fliq0 - Δf)
end

# Piso da defasagem fraccionária inicial K = u_p/u_g. A relação de Kliegel
# pressupõe que a partícula JÁ atingiu a defasagem estacionária ao chegar à linha
# inicial. Para partículas muito grandes devolve valores absurdos como condição
# inicial — a 500 µm dá K = 0.08, isto é, partícula a 90 m/s com gás a 1085, que
# nunca teria atravessado o convergente para lá chegar. O piso marca onde a
# hipótese deixa de se sustentar; `metricas["K_lag_saturado"]` diz se foi
# accionado.
const K_MIN_LAG = 0.15

"""
    _mu_gas(T), _k_gas(T)

Viscosidade e condutividade dos produtos de combustão, por lei de potência
ancorada em 3000 K. Mesmas constantes do modelo escalar, para que a comparação
entre os dois isole o efeito do ACOPLAMENTO e não o das propriedades.
"""
@inline _mu_gas(T::Float64) = 7.0e-5 * (T / 3000.0)^0.7
@inline _k_gas(T::Float64)  = 0.40   * (T / 3000.0)^0.7

"""
    _tau_v(d, rho_s, mu, Re) -> s

Tempo de relaxação de velocidade, Stokes com correcção de Schiller–Naumann:

    f_D = 1 + 0.15·Re_p^0.687      (Re_p ≲ 1000)

Sem a correcção, partículas grandes em gás rápido teriam arrasto subestimado —
justamente o regime que domina a perda em motores grandes.
"""
@inline function _tau_v(d::Float64, rho_s::Float64, mu::Float64, Re::Float64)
    f_D = 1.0 + 0.15 * max(Re, 0.0)^0.687
    return rho_s * d^2 / (18.0 * mu * f_D)
end

"""
    _tau_T(d, rho_s, cp_s, k, Re, Pr) -> s

Tempo de relaxação térmica com Nusselt de Ranz–Marshall:

    Nu = 2 + 0.6·Re_p^{1/2}·Pr^{1/3}

Com `Nu = 2` (partícula parada) recupera-se `τ_T = ρ_s c_s d²/(12 k)`, que é a
forma usada no modelo escalar.
"""
@inline function _tau_T(d::Float64, rho_s::Float64, cp_s::Float64,
                        k::Float64, Re::Float64, Pr::Float64)
    Nu = 2.0 + 0.6 * sqrt(max(Re, 0.0)) * cbrt(max(Pr, 1e-6))
    return rho_s * cp_s * d^2 / (6.0 * Nu * k)
end

# ==============================================================================
# 2. RESULTADO
# ==============================================================================

"""
    ResultadoBifasico

Saída da integração acoplada.

| Campo | Descrição |
|:--|:--|
| `x`, `A` | malha axial e área [m, m²] |
| `u_g`, `u_p` | velocidades de gás e partícula [m/s] |
| `T_g`, `T_p` | temperaturas [K] |
| `p`, `M` | pressão estática [Pa] e Mach do gás |
| `Isp_acoplado` | Isp calculado com a defasagem [s] |
| `Isp_equilibrio` | limite superior (partículas coladas) [s] |
| `Isp_congelado` | limite inferior (partículas não aceleram) [s] |
| `eta_2ph` | `Isp_acoplado / Isp_equilibrio` — a perda CALCULADA |
| `lag_vel`, `lag_term` | defasagem relativa na saída [–] |
| `tau_residencia` | tempo de trânsito real da partícula [s] |
"""
struct ResultadoBifasico
    x   ::Vector{Float64}
    A   ::Vector{Float64}
    u_g ::Vector{Float64}
    u_p ::Vector{Float64}
    T_g ::Vector{Float64}
    T_p ::Vector{Float64}
    p   ::Vector{Float64}
    M   ::Vector{Float64}
    Re_p::Vector{Float64}
    Isp_acoplado   ::Float64
    Isp_equilibrio ::Float64
    Isp_congelado  ::Float64
    eta_2ph        ::Float64
    empuxo_N       ::Float64
    lag_vel        ::Float64
    lag_term       ::Float64
    tau_residencia ::Float64
    tau_v_saida    ::Float64
    tau_T_saida    ::Float64
    metricas       ::Dict{String,Any}
end

# ==============================================================================
# 3. LIMITES ANALÍTICOS
# ==============================================================================

"""
    _decompor_mistura(gas, part, alpha_ox) -> (cp_g, R_g, γ_g, cp_mix, R_mix, γ_mix)

Separa as propriedades do GÁS a partir das da MISTURA.

`gas.gamma` e `gas.R` são da mistura, condensados incluídos — é o que o CEA
entrega e o que a base de propelentes guarda. Verificado contra RocketCEA para
AP/HTPB/Al 70/14/16 a 7 MPa: `GAMMAs = 1.1400` e `M,(1/n) = 27.568`, que dá
`R = 301.6`, exactamente os valores do registo `APCP_HTPB_16Al`.

O `M,(1/n)` do CEA é massa TOTAL por mole de gás, portanto já traz a carga
condensada embutida. Uma versão anterior deste ficheiro fazia
`R_mix = (1-α_ox)*gas.R`, tratando a entrada como se fosse só do gás — dupla
contagem que baixava R de 301.6 para 210.5, isto é 43 %, e punha a velocidade
característica em 1406 m/s contra os 1580 do CEA. Agora inverte-se a relação:

    R_g  = R_mix/(1-α)          cp_g = (cp_mix - α*c_s)/(1-α)

o que devolve R_g de cerca de 432 J/(kg·K) contra os 422 que o CEA dá para o gás
sozinho.

Limite conhecido: o `GAMMAs` do CEA é a derivada isentrópica da mistura
reactiva, que não é exactamente cp/cv. Tomá-lo como γ de gás perfeito é a
aproximação de fundo deste módulo, e é o que sobra do défice face ao CEA depois
de corrigida a dupla contagem.
"""
function _decompor_mistura(gas, part::PropriedadesParticula, alpha_ox::Float64)
    γ_mix  = gas.gamma
    R_mix  = gas.R
    cp_mix = γ_mix * R_mix / (γ_mix - 1.0)
    alpha_ox <= 0.0 && return (cp_mix, R_mix, γ_mix, cp_mix, R_mix, γ_mix)

    R_g  = R_mix / (1.0 - alpha_ox)
    cp_g = (cp_mix - alpha_ox * part.cp_s) / (1.0 - alpha_ox)
    cp_g > R_g || error("_decompor_mistura: cp do gás ($(round(cp_g)) J/kg·K) não " *
                        "excede R ($(round(R_g))) — γ e R da mistura são " *
                        "incompatíveis com α_ox = $(round(alpha_ox, digits=3)).")
    γ_g  = cp_g / (cp_g - R_g)
    return (cp_g, R_g, γ_g, cp_mix, R_mix, γ_mix)
end

"""
    _razao_area(M, γ) -> A/A*

Relação isentrópica de área. É a inversa de `mach_from_area_ratio`; serve para
descobrir a área sónica que corresponde a um estado (M, A) já conhecido.
"""
@inline function _razao_area(M::Float64, γ::Float64)
    return (1.0/M) * ((2.0/(γ+1.0))*(1.0 + 0.5*(γ-1.0)*M^2))^((γ+1.0)/(2.0*(γ-1.0)))
end

"""
    isp_equilibrio(gas, part, alpha_ox, eps, P0, Pa) -> (Isp, u_e, T_e)

Limite de EQUILÍBRIO: partículas infinitamente pequenas, sempre à velocidade e
temperatura do gás. A mistura comporta-se como um gás único com

    cp_mix = (1−α)·cp_g + α·cp_s        R_mix = (1−α)·R_g
    γ_mix  = cp_mix/(cp_mix − R_mix)

As partículas contribuem com entalpia e massa mas não com pressão — é daí que
vem a queda de `R` efectivo, e é por isso que mesmo o equilíbrio perfeito tem
Isp menor que o do gás puro.
"""
function isp_equilibrio(gas, part::PropriedadesParticula, alpha_ox::Float64,
                        eps::Float64, P0::Float64, Pa::Float64)
    _, _, _, cp_mix, R_mix, γ_mix = _decompor_mistura(gas, part, alpha_ox)
    T0 = gas.T0 * gas.eta_cstar^2

    M_e  = mach_from_area_ratio(eps, γ_mix)
    fM   = 1.0 + 0.5 * (γ_mix - 1.0) * M_e^2
    T_e  = T0 / fM
    p_e  = P0 * fM^(-γ_mix / (γ_mix - 1.0))
    u_e  = M_e * sqrt(γ_mix * R_mix * T_e)

    # Calor latente da fase condensada. Em equilíbrio a partícula está sempre à
    # temperatura do gás, portanto solidifica assim que este cruza a fusão — o
    # mais cedo possível, e por isso o mais convertido possível em velocidade.
    # É o que mantém este limite como cota SUPERIOR do caso acoplado, onde a
    # partícula fica para trás e liberta o calor mais tarde e mais abaixo.
    # ── Solidificação da fase condensada ────────────────────────────────────
    #
    # Se a saída está abaixo da fusão, a partícula — colada ao gás — atravessou-a
    # e solidificou por completo. Isso muda o estado de saída inteiro, não é uma
    # parcela que se some no fim: duas tentativas anteriores erraram por aí.
    # Somar ΔH todo à energia cinética viola a segunda lei (inflava o limite em
    # ~4 %); aplicar-lhe um factor de Carnot a partir da fusão fica 0,6 % curto.
    #
    # O estado de saída fecha com quatro relações, sem aproximação nenhuma:
    #
    #   energia     u_e² = 2[cp_mix(T0 − T_e) + α·ΔH]
    #   entropia    cp_mix·ln(T_e/T0) − R_mix·ln(p_e/P0) − α·ΔH/T_fusão = 0
    #   estado      ρ_e = p_e/(R_mix·T_e)
    #   áreas       ρ_e·u_e = P0·Γ_mix/(√(R_mix·T0)·ε)
    #
    # O termo −α·ΔH/T_fusão é a queda de entropia da solidificação. Restam uma
    # incógnita (T_e) e uma equação, resolvida por bisseção — ρ_e·u_e é monótono
    # em T_e no ramo supersónico. A garganta não é afectada: lá o escoamento
    # ainda está acima da fusão (T* ≈ 2940 K contra 2327 K).
    if part.dH_fusao > 0.0 && T0 > part.T_fusao && T_e < part.T_fusao
        ΔH_esp = alpha_ox * part.dH_fusao          # por unidade de massa TOTAL
        Γ_mix  = sqrt(γ_mix) * (2.0/(γ_mix+1.0))^((γ_mix+1.0)/(2.0*(γ_mix-1.0)))
        alvo   = P0 * Γ_mix / (sqrt(R_mix * T0) * eps)   # ρ_e·u_e imposto

        u_de(T)  = sqrt(max(2.0*(cp_mix*(T0 - T) + ΔH_esp), 1.0))
        p_de(T)  = P0 * exp((cp_mix*log(T/T0) - ΔH_esp/part.T_fusao) / R_mix)
        ρu_de(T) = (p_de(T) / (R_mix*T)) * u_de(T)

        lo, hi = 0.02*T0, part.T_fusao
        if (ρu_de(lo) - alvo) * (ρu_de(hi) - alvo) < 0.0
            for _ in 1:80
                mid = 0.5*(lo + hi)
                (ρu_de(mid) - alvo) * (ρu_de(lo) - alvo) <= 0.0 ? (hi = mid) : (lo = mid)
            end
            T_e = 0.5*(lo + hi)
            u_e = u_de(T_e)
            p_e = p_de(T_e)
            M_e = u_e / sqrt(γ_mix * R_mix * T_e)
        end
    end
    return u_e / 9.80665, u_e, T_e, p_e, γ_mix, M_e
end

# ==============================================================================
# 4. INTEGRAÇÃO ACOPLADA
# ==============================================================================

"""
    escoamento_bifasico(gas, geo; frac_alumina, d43_um, P0, Pa, n_passos)
        -> ResultadoBifasico

Integra o escoamento acoplado gás–partícula no **divergente**, do ponto sónico
até à saída.

# Argumentos
- `gas` : `GasQuente`
- `geo` : `GeometriaTubeira` (de `tubeira_conica` ou `tubeira_de_simulacao`)
- `frac_alumina` : fracção mássica de Al no propelente [0–1]
- `d43_um` : diâmetro médio das partículas [µm]. 0 → predito por Hermsen.
- `P0` : pressão de estagnação [Pa]
- `Pa` : pressão ambiente [Pa]
- `n_passos` : subdivisões da integração RK4

# Por que só o divergente
No convergente as velocidades são baixas e o tempo de trânsito longo, e é na
expansão que o gás acelera rápido demais para as partículas acompanharem.

# ATENÇÃO — a condição inicial é OPTIMISTA, e não é a de Kliegel

Aqui arranca-se com `u_p = u_g` (equilíbrio). Isso NÃO é o que a fonte primária
faz: Quan & Kliegel (TRW/NASA NAS 9-4358, 1967) §2.2.4 usa *constant fractional
lag relationships* — resolvem a equação de arrasto na região da garganta
desprezando os termos derivados e obtêm uma defasagem FINITA já na linha
inicial, `u_p = K·u_g` com

    K²·τ_v·(du_g/dx) = 1 − K

Uma versão anterior deste comentário atribuía a hipótese de equilíbrio a
Kliegel–Nickerson. Era falso, e a leitura do relatório original desmentiu-o.

Consequência prática: como qualquer defasagem já presente à entrada só
aumentaria a perda, esta condição inicial faz o modelo **subestimar** η_2ph —
é optimista quanto ao Isp, não conservadora. É também a origem da fragilidade
com carga alta (ver `metricas["aviso_arranque"]`): assumir equilíbrio obriga a
escolher ONDE começar, e esse ponto afasta-se da garganta à medida que a carga
sobe. Uma condição inicial de defasagem fracionária removeria as duas coisas de
uma vez, e é a melhoria de maior prioridade neste ficheiro.
"""
function escoamento_bifasico(
    gas, geo;
    frac_alumina ::Float64 = 0.16,
    d43_um       ::Float64 = 0.0,
    P0           ::Float64 = 5.0e6,
    Pa           ::Float64 = 101325.0,
    n_passos     ::Int     = 400,
    particula    ::PropriedadesParticula = PropriedadesParticula(),
    ci_equilibrio::Bool    = false,
    # Mach do gás onde a integração arranca. 1.20 e não 1.02: junto ao sónico o
    # coeficiente de du_g/dx é quase nulo e a procura de quantidade de movimento
    # da partícula domina-o, o que faz a solução divergir de forma errática —
    # medido, com η a saltar para centenas em cerca de um terço de uma varredura
    # de (d, carga, tubeira). Com 1.20 a grelha fica limpa e monótona em toda a
    # faixa física. O preço é assumir equilíbrio um pouco mais adentro do
    # divergente, e está declarado em `metricas["razao_area_ini"]`.
    mach_arranque::Float64 = 1.20,
)
    # ── Fracção mássica de Al₂O₃ nos produtos ────────────────────────────────
    #
    # O oxidante do alumínio já está DENTRO do propelente (vem do perclorato),
    # portanto a combustão não importa massa de lado nenhum: 1 kg de propelente
    # dá 1 kg de produtos. Logo a massa de Al₂O₃ por kg de propelente é também a
    # sua fracção nos produtos, e α_ox = ξ_ox sem normalização nenhuma.
    #
    #     ξ_Al = 0.16  →  ξ_ox = 0.16 × 102/54 = 0.302  →  α_ox = 0.302
    #                     gás = 1 − 0.302 = 0.698       →  Y    = 0.433
    #
    # A forma anterior — α_ox = ξ_ox/(1+ξ_ox), Y = α/(1−α) — está deslocada um
    # nível na cadeia: dava α_ox = 0.232 e Y = 0.302, e o facto de esse Y sair
    # exactamente igual a ξ_ox é a assinatura do erro. Só valeria se ξ_ox fosse
    # definido por kg de GÁS, e não é. Subestimava a carga condensada em 23 %.
    ξ_Al  = clamp(frac_alumina, 0.0, 0.30)
    ξ_ox  = ξ_Al * (102.0 / 54.0)          # Al → Al₂O₃ (2Al + 3/2 O₂)
    α_ox  = ξ_ox                           # fracção mássica da fase condensada
    Y     = α_ox / (1.0 - α_ox)            # razão ṁ_p/ṁ_g

    T0    = gas.T0 * gas.eta_cstar^2
    A_t   = geo.A_t
    eps   = geo.A[end] / A_t

    # Diâmetro: dado, ou predito por Hermsen a partir da geometria e pressão
    d_p = if d43_um > 0.0
        d43_um * 1e-6
    else
        Dt = 2.0 * geo.R_t
        calcular_d43_hermsen(Dt, ξ_Al, P0, 2.0) * 1e-6
    end
    # Reconstrói só para fixar o d43; TODOS os outros campos têm de ser
    # repassados. Esquecer um faz o campo voltar ao padrão em silêncio — foi o
    # que aconteceu com T_fusao/dH_fusao quando foram acrescentados, e o efeito
    # foi um `dH_fusao=0` do chamador ser ignorado sem aviso nenhum.
    part = PropriedadesParticula(rho_s    = particula.rho_s,
                                 cp_s     = particula.cp_s,
                                 d43      = d_p,
                                 T_fusao  = particula.T_fusao,
                                 dH_fusao = particula.dH_fusao)

    # ── Vazão e condição sónica: da MISTURA, não do gás puro ─────────────────
    #
    # A hipótese declarada é que as partículas chegam à garganta em equilíbrio
    # (câmara e convergente dão tempo de sobra). Nesse regime o ponto sónico é o
    # da mistura, cuja velocidade do som é MENOR: as partículas acrescentam
    # massa e capacidade térmica mas não contribuem para a pressão.
    #
    #     cp_mix = (1−α)cp_g + α·c_s      R_mix = (1−α)R_g      γ_mix = cp/(cp−R)
    #
    # Usar γ e R do gás puro aqui era um erro: a vazão saía errada e o estado
    # inicial não era sónico para a mistura, o que impedia o limite d→0 de
    # recuperar o equilíbrio.
    #
    # A entrada `gas.gamma`/`gas.R` JÁ É da mistura (vem do CEA — ver
    # `_decompor_mistura`), portanto usa-se directamente aqui e derivam-se as
    # propriedades do gás por inversão, para as equações do gás.
    cp_g, R, γ, cp_mix, R_mix, γ_mixt = _decompor_mistura(gas, part, α_ox)

    Γ_mix  = sqrt(γ_mixt) * (2.0/(γ_mixt+1.0))^((γ_mixt+1.0)/(2.0*(γ_mixt-1.0)))
    mdot   = P0 * A_t * Γ_mix / sqrt(R_mix * T0)
    mdot_g = (1.0 - α_ox) * mdot
    mdot_p = α_ox * mdot

    # ── Energia total da mistura (constante) ────────────────────────────────
    # Partículas entram em equilíbrio térmico com o gás na câmara.
    # A entalpia da partícula inclui o calor latente enquanto ela estiver
    # líquida. `f_liq` acompanha a fracção líquida ao longo da tubeira; ao
    # solidificar, esse termo migra da partícula para o gás e o balanço fecha.
    f_liq_camara = (part.dH_fusao > 0.0 && T0 > part.T_fusao) ? 1.0 : 0.0
    E0 = mdot_g * cp_g * T0 +
         mdot_p * (part.cp_s * T0 + f_liq_camara * part.dH_fusao)

    # ── Estado inicial: sónico DA MISTURA, ligeiramente supersónico ──────────
    # Em M = 1 o coeficiente sónico anula-se — mas o numerador também, porque
    # dA/dx = 0 exactamente na garganta. Arrancar em x = x_t é pois um 0/0 mal
    # condicionado: o coeficiente é uma diferença quase cancelada de termos ~20×
    # maiores, e qualquer resíduo em dA/dx dá um du_g/dx enorme.
    #
    # A saída é arrancar um pouco a JUSANTE, na primeira estação com área
    # sensivelmente maior que a da garganta, e tirar M₀ da razão de áreas local
    # pela relação isentrópica da MISTURA (ramo supersónico). Assim a condição
    # inicial e a geometria são consistentes entre si, e dA/dx já é francamente
    # positivo quando a integração começa.
    idx_div = findall(xi -> xi > geo.x_t, geo.x)
    length(idx_div) >= 2 ||
        error("escoamento_bifasico: divergente com menos de 2 estações.")

    # Onde começar: no ponto sónico DO GÁS, não no da mistura.
    #
    # A mistura estrangula na garganta (A/A_t = 1), mas aí o gás ainda é
    # subsónico — a velocidade do som da mistura é menor, e M_gás = M_mist ·
    # √(γ_mist R_mist / γR) < M_mist. O ponto sónico do gás fica portanto A
    # JUSANTE da garganta, tanto mais quanto maior a carga de partículas.
    #
    # Isso importa porque, sobre um passo de integração infinitesimal, a
    # partícula não tem tempo de relaxar (Δt/τ_v → 0 ⇒ B_c → 0) e o coeficiente
    # de du_g/dx reduz-se ao do gás puro, (ṁ_g/γ)(1−1/M_gás²). A singularidade
    # que a integração vê localmente é a do gás. Arrancar numa razão de áreas
    # FIXA (1.02) funcionava até ~16 % de Al e depois punha o arranque no ramo
    # subsónico do gás, com η a saltar para 6 — o modelo devolvia lucro em vez
    # de perda.
    #
    # Preço da correcção: assume-se equilíbrio até um pouco mais adentro do
    # divergente. Para cargas altas o ponto sónico do gás afasta-se da garganta
    # e a hipótese enfraquece — daí o aviso em `metricas["aviso_arranque"]`.
    M_ARR = mach_arranque                          # margem sobre o sónico do gás

    # ── Estado inicial completo em função de x ──────────────────────────────
    #
    # Devolve o Mach do gás que sai da EQUAÇÃO DA ENERGIA, e não o da relação
    # isentrópica. A distinção é essencial: com a defasagem inicial de Kliegel a
    # mistura leva menos energia cinética, logo o gás fica mais quente e o seu
    # Mach é MENOR do que a relação isentrópica sugere. Procurar o arranque pelo
    # Mach isentrópico deixava a integração começar a M_gás = 0.967 — subsónica,
    # ramo errado — enquanto a guarda, que testava o mesmo valor isentrópico,
    # não dava por nada. O sintoma era η ≈ 3.6 e a velocidade de saída a colapsar
    # de 2290 para 120 m/s.
    function construir_ci(x)
        A   = _area_e_deriv(geo, x)[1]
        dA  = _area_e_deriv(geo, x)[2]
        ar  = max(A / A_t, 1.0 + 1e-9)
        M   = mach_from_area_ratio(ar, γ_mixt)
        fM  = 1.0 + 0.5*(γ_mixt - 1.0)*M^2
        Tgi = T0 / fM                              # isentrópico, ponto de partida
        ug  = M * sqrt(γ_mixt * R_mix * Tgi)

        dug_dx = ug * dA / (A * (M^2 - 1.0))
        dTg_dx = -(ug / cp_mix) * dug_dx
        ρg     = mdot_g / (ug * A)

        # Duas passagens: a segunda usa o T_g da energia para as propriedades
        # de transporte e para a defasagem térmica.
        Tg = Tgi; up = ug; Tp = Tgi; K = 1.0; fl = 0.0; sat = false
        for _ in 1:2
            μ, k = _mu_gas(Tg), _k_gas(Tg)
            Pr   = cp_g * μ / k
            K    = 1.0
            if !ci_equilibrio && α_ox > 0.0
                for _ in 1:4          # τ_v depende de Re, que depende de K
                    Re = ρg * abs(ug * (1.0 - K)) * part.d43 / max(μ, 1e-12)
                    τv = _tau_v(part.d43, part.rho_s, μ, Re)
                    a  = τv * dug_dx
                    K  = a > 1e-12 ? (-1.0 + sqrt(1.0 + 4.0*a)) / (2.0*a) : 1.0
                end
            end
            sat = K < K_MIN_LAG
            K   = max(K, K_MIN_LAG)
            up  = K * ug

            Re_p = ρg * abs(ug - up) * part.d43 / max(μ, 1e-12)
            τT   = _tau_T(part.d43, part.rho_s, part.cp_s, k, Re_p, Pr)
            Tp   = ci_equilibrio ? Tg : Tg - τT * up * dTg_dx
            fl   = (part.dH_fusao > 0.0 && Tp > part.T_fusao) ? 1.0 : 0.0

            # T_g pela ENERGIA, exactamente como `estado` fará no primeiro passo
            h_p = part.cp_s*Tp + fl*part.dH_fusao + 0.5*up^2
            Tg  = max((E0 - mdot_p*h_p)/(mdot_g*cp_g) - ug^2/(2.0*cp_g), 50.0)
        end

        return (A=A, M_mix=M, u_g=ug, T_g=Tg, u_p=up, T_p=Tp, f_liq=fl,
                K=K, K_sat=sat, M_gas=ug/sqrt(γ*R*Tg))
    end
    mach_gas_em(x) = construir_ci(x).M_gas

    i_limite = length(geo.x) - 1
    i_ini    = idx_div[1]
    while i_ini < i_limite && mach_gas_em(geo.x[i_ini]) < M_ARR
        i_ini += 1
    end

    # Bisseção dentro do segmento que faz a travessia. Sem isto, x_ini salta de
    # nó em nó da malha da geometria e η ganha degraus de ~1e-4 — inofensivos
    # num caso isolado, mas numa varredura aparecem como não-monotonia que não
    # existe na física.
    x_ini = geo.x[i_ini]
    if i_ini > idx_div[1] && mach_gas_em(geo.x[i_ini-1]) < M_ARR
        xa, xb = geo.x[i_ini-1], geo.x[i_ini]
        for _ in 1:60
            xm = 0.5*(xa + xb)
            mach_gas_em(xm) < M_ARR ? (xa = xm) : (xb = xm)
        end
        x_ini = xb
    end

    x_fim = geo.x[end]
    x_fim > x_ini ||
        error("escoamento_bifasico: sem divergente utilizável a jusante do " *
              "ponto sónico do gás (α_ox = $(round(α_ox, digits=3)) pode ser " *
              "alto demais para esta razão de expansão).")

    ci    = construir_ci(x_ini)
    A_ini = ci.A
    M0    = ci.M_mix
    T_g0  = ci.T_g          # da ENERGIA, coerente com o que `estado` calculará
    u_g0  = ci.u_g

    # ── Condição inicial: DEFASAGEM FRACCIONÁRIA (Quan & Kliegel §2.2.4) ─────
    #
    # A partícula NÃO chega à linha inicial em equilíbrio. Resolvendo a equação
    # de arrasto na região da garganta e desprezando os termos derivados (que lá
    # são pequenos face aos outros), procura-se a solução em que a razão
    # u_p/u_g ≡ K é estacionária:
    #
    #     u_p du_p/dx = (u_g − u_p)/τ_v ,  u_p = K u_g
    #  ⇒  K²·τ_v·(du_g/dx) = 1 − K       ⇒  K = (−1 + √(1+4a))/(2a),  a ≡ τ_v du_g/dx
    #
    # O gradiente vem da relação de área da mistura em equilíbrio,
    # du/u = dA/A /(M²−1), e o de temperatura da energia, dT/dx = −(u/cp)du/dx.
    # A defasagem térmica sai do mesmo argumento: T_p − T_g ≈ −τ_T u_p dT_g/dx,
    # positiva porque o gás arrefece e a partícula fica para trás.
    #
    # Assumir K = 1 (equilíbrio) subestima a perda — é optimista quanto ao Isp.
    # `ci_equilibrio = true` repõe essa hipótese, para comparação.
    K_lag     = ci.K
    K_saturou = ci.K_sat
    u_p0      = ci.u_p
    T_p0      = ci.T_p

    # Guarda dura: a integração percorre o ramo SUPERSÓNICO do gás. Arrancar com
    # M_gás ≤ 1 põe-na no ramo errado e o resultado sai sem sentido (η > 1) em
    # vez de sair com erro. Prefere-se falhar alto — a alternativa já aconteceu
    # três vezes, e um η de 3.6 num relatório é pior que uma excepção.
    #
    # O Mach testado é o da ENERGIA (ver `construir_ci`). Testar o isentrópico,
    # como uma versão anterior fazia, deixava passar arranques a M = 0.967.
    M_g0 = ci.M_gas
    M_g0 > 1.0 ||
        error("escoamento_bifasico: o gás ainda é subsónico (M_gás = " *
              "$(round(M_g0, digits=4))) onde a integração arrancaria. " *
              "Carga de partículas α_ox = $(round(α_ox, digits=3)) alta demais " *
              "para a razão de expansão desta tubeira ($(round(eps, digits=2))).")
    razao_ini = A_ini / A_t

    # Malha AGRUPADA junto à garganta: xs = x_ini + L·s², s uniforme.
    #
    # Não é cosmética. Quando as partículas desacoplam (d grande) tem-se B_c→0
    # e o coeficiente de du_g/dx reduz-se a (ṁ_g/γ)(1−1/M_gás²) — o gás passa a
    # ter a sua PRÓPRIA singularidade sónica, logo a seguir à garganta da
    # mistura, onde M_gás ≈ 1.004. Aí o coeficiente vale ~0.16 contra termos
    # individuais de ~19: du_g/dx é enorme e hipersensível. Com malha uniforme
    # o primeiro passo atravessa essa zona de uma vez e o erro (~6 % em u_g)
    # propaga-se até à saída, dando η > 1 — acima do limite de equilíbrio.
    L_div_int = x_fim - x_ini
    xs = [x_ini + L_div_int * (i / n_passos)^2 for i in 0:n_passos]
    Ax = similar(xs); dA = similar(xs)
    @inbounds for i in eachindex(xs)
        Ax[i], dA[i] = _area_e_deriv(geo, xs[i])
    end

    # ── Integração exponencial (semi-implícita) ──────────────────────────────
    #
    # As equações da partícula são RELAXAÇÕES LINEARES para o estado do gás, com
    # constantes τ_v e τ_T ∝ d². Para partículas pequenas τ_v cai a 1e-13 s
    # enquanto o passo cobre ~1e-6 s: um integrador explícito nesse regime é
    # incondicionalmente instável, e um RK4 de passo fixo explode (verificado:
    # Isp da ordem de 1e80 a 0.01 µm).
    #
    # Diminuir o passo não serve — exigiria 1e7 passos por milímetro. A saída é
    # integrar a relaxação ANALITICAMENTE sobre o passo, que é exacto quando o
    # estado do gás é constante nele:
    #
    #     u_p(t+Δt) = u_g + (u_p − u_g)·exp(−Δt/τ_v)
    #
    # Isto é estável para qualquer d e recupera exactamente os dois limites:
    # Δt/τ_v → ∞ dá u_p = u_g (equilíbrio); → 0 dá u_p inalterado (congelado).
    # O gás não é rígido, e é avançado por Heun com a derivada EFECTIVA da
    # partícula no passo — o que mantém a consistência do acoplamento.
    n   = length(xs)
    u_g = zeros(n); u_p = zeros(n); T_p = zeros(n)
    T_g = zeros(n); pv  = zeros(n); Mv  = zeros(n); Rev = zeros(n)
    fliq = zeros(n)
    u_g[1], u_p[1], T_p[1] = u_g0, u_p0, T_p0
    fliq[1] = ci.f_liq      # o mesmo que entrou no balanço de energia da CI

    τv_s = 0.0; τT_s = 0.0; t_res = 0.0

    # Estado do gás e tempos de relaxação num ponto (sem derivadas).
    # `f_liq_` entra na energia: o calor latente ainda retido pela partícula é
    # entalpia que não está no gás.
    function estado(u_g_, u_p_, T_p_, f_liq_, A)
        u_g_ = max(u_g_, 1.0); u_p_ = max(u_p_, 1.0)
        h_p = part.cp_s*T_p_ + f_liq_*part.dH_fusao + 0.5*u_p_^2
        Tg  = (E0 - mdot_p*h_p)/(mdot_g*cp_g) - u_g_^2/(2.0*cp_g)
        Tg = max(Tg, 50.0)
        ρ_g = mdot_g / (u_g_ * A)
        p   = ρ_g * R * Tg
        M   = u_g_ / sqrt(γ * R * Tg)
        μ   = _mu_gas(Tg); k = _k_gas(Tg)
        Pr  = cp_g * μ / k
        Re  = ρ_g * abs(u_g_ - u_p_) * part.d43 / max(μ, 1e-12)
        return Tg, p, M, Re,
               _tau_v(part.d43, part.rho_s, μ, Re),
               _tau_T(part.d43, part.rho_s, part.cp_s, k, Re, Pr)
    end

    # du_g/dx com a partícula tratada IMPLICITAMENTE.
    #
    # `du_p/dx` não é independente de `du_g/dx`: quanto mais acoplada a
    # partícula, mais ela segue o gás. Resolvendo a relaxação exponencial sobre
    # o passo com alvo linear obtém-se a forma AFIM
    #
    #     du_p/dx = A_c + B_c·G ,   A_c = (u_g−u_p)(1−E)/h
    #                               B_c = 1 − τ_v·u_p·(1−E)/h
    #
    # cujos limites são os certos: acoplado (E→0) dá B_c→1, ou seja du_p/dx =
    # du_g/dx; congelado (E→1) dá B_c→0.
    #
    # Passar B_c·G para o lado esquerdo é o que faz o coeficiente efectivo
    # anular-se no ponto sónico da MISTURA, e não no do gás puro. Tratando
    # du_p/dx como explícito, o coeficiente era (ṁ_g/γ)(1−1/M_gás²), que é
    # NEGATIVO no arranque — o gás está a M≈0.88 quando a mistura está a M=1 —
    # e a integração disparava.
    # `dfl` = df_liq/dx, a taxa de solidificação. Entra aqui porque dT_g/dx sai
    # da equação da ENERGIA, e essa tem o termo ΔH·df_liq/dx assim que há
    # mudança de fase:
    #
    #   dT_g/dx = −(Y/cp_g)(c_s dT_p/dx + ΔH df_liq/dx + u_p du_p/dx) − (u_g/cp_g)G
    #
    # Pôr o calor latente só na energia e esquecê-lo aqui deixa a quantidade de
    # movimento a ver um gradiente de pressão que não corresponde ao campo de
    # temperatura — e o resultado foi η > 1 no limite d→0, energia a aparecer do
    # nada. As duas equações têm de ver a mesma libertação de calor.
    function dug_dx(u_g_, u_p_, Tg, p, M, A_c, B_c, dTp, dfl, A, dAdx)
        k_th = A*p*Y/(Tg*cp_g)
        coef = (mdot_g/γ)*(1.0 - 1.0/M^2) + mdot_p*B_c - k_th*u_p_*B_c
        rhs  = -mdot_p*A_c +
               k_th*(part.cp_s*dTp + part.dH_fusao*dfl + u_p_*A_c) +
               p*dAdx
        abs(coef) < 1e-12 && (coef = (coef >= 0 ? 1.0 : -1.0) * 1e-12)
        return rhs / coef
    end

    # Relaxação exponencial para um alvo que VARIA LINEARMENTE no passo.
    #
    #   dy/dt = (Y(t) − y)/τ,  Y(t) = Y0 + Ẏ·t
    #   ⇒ y(Δt) = Y_fim − τẎ + (y0 − Y0 + τẎ)·e^(−Δt/τ)
    #
    # O termo −τẎ é essencial: no limite rígido (E→0) dá y = Y_fim − τẎ, ou
    # seja, a partícula segue o gás com um atraso proporcional a τ. Relaxar para
    # um alvo CONGELADO no valor inicial daria y → Y0 e, com T_p = T_g, derivada
    # zero — o oposto do comportamento correcto.
    #
    # Serve a relaxação TÉRMICA. A de velocidade é tratada implicitamente pela
    # forma afim A_c/B_c em `dug_dx`, porque só assim o coeficiente sónico sai
    # certo; a térmica não entra nesse coeficiente e pode ficar explícita.
    @inline function _relax(y0, Y0, Y_fim, τ, Δt)
        Ẏ = (Y_fim - Y0) / max(Δt, 1e-300)
        E = exp(-min(Δt/τ, 700.0))
        return Y_fim - τ*Ẏ + (y0 - Y0 + τ*Ẏ)*E
    end

    @inbounds for i in 1:n-1
        h = xs[i+1] - xs[i]

        Tg1, p1, M1, Re1, τv1, τT1 = estado(u_g[i], u_p[i], T_p[i], fliq[i], Ax[i])
        T_g[i], pv[i], Mv[i], Rev[i] = Tg1, p1, M1, Re1
        τv_s, τT_s = τv1, τT1

        # Iteração de ponto fixo: só para os coeficientes (Δt depende de u_p, e o
        # estado a jusante depende de u_g). O ACOPLAMENTO em si já é implícito
        # via A_c/B_c, por isso três passagens bastam com folga.
        ug_fim = u_g[i]; up_fim = u_p[i]; Tp_fim = T_p[i]
        fliq_fim = fliq[i]
        Tg2 = Tg1; p2 = p1; M2 = M1
        # τ avaliado no PONTO MÉDIO do passo, não à esquerda. Com τ congelado à
        # esquerda o esquema cai a 1ª ordem e as malhas grosseiras dão η > 1 —
        # acima do limite de equilíbrio, que é fisicamente impossível. A
        # iteração de ponto fixo já visita o estado a jusante, portanto a média
        # sai de graça.
        τvm = τv1; τTm = τT1
        for _ in 1:3
            Δt  = h / max(0.5*(u_p[i] + up_fim), 1.0)
            Ev  = exp(-min(Δt/τvm, 700.0))
            upm = max(0.5*(u_p[i] + up_fim), 1.0)

            # Forma afim du_p/dx = A_c + B_c·G (ver `dug_dx`)
            A_c = (u_g[i] - u_p[i]) * (1.0 - Ev) / h
            B_c = 1.0 - τvm * upm * (1.0 - Ev) / h

            # A relaxação térmica fica explícita: entra na quantidade de
            # movimento só pelo termo de troca de calor, uma ordem de grandeza
            # abaixo do de arrasto, e não altera a condição sónica.
            # A solidificação entra AQUI, sobre a temperatura que a relaxação
            # daria: enquanto houver líquido a partícula fica presa na fusão e o
            # calor latente passa para o gás.
            Tp_rel = _relax(T_p[i], Tg1, Tg2, τTm, Δt)
            Tp_fim, fliq_fim = _avancar_fase(T_p[i], fliq[i], Tp_rel, part)
            dTp    = (Tp_fim - T_p[i]) / h
            dfl    = (fliq_fim - fliq[i]) / h

            g1 = dug_dx(u_g[i], u_p[i], Tg1, p1, M1, A_c, B_c, dTp, dfl,
                        Ax[i], dA[i])
            ug_pred = u_g[i] + h*g1
            up_pred = u_p[i] + h*(A_c + B_c*g1)
            Tg2, p2, M2, _, τv2, τT2 = estado(ug_pred, up_pred, Tp_fim, fliq_fim,
                                              Ax[i+1])
            g2 = dug_dx(ug_pred, up_pred, Tg2, p2, M2, A_c, B_c, dTp, dfl,
                        Ax[i+1], dA[i+1])

            G      = 0.5*(g1 + g2)
            ug_fim = u_g[i] + h*G
            up_fim = u_p[i] + h*(A_c + B_c*G)
            τvm    = 0.5*(τv1 + τv2)
            τTm    = 0.5*(τT1 + τT2)
        end

        u_g[i+1] = ug_fim
        u_p[i+1] = up_fim
        T_p[i+1] = Tp_fim
        fliq[i+1] = fliq_fim
        t_res += h / max(0.5*(u_p[i] + u_p[i+1]), 1.0)
    end
    Tgf, pf, Mf, Ref, τvf, τTf = estado(u_g[n], u_p[n], T_p[n], fliq[n], Ax[n])
    T_g[n], pv[n], Mv[n], Rev[n] = Tgf, pf, Mf, Ref
    τv_s, τT_s = τvf, τTf

    # ── Empuxo e Isp ─────────────────────────────────────────────────────────
    A_e = Ax[end]
    F_acoplado = mdot_g*u_g[n] + mdot_p*u_p[n] + (pv[n] - Pa)*A_e
    Isp_ac = F_acoplado / (mdot * 9.80665)

    Isp_eq, u_eq, T_eq, p_eq, γ_mix, M_eq = isp_equilibrio(gas, part, α_ox, eps, P0, Pa)
    F_eq   = mdot*u_eq + (p_eq - Pa)*A_e
    Isp_eq_F = F_eq / (mdot * 9.80665)

    # ── Limite CONGELADO, calculado à parte ─────────────────────────────────
    # Deliberadamente NÃO usa a solução acoplada: um limite que dependesse dela
    # não serviria para a validar. Aqui o gás expande sozinho, com γ e R do gás
    # puro, a partir do seu estado no arranque; as partículas ficam com a
    # velocidade e a temperatura que tinham na garganta.
    M_g0    = u_g0 / sqrt(γ * R * T_g0)
    A_est_g = Ax[1] / _razao_area(M_g0, γ)          # área sónica do gás sozinho
    M_e_fr  = mach_from_area_ratio(max(A_e / A_est_g, 1.0 + 1e-6), γ)
    fM_fr   = 1.0 + 0.5*(γ - 1.0)*M_e_fr^2
    T0_g_ef = T_g0 * (1.0 + 0.5*(γ - 1.0)*M_g0^2)
    p0_g_ef = pv[1] * (1.0 + 0.5*(γ - 1.0)*M_g0^2)^(γ/(γ - 1.0))
    T_e_fr  = T0_g_ef / fM_fr
    u_e_fr  = M_e_fr * sqrt(γ * R * T_e_fr)
    p_e_fr  = p0_g_ef * fM_fr^(-γ/(γ - 1.0))
    F_fr    = mdot_g*u_e_fr + mdot_p*u_p0 + (p_e_fr - Pa)*A_e
    Isp_fr  = F_fr / (mdot * 9.80665)

    η_2ph = Isp_ac / max(Isp_eq_F, 1e-9)

    # ── Validade física da solução ──────────────────────────────────────────
    #
    # Num divergente o gás só acelera, o escoamento é supersónico em todo o lado
    # e a defasagem só pode custar impulso — η ≤ 1. Quando qualquer destas
    # falha, a integração saiu do ramo supersónico e o resultado não tem
    # significado: já se viram η de 400.
    #
    # Falha alto em vez de devolver o número. Ao longo do desenvolvimento este
    # modo de avaria apareceu quatro vezes por causas diferentes, e das quatro
    # nenhuma era visível a olho no valor devolvido. Os chamadores (CaseRunner)
    # apanham a excepção e caem no modelo escalar com aviso.
    if !(η_2ph <= 1.0 + 1e-3) || !issorted(u_g) || minimum(Mv) <= 1.0
        motivo = !(η_2ph <= 1.0 + 1e-3) ? "η = $(round(η_2ph, digits=3)) > 1" :
                 !issorted(u_g)         ? "o gás desacelera no divergente" :
                                          "o escoamento fica subsónico (M_min = $(round(minimum(Mv), digits=3)))"
        error("escoamento_bifasico: solução não-física — $motivo.\n" *
              "  d43 = $(round(part.d43*1e6, digits=2)) µm, α_ox = " *
              "$(round(α_ox, digits=3)), ε = $(round(eps, digits=2)), " *
              "M_gás no arranque = $(round(M_g0, digits=3)).\n" *
              "  A condição inicial de defasagem fraccionária perde robustez " *
              "para partículas grandes (acima de ~10 µm): o défice inicial de " *
              "velocidade é grande e fechá-lo sobrecarrega a quantidade de " *
              "movimento do gás junto ao sónico. Use `ci_equilibrio = true` " *
              "(mais optimista, mas robusto em toda a faixa) ou um d43 menor.")
    end

    met = Dict{String,Any}(
        "d43_um"         => part.d43 * 1e6,
        "alpha_ox"       => α_ox,
        "xi_ox"          => ξ_ox,
        "Y_massa"        => Y,
        "mdot_total"     => mdot,
        "mdot_gas"       => mdot_g,
        "mdot_part"      => mdot_p,
        "u_g_saida"      => u_g[n],
        "u_p_saida"      => u_p[n],
        "T_g_saida"      => T_g[n],
        "T_p_saida"      => T_p[n],
        "p_saida_Pa"     => pv[n],
        "M_saida"        => Mv[n],
        "Re_p_saida"     => Rev[n],
        "gamma_mix_eq"   => γ_mix,
        # Propriedades separadas por fase. As de MISTURA são as que entram
        # (vêm do CEA); as do GÁS saem por inversão em `_decompor_mistura`.
        "cp_gas"         => cp_g,
        "R_gas"          => R,
        "gamma_gas"      => γ,
        "cp_mistura"     => cp_mix,
        "R_mistura"      => R_mix,
        "gamma_mistura"  => γ_mixt,
        "razao_expansao" => eps,
        "n_passos"       => n_passos,
        # Onde a integração arrancou, em razão de áreas. É a medida de quanto se
        # está a assumir equilíbrio para dentro do divergente: 1.0 seria a
        # garganta (hipótese mínima); valores altos vêm de carga alta e
        # enfraquecem a hipótese.
        "razao_area_ini" => razao_ini,
        "M_gas_ini"      => M_g0,
        "K_lag_inicial"  => K_lag,      # u_p/u_g na linha inicial (1 = equilíbrio)
        "K_lag_saturado" => K_saturou,  # relação de Kliegel fora de validade
        "ci_equilibrio"  => ci_equilibrio,
        # Mudança de fase da Al₂O₃
        "f_liq_saida"    => fliq[n],            # 0 = solidificou toda na tubeira
        "f_liq_inicial"  => fliq[1],
        "T_fusao"        => part.T_fusao,
        "dH_fusao"       => part.dH_fusao,
        # Limiar 1.45 não é arbitrário: medido. Varrendo a carga de 2 % a 30 %
        # de Al, η decresce monotonicamente com α_ox até A/A_t ≈ 1.59 e a partir
        # daí INVERTE — a hipótese de equilíbrio até ao arranque passa a devolver
        # mais do que a carga extra custa. 1.45 deixa margem antes disso.
        #
        # O limiar depende de `mach_arranque`: com o valor antigo de 1.02 a
        # inversão dava-se a A/A_t ≈ 1.20 e o limiar era 1.15. Se mexer num,
        # remeça o outro — a varredura está em `t_recal.jl`.
        #
        # Para APCP na faixa documentada (ξ_Al ≤ 0.22) chega-se no máximo a
        # A/A_t = 1.32, ou seja, o uso normal fica dentro da zona de confiança.
        "aviso_arranque" => razao_ini > 1.45 ?
            "equilíbrio assumido até A/A_t = $(round(razao_ini, digits=2)); " *
            "carga de partículas alta (α_ox = $(round(α_ox, digits=3))) empurra " *
            "o ponto sónico do gás para jusante e a perda calculada perde " *
            "confiança — acima de A/A_t ≈ 1.59 ela chega a decrescer com a carga" : "",
    )

    return ResultadoBifasico(xs, Ax, u_g, u_p, T_g, T_p, pv, Mv, Rev,
        Isp_ac, Isp_eq_F, Isp_fr, η_2ph, F_acoplado,
        (u_g[n] - u_p[n]) / max(u_g[n], 1e-9),
        (T_p[n] - T_g[n]) / max(T_g[n], 1e-9),
        t_res, τv_s, τT_s, met)
end

"""
    _area_e_deriv(geo, x) -> (A, dA/dx)

Área e a sua derivada, obtidas interpolando o RAIO linearmente:

    R(x) = R_j + t·(R_{j+1} − R_j)      A = πR²      dA/dx = 2πR·dR/dx

Interpolar o raio (e não a área) é exacto num troço cónico, que é o caso de
todas as tubeiras aqui. Mais importante: `dA/dx` sai analiticamente do segmento
em vez de por diferenças finitas sobre uma área já interpolada — isso produzia
um dente-de-serra nos nós da malha da geometria, que a integração amplificava e
que estragava a ordem de convergência.
"""
function _area_e_deriv(geo, x::Float64)
    xs, Rs = geo.x, geo.R
    n = length(xs)
    j = clamp(searchsortedlast(xs, x), 1, n-1)
    hj = xs[j+1] - xs[j]
    t  = hj > 0 ? clamp((x - xs[j]) / hj, 0.0, 1.0) : 0.0
    dR = hj > 0 ? (Rs[j+1] - Rs[j]) / hj : 0.0
    R  = Rs[j] + t * (Rs[j+1] - Rs[j])
    return π*R^2, 2π*R*dR
end

# ==============================================================================
# 5. PONTE COM O SIMULADOR
# ==============================================================================

"""
    eta_2ph_acoplado(inp, D43_um, P0_Pa; Pa, n_passos) -> (η, ResultadoBifasico)

Calcula η_2ph para um `CaseInput` já resolvido, integrando o escoamento
bifásico ACOPLADO na tubeira real do caso. Substitui o modelo escalar
`calcular_2fases_fisica` quando `ConfigModelo.usar_2fases_acoplado = true`.

Substitui apenas o ÚLTIMO passo da cadeia — o modelo de Stokes escalar. O
diâmetro `D43_um` continua a vir de onde vinha (dado pelo utilizador ou
predito por Hermsen a partir da câmara), porque D₄₃ é propriedade da
combustão, não da tubeira. O que muda é como a perda é obtida a partir dele:
antes por correlação calibrada, agora por integração da geometria.

Só o divergente é integrado, portanto o convergente da geometria construída
aqui é irrelevante para o resultado — serve apenas para localizar a garganta.
"""
function eta_2ph_acoplado(inp, D43_um::Real, P0_Pa::Real;
                          Pa::Real = 101325.0, n_passos::Int = 400)
    inp.D_saida > inp.D_garganta_ini ||
        error("eta_2ph_acoplado: D_saida ($(inp.D_saida) m) deve ser > D_garganta_ini " *
              "($(inp.D_garganta_ini) m) — sem divergente não há o que integrar.")
    D43_um > 0.0 ||
        error("eta_2ph_acoplado: D43_um deve ser > 0 (predito por Hermsen a montante).")

    # Geometria: só o divergente é integrado; o convergente serve p/ localizar a
    # garganta. Reusa a tubeira cónica do Produto_comercial (mesma GeometriaTubeira,
    # campos x/R/A/A_t/R_t que o escoamento_bifasico lê). O comprimento do divergente
    # vem do ângulo real (alpha_divergencia) → residência física correta.
    geo = tubeira_conica(D_entrada     = max(inp.D_ext, 1.5 * inp.D_garganta_ini),
                         D_garganta    = inp.D_garganta_ini,
                         D_saida       = inp.D_saida,
                         alpha_div_deg = inp.alpha_divergencia,
                         N_x           = 201)

    gas = GasQuente(T0 = inp.Tc, gamma = inp.gamma, R = inp.R, eta_cstar = 1.0)

    r = escoamento_bifasico(gas, geo;
                            frac_alumina = inp.frac_alumina,
                            d43_um       = Float64(D43_um),
                            P0           = Float64(P0_Pa),
                            Pa           = Float64(Pa),
                            n_passos     = n_passos)
    return r.eta_2ph, r
end

# ── Distribuição de tamanho de partícula (log-normal, NASA SP-8039 §2.1.3.2.1.2) ──
# CDF normal padrão via erf (Abramowitz–Stegun 7.1.26, |erro|<1.5e-7).
@inline function _Phi_std(z::Float64)
    x = z / sqrt(2.0); s = sign(x); x = abs(x)
    t = 1.0 / (1.0 + 0.3275911*x)
    y = 1.0 - (((((1.061405429*t - 1.453152027)*t) + 1.421413741)*t
               - 0.284496736)*t + 0.254829592)*t*exp(-x*x)
    return 0.5*(1.0 + s*y)
end

"""
    _classes_lognormal(d43, sigma_g, N) -> Vector{Tuple{Float64,Float64}}

Discretiza uma distribuição LOG-NORMAL de MASSA (mass-mean = `d43`, desvio-padrão
geométrico `sigma_g`) em `N` classes `(d_i, w_i)`, Σw_i=1, com o d_i reescalado p/ o
mass-mean discreto bater exatamente `d43`. `sigma_g ≤ 1` ⇒ 1 classe (recupera d43).
"""
function _classes_lognormal(d43::Float64, sigma_g::Float64, N::Int)
    (sigma_g <= 1.0 + 1e-6 || N <= 1) && return [(d43, 1.0)]
    σL = log(sigma_g)
    μL = log(d43) - 0.5*σL^2                     # mass-mean = exp(μL+σL²/2) = d43
    zed = range(-3.0, 3.0, length = N+1)
    cls = Tuple{Float64,Float64}[]
    for i in 1:N
        w  = _Phi_std(zed[i+1]) - _Phi_std(zed[i])
        di = exp(μL + σL*0.5*(zed[i] + zed[i+1]))
        push!(cls, (di, w))
    end
    sw = sum(w for (_, w) in cls)
    cls = [(d, w/sw) for (d, w) in cls]          # renormaliza Σw=1
    md = sum(d*w for (d, w) in cls)              # reescala d p/ mass-mean = d43 exato
    return [(d*d43/md, w) for (d, w) in cls]
end

"""
    eta_2ph_distribuicao(inp, D43_um, P0_Pa; sigma_g=1.8, N_classes=7, Pa, n_passos)
        -> (η, info)

η_2ph com DISTRIBUIÇÃO log-normal de tamanho de partícula (NASA SP-8039), em vez de
um d43 único: discretiza a distribuição (mass-mean = `D43_um` [µm]), integra o
escoamento bifásico acoplado POR classe e faz a média MÁSSICA da perda. A cauda de
partículas grandes lag mais → η ligeiramente MENOR (mais perda) que o monodisperso —
o efeito que este item captura. `sigma_g=1` recupera `eta_2ph_acoplado`.
`info`: `eta_mono` (η do d43 único), `classes` (vetor (d_i[µm], w_i, η_i)), `sigma_g`.
"""
function eta_2ph_distribuicao(inp, D43_um::Real, P0_Pa::Real;
                              sigma_g::Real = 1.8, N_classes::Int = 7,
                              Pa::Real = 101325.0, n_passos::Int = 400)
    cls = _classes_lognormal(Float64(D43_um), Float64(sigma_g), N_classes)
    η_acc = 0.0; det = Tuple{Float64,Float64,Float64}[]
    for (di, wi) in cls
        ηi, _ = eta_2ph_acoplado(inp, di, P0_Pa; Pa = Pa, n_passos = n_passos)
        η_acc += wi*ηi
        push!(det, (di, wi, ηi))
    end
    η_mono, _ = eta_2ph_acoplado(inp, Float64(D43_um), P0_Pa; Pa = Pa, n_passos = n_passos)
    return η_acc, (eta_mono = η_mono, classes = det, sigma_g = Float64(sigma_g))
end

# ==============================================================================
# 6. RELATÓRIO
# ==============================================================================

"""
    imprimir_bifasico(res::ResultadoBifasico)

Relatório de consola, enquadrando o resultado entre os limites analíticos.
"""
function imprimir_bifasico(res::ResultadoBifasico)
    m = res.metricas
    println("\n" * "="^66)
    println("     ESCOAMENTO BIFÁSICO ACOPLADO — GÁS + Al₂O₃")
    println("="^66)
    @printf("  d43 = %.2f µm    α_Al₂O₃ = %.1f %% da massa    ε = %.2f\n",
            m["d43_um"], 100*m["alpha_ox"], m["razao_expansao"])
    @printf("  ṁ = %.4f kg/s  (gás %.4f + partículas %.4f)\n",
            m["mdot_total"], m["mdot_gas"], m["mdot_part"])
    println("-"^66)
    println("  SAÍDA")
    @printf("  %-24s %10.1f m/s\n", "velocidade do gás:",       m["u_g_saida"])
    @printf("  %-24s %10.1f m/s\n", "velocidade da partícula:", m["u_p_saida"])
    @printf("  %-24s %10.1f %%\n",  "defasagem de velocidade:", 100*res.lag_vel)
    @printf("  %-24s %10.1f K\n",   "T do gás:",                m["T_g_saida"])
    @printf("  %-24s %10.1f K\n",   "T da partícula:",          m["T_p_saida"])
    @printf("  %-24s %10.1f K\n",   "defasagem térmica:",
            m["T_p_saida"] - m["T_g_saida"])
    @printf("  %-24s %10.4f\n",     "Re_p na saída:",           m["Re_p_saida"])
    println("-"^66)
    println("  TEMPOS")
    @printf("  %-24s %10.3e s   ← CALCULADO, não calibrado\n",
            "residência real:", res.tau_residencia)
    @printf("  %-24s %10.3e s\n", "τ_v na saída:", res.tau_v_saida)
    @printf("  %-24s %10.3e s\n", "τ_T na saída:", res.tau_T_saida)
    @printf("  %-24s %10.3f\n",   "Stokes (τ_v/τ_res):",
            res.tau_v_saida / max(res.tau_residencia, 1e-12))
    println("-"^66)
    println("  IMPULSO ESPECÍFICO")
    @printf("  %-24s %10.2f s   (limite: partículas coladas)\n",
            "equilíbrio:", res.Isp_equilibrio)
    @printf("  %-24s %10.2f s   ← ACOPLADO\n", "calculado:", res.Isp_acoplado)
    @printf("  %-24s %10.2f s   (limite: sem acelerar)\n",
            "congelado:", res.Isp_congelado)
    @printf("  %-24s %10.4f      (perda de %.2f %%)\n",
            "η_2ph:", res.eta_2ph, 100*(1 - res.eta_2ph))
    dentro = res.Isp_congelado <= res.Isp_acoplado <= res.Isp_equilibrio * 1.0001
    @printf("  %-24s %10s\n", "entre os limites:", dentro ? "sim" : "NÃO — verificar")
    println("="^66)
    return nothing
end
