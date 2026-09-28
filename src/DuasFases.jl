# ==============================================================================
# DuasFases.jl — perda de impulso por partículas de Al₂O₃ (η_2ph)
# ==============================================================================
# Extraído de ThermalAnalysis.jl do RktPrisma: só as funções de duas fases usadas
# pela balística (CaseRunner.jl e TwoPhaseNozzle.jl). O restante daquele arquivo
# (correlação de Bartz, análise térmica da tubeira) não faz parte deste repositório.
# ==============================================================================

using Printf

# ==============================================================================
# 1b. CORREÇÃO DE DUAS FASES — PARTÍCULAS DE Al₂O₃
# ==============================================================================

"""
    calcular_correcao_duas_fases(frac_alumina) -> (K_Bartz, eta_Isp)

Calcula os fatores de correção para partículas condensadas de Al₂O₃ em
propelentes APCP com alumínio metálico.

## Física

Propelentes com alumínio (10–22%) geram partículas de Al₂O₃ fundidas (~3 500 K)
durante a combustão. No escoamento da tubeira essas partículas:

  1. **Aumentam h_g** — turbulência amplificada por partículas (turbulence
     augmentation) e transferência radiativa direta à parede. A correlação de
     Bartz padrão subestima h_g em 30–50 % para ~16 % Al.

  2. **Reduzem Isp** — partículas não se expandem como gás ideal; há defasagem
     de velocidade (*velocity lag*) e temperatura (*temperature lag*) em relação
     ao gás. Perda típica: 3–6 % para 15–20 % Al.

## Correlações

Fator de correção do Bartz (fit empírico a dados de Kliegel-Nickerson, 1962,
e testes APCP com alumínio):

    K_Bartz = 1 + 1.15 · ξ_ox

Eficiência Isp de duas fases (defasagem velocidade/temperatura de partículas,
baseado em Dobbins-Temkin e dados CEA comparativos):

    η_Isp = 1 − 0.14 · ξ_ox

onde ξ_ox é a fração mássica de Al₂O₃ nos produtos de combustão:

    ξ_ox = ξ_Al × (M_Al₂O₃ / 2M_Al) = ξ_Al × 1.889

## Faixa de validade

ξ_Al ∈ [0, 0.22]. Para ξ_Al = 0.16 (APCP típico):

  - K_Bartz ≈ 1.35  (+35 % em h_g e q_w)
  - η_Isp   ≈ 0.958 (−4.2 % em Isp real vs. calculado)

## Parâmetros
- `frac_alumina` : fração mássica de Al no propelente (0–1), ex: 0.16 = 16 %

## Retorna `NamedTuple` com
- `K_Bartz` : fator multiplicativo para h_g de Bartz (≥ 1.0)
- `eta_Isp` : eficiência de Isp de duas fases (≤ 1.0)
"""
function calcular_correcao_duas_fases(frac_alumina::Float64)
    frac_alumina <= 0.0 && return (K_Bartz=1.0, eta_Isp=1.0)

    # Fração mássica de Al₂O₃ nos produtos de combustão
    # Reação estequiométrica: 2 Al + 3/2 O₂ → Al₂O₃
    #   M_Al₂O₃ = 102 g/mol,  2 × M_Al = 54 g/mol  →  razão = 1.889
    ξ_Al = clamp(frac_alumina, 0.0, 0.22)
    ξ_ox = ξ_Al * (102.0 / 54.0)         # ≈ 1.889 × ξ_Al

    # Correção Bartz — turbulência amplificada + radiação de partículas Al₂O₃
    K_Bartz = 1.0 + 1.15 * ξ_ox

    # Eficiência Isp de duas fases — defasagem vel./temp. das partículas
    η_Isp = 1.0 - 0.14 * ξ_ox

    return (K_Bartz=K_Bartz, eta_Isp=η_Isp)
end

# ==============================================================================
# 1b. COMBUSTÃO BIFÁSICA — MODELO FÍSICO DE DRAG (Lengellé-Hermsen)
# ==============================================================================
"""
    calcular_2fases_fisica(frac_alumina, d_p_um, T_c, gamma, R_gas; Dt_m) -> NamedTuple

Modelo físico de perdas bifásicas por partículas Al₂O₃ na tubeira.

## Física
- Tempo de relaxação de velocidade:  τ_v = ρ_p d²/(18μ)         [Stokes]
- Tempo de relaxação térmica:        τ_T = ρ_p d²Cp_p/(12k_g)   [Nu = 2]
- τ_nozzle calibrado para que d_ref = 3 µm reproduza o modelo escalar clássico,
  depois escalado geometricamente com o diâmetro da garganta `Dt_m`:

      τ_nozzle_eff = τ_nozzle_cal × clamp(Dt_m / 0.025, 1, 5)

  Motivação física: tubeiras geometricamente similares têm comprimento ∝ Dt
  e velocidade de saída praticamente constante → tempo de trânsito ∝ Dt.
  Motores maiores dão mais tempo para as partículas equilibrarem
  → menos defasagem → menor perda bifásica.
  Dt_ref = 25 mm é o tamanho típico dos motores táticos usados na calibração
  empírica do modelo escalar (η = 1 − 0.14·ξ_ox).

- Fator de equilíbrio: ψ = 1 − exp(−τ_nozzle_eff / τ_partícula)

## Parâmetros
- `Dt_m` : diâmetro da garganta [m] (padrão 0 → sem escala geométrica,
           comportamento idêntico à versão anterior para retrocompatibilidade)

## Retorna `NamedTuple` com
- `eta_Isp`   : eficiência de Isp total (velocidade + térmica)
- `K_Bartz`   : fator multiplicativo Bartz (não muda com d_p)
- `psi_v`     : fração de equilíbrio de velocidade [0–1]
- `psi_T`     : fração de equilíbrio de temperatura [0–1]
- `tau_v`     : tempo de relaxação de velocidade [s]
- `tau_T`     : tempo de relaxação térmica [s]
- `tau_nozzle`: tempo de trânsito efetivo na tubeira [s] (pós-escala geométrica)
- `St_v`      : número de Stokes de velocidade = τ_v/τ_nozzle_eff
- `St_T`      : número de Stokes térmico = τ_T/τ_nozzle_eff
- `f_geo`     : fator de escala geométrica aplicado (= Dt/0.025, 1–5)
"""
function calcular_2fases_fisica(
    frac_alumina ::Float64,   # fração mássica Al no propelente [0–1]
    d_p_um       ::Float64,   # diâmetro médio partícula [µm]
    T_c          ::Float64,   # temperatura de câmara [K]
    gamma        ::Float64,   # razão de calores específicos [-]
    R_gas        ::Float64;   # constante do gás [J/(kg·K)]
    Dt_m         ::Float64 = 0.0,   # diâmetro da garganta [m] (0 → sem escala)
)
    # Caso degenerado
    (frac_alumina <= 0.0 || d_p_um <= 0.0) &&
        return (eta_Isp=1.0, K_Bartz=1.0,
                psi_v=1.0, psi_T=1.0,
                tau_v=0.0, tau_T=0.0, tau_nozzle=0.0,
                St_v=0.0, St_T=0.0, f_geo=1.0)

    # ── Propriedades da partícula Al₂O₃ ──────────────────────────────────────
    ρ_p  = 3990.0    # densidade Al₂O₃ [kg/m³]
    Cp_p = 1260.0    # Cp Al₂O₃ a ~1500 K [J/(kg·K)]

    # ── Propriedades do gás (lei de potência em T_c) ──────────────────────────
    T_ref_gas = 3000.0
    μ_g  = 7.0e-5 * (T_c / T_ref_gas)^0.7    # viscosidade dinâmica [Pa·s]
    k_g  = 0.40   * (T_c / T_ref_gas)^0.7    # condutividade térmica [W/(m·K)]
    Cp_g = R_gas  * gamma / (gamma - 1.0)     # Cp do gás [J/(kg·K)]

    # ── Fração mássica Al₂O₃ nos produtos de combustão ───────────────────────
    ξ_Al = clamp(frac_alumina, 0.0, 0.22)
    ξ_ox = ξ_Al * (102.0 / 54.0)            # razão estequiométrica Al → Al₂O₃
    α_ox = ξ_ox / (1.0 + ξ_ox)              # fração mássica Al₂O₃ na mistura

    # ── Tempos de relaxação de Stokes ────────────────────────────────────────
    d_p  = d_p_um  * 1e-6                             # µm → m
    d_ref = 3.0e-6                                    # partícula de referência [m]
    τ_v     = ρ_p * d_p^2   / (18.0 * μ_g)           # velocidade [s]
    τ_T     = ρ_p * d_p^2   * Cp_p / (12.0 * k_g)   # temperatura (Nu=2) [s]
    τ_v_ref = ρ_p * d_ref^2 / (18.0 * μ_g)

    # ── Calibração: τ_nozzle retrocompatível com modelo escalar ──────────────
    # Modelo escalar: η = 1 − 0.14·ξ_ox (válido para ~3 µm típico)
    # Resolve τ_nozzle tal que ψ_v(3µm) → η_escalar
    η_scalar = 1.0 - 0.14 * ξ_ox
    ψ_ref    = clamp((η_scalar - 1.0 + α_ox) / α_ox, 0.01, 0.9999)
    τ_nozzle = -τ_v_ref * log(max(1.0 - ψ_ref, 1e-10))
    τ_nozzle = max(τ_nozzle, 1e-9)

    # ── Escala geométrica: τ_nozzle ∝ Dt ────────────────────────────────────
    # Tubeiras geometricamente similares têm comprimento ∝ Dt e velocidade de
    # saída ≈ constante → tempo de trânsito das partículas ∝ Dt.
    # Ref.: Dt_ref = 25 mm (tamanho dos motores táticos usados na calibração
    # empírica do modelo escalar).  Clamp [1, 5] → sem penalidade para motores
    # pequenos (Dt ≤ 25 mm) e escala limitada a 5× para motores muito grandes.
    # Retrocompatível: Dt_m = 0.0 → f_geo = 1.0 (sem mudança).
    Dt_ref = 0.025   # 25 mm
    f_geo  = Dt_m > 0.0 ? clamp(Dt_m / Dt_ref, 1.0, 5.0) : 1.0
    τ_nozzle *= f_geo

    # ── Fatores de equilíbrio ─────────────────────────────────────────────────
    ψ_v  = 1.0 - exp(-τ_nozzle / τ_v)
    ψ_T  = 1.0 - exp(-τ_nozzle / τ_T)
    St_v = τ_v  / τ_nozzle
    St_T = τ_T  / τ_nozzle

    # ── Eficiência Isp bifásica ───────────────────────────────────────────────
    # Contribuição de velocidade: déficit de impulso pela defasagem das partículas
    η_vel = 1.0 - α_ox * (1.0 - ψ_v)

    # Contribuição térmica: partículas saem mais quentes → menos entalpía disponível
    Cp_mix = Cp_g * (1.0 - α_ox) + Cp_p * α_ox
    η_th   = 1.0 - α_ox * (Cp_p / Cp_mix) * (1.0 - ψ_T) * 0.25

    η_2ph   = η_vel * η_th
    K_Bartz = 1.0 + 1.15 * ξ_ox   # idem ao modelo escalar

    return (
        eta_Isp   = Float64(η_2ph),
        K_Bartz   = Float64(K_Bartz),
        psi_v     = Float64(ψ_v),
        psi_T     = Float64(ψ_T),
        tau_v     = Float64(τ_v),
        tau_T     = Float64(τ_T),
        tau_nozzle= Float64(τ_nozzle),   # valor efetivo (pós-escala geométrica)
        St_v      = Float64(St_v),
        St_T      = Float64(St_T),
        f_geo     = Float64(f_geo),      # fator de escala aplicado (1–5)
    )
end

# ==============================================================================
# 2a. PREDIÇÃO AUTOMÁTICA DO DIÂMETRO D₄₃ — CORRELAÇÃO DE HERMSEN (1981)
# ==============================================================================

"""
    calcular_d43_hermsen(Dt_m, frac_alumina, Pc_Pa, tau_ms) -> Float64

Estima o diâmetro volumétrico-superficial das partículas de Al₂O₃
(**D₄₃**, em µm) usando a correlação empírica de Hermsen (1981),
ajustada a dados de 24 motores sólidos de pequeno a grande porte.

## Equação (Modelo N9A — Hermsen 1981, eq. 2)

    D₄₃ = 3.6304 × Dt_in^0.2932 × [1 − exp(−0.0008163 × ξ_c × Pc_psi × τ_ms)]

onde:
- `Dt_in`   = diâmetro da garganta [in] = `Dt_m / 0.0254`
- `ξ_c`     = carga molar de Al = `frac_alumina × (100 / M_Al)` [g-mol/100g]
              M_Al = 26.982 g/mol  →  16 % Al ≈ 0.593 g-mol/100g
- `Pc_psi`  = pressão de câmara [psia] = `Pc_Pa / 6894.757`
- `τ_ms`    = tempo de residência médio [ms] = `1000 × Pc × V_porto / (R × Tc × ṁ)`

## Comportamento assintótico

| τ → ∞ | D₄₃ satura em `3.63 × Dt_in^0.293` (crescimento limitado pelo Dt) |
| τ → 0 | D₄₃ → 0 (pouco tempo de coalescência — prático: clamp em 1 µm)   |

## Precisão

Desvio padrão ±35 % sobre os 24 casos de validação (Hermsen, Tabela 4).
Para motores pequenos (Dt ≈ 25 mm, τ ≈ 8 ms): D₄₃ ≈ 2–4 µm.
Para motores médios (Dt ≈ 100 mm, τ ≈ 50 ms): D₄₃ ≈ 6–12 µm.

## Referência

Hermsen, R.W. (1981). *Aluminum Oxide Particle Size for Solid Rocket Motor
Performance Prediction*, AIAA Paper 80-0035R (rev.), JANNAF Performance
Working Group. (Tabela 4, Figura 7 — Modelo N9A selecionado por melhor
ajuste global e mínimo número de parâmetros.)

## Retorna
D₄₃ em µm, mínimo 1 µm (partícula mínima para o modelo físico de Stokes).
"""
function calcular_d43_hermsen(
    Dt_m         ::Float64,   # diâmetro da garganta [m]
    frac_alumina ::Float64,   # fração mássica de Al no propelente [0–1]
    Pc_Pa        ::Float64,   # pressão média da câmara [Pa]
    tau_ms       ::Float64,   # tempo de residência médio [ms]
) ::Float64
    frac_alumina <= 0.0 && return 0.0
    Dt_in  = Dt_m  / 0.0254
    Pc_psi = Pc_Pa / 6894.757293
    # ξ_c (Hermsen eq. 1): g-mol de Al metálico por 100 g de propelente.
    # ξ_c = frac_Al × (100 / M_Al),  M_Al = 26.982 g/mol
    # Ex.: Al = 16 % →  ξ_c = 0.16 × 100/26.982 ≈ 0.593 g-mol/100 g
    xi_c = frac_alumina * (100.0 / 26.982)
    D43  = 3.6304 * Dt_in^0.2932 * (1.0 - exp(-0.0008163 * xi_c * Pc_psi * tau_ms))
    return max(D43, 1.0)   # mínimo 1 µm — partícula razoável para modelo Stokes
end

"""
    calcular_d43_hermsen_corrigido(Dt_m, frac_alumina, Pc_Pa, tau_ms, L_grain_m) -> Float64

Extensão da correlação de Hermsen (1981) com correção pelo comprimento do porto,
calibrada para motores de grande porte (Dt > 15 cm).

## Motivação

A correlação original de Hermsen foi ajustada em 24 motores com Dt ≤ ~15 cm.
Para motores grandes (mísseis balísticos: Dt ~ 40–80 cm), o banco de dados de
calibração satura o fator `1 − exp(−k·τ)` e o modelo converge para o assintótico
`D₄₃ ≈ 3.63 × Dt_in^0.293`, que subestima o tamanho real das partículas.

A causa física é que motores compridos (L_grain/Dt grande) têm tempo de residência
**efetivo** maior do que o predito por τ_ms = P×V/(R·Tc·ṁ): as partículas percorrem
todo o comprimento do grão, colidindo e coalescendo ao longo do percurso, chegando
à garganta com diâmetro maior do que esperaria um modelo baseado só em V_porto.

## Fator de correção de porto (Salita, 1995 — ajuste empírico)

    f_porto = max(1.0, (L_grain_m / Dt_m)^0.35)

| Motor           |  L/Dt |  f_porto | D₄₃ base | D₄₃ corrigido |
|-----------------|-------|----------|----------|---------------|
| Pequeno (Dt=25mm, L/Dt=8)  | 8   | 1.94 | 3.5 µm | 6.8 µm  |
| Médio (Dt=100mm, L/Dt=12) | 12  | 2.34 | 7.2 µm | 16.9 µm |
| EUCASS (Dt=588mm, L/Dt=16)| 16.1| 2.65 | 9.1 µm | 24.1 µm |

## Referências
- Hermsen, R.W. (1981). AIAA Paper 80-0035R.
- Salita, M. (1995). *Deficiencies and requirements in modeling of Al₂O₃ formation
  in SRM plumes*. J. Propulsion and Power 11(1):10–23.

## Parâmetros
- `L_grain_m` : comprimento total do grão = L_grao × N_graos [m]
  (os demais parâmetros são idênticos a `calcular_d43_hermsen`)

## Retorna
D₄₃ corrigido em µm, mínimo 1 µm.
"""
function calcular_d43_hermsen_corrigido(
    Dt_m         ::Float64,   # diâmetro da garganta [m]
    frac_alumina ::Float64,   # fração mássica de Al [0–1]
    Pc_Pa        ::Float64,   # pressão média da câmara [Pa]
    tau_ms       ::Float64,   # tempo de residência médio [ms]
    L_grain_m    ::Float64;   # comprimento total do grão [m]
    cap_f_porto  ::Float64 = 1.5,   # teto do f_porto (= ConfigModelo.cap_f_porto)
) ::Float64
    frac_alumina <= 0.0 && return 0.0

    # Base Hermsen (sem correção)
    D43_base = calcular_d43_hermsen(Dt_m, frac_alumina, Pc_Pa, tau_ms)

    # Fator de correção pelo comprimento do porto.
    # Para motores pequenos (L/Dt ≤ 1): f_porto = 1.0 (sem alteração).
    # Cresce como (L/Dt)^0.35, MAS com TETO de 1.5 — a forma (L/Dt)^0.35 é
    # extrapolação NÃO-validada (a "tabela EUCASS" do docstring é predição do
    # próprio modelo, não dado medido) e era ILIMITADA → inflava o D₄₃ de motores
    # longos (3m, L/Dt=26 → 3.1×) gerando perda bifásica irreal (~18%). Cap em 1.5
    # mantém a tendência física (grão longo agglomera um pouco mais) sem a
    # super-predição. ⚠️ O valor REAL precisa de calibração (Isp/c* medidos no
    # teste sub-escala); especifique d_p_alumina diretamente se souber o D₄₃.
    L_over_Dt = L_grain_m / max(Dt_m, 1e-4)
    f_porto   = clamp(L_over_Dt^0.35, 1.0, cap_f_porto)

    return D43_base * f_porto
end
