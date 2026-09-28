using DelimitedFiles
using Statistics
using Printf

# =========================================================
# MÉTRICAS E COMPARAÇÃO CFD x OPENMOTOR (VERSÃO ATUALIZADA)
# =========================================================

"""
Lê o CSV do CFD.

Formato novo (13 colunas):
1=t, 2=Pcam, 3=Phead, 4=F, 5=Mach, 6=m_dot_s, 7=m_dot_g,
8=p_e, 9=rho_e, 10=u_e, 11=Tsup, 12=n_ign, 13=A_queima

Formato antigo (12 colunas):
1=t, 2=Pcam, 3=F, 4=Mach, 5=m_dot_s, 6=m_dot_g,
7=p_e, 8=rho_e, 9=u_e, 10=Tsup, 11=n_ign, 12=A_queima

Retorna:
t, P, F, Aq

Obs.:
- Se existir Phead (formato novo), ele é usado como pressão principal de comparação.
- Se não existir, usa Pcam (formato antigo).
"""
function ler_csv_cfd(caminho::String)
    if !isfile(caminho)
        return nothing, nothing, nothing, nothing
    end

    dados = readdlm(caminho, ',', Float64)
    ncols = size(dados, 2)

    if ncols >= 13
        # novo formato: usa Phead
        t  = dados[:, 1]
        P  = dados[:, 3]
        F  = dados[:, 4]
        Aq = dados[:, 13]
        return t, P, F, Aq

    elseif ncols == 12
        # formato antigo
        t  = dados[:, 1]
        P  = dados[:, 2]
        F  = dados[:, 3]
        Aq = dados[:, 12]
        return t, P, F, Aq

    else
        error("CSV do CFD com formato inesperado: $(ncols) colunas.")
    end
end

"""
Lê o CSV do OpenMotor (ajusta se tiver cabeçalho).

Retorna:
t, P, F
"""
function ler_csv_openmotor(caminho::String)
    if !isfile(caminho)
        return nothing, nothing, nothing
    end

    dados_raw = readdlm(caminho, ',', Any)
    inicio = (dados_raw[1,1] isa Number) ? 1 : 2
    dados = Float64.(dados_raw[inicio:end, :])

    return dados[:, 1], dados[:, 2], dados[:, 3]
end

# ---------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------

function interp_linear(x_ref, y_ref, xq)
    itp = (x) -> begin
        if x <= x_ref[1]
            return y_ref[1]
        end
        if x >= x_ref[end]
            return y_ref[end]
        end

        i = findfirst(v -> v > x, x_ref) - 1
        return y_ref[i] + (y_ref[i+1] - y_ref[i]) * (x - x_ref[i]) / (x_ref[i+1] - x_ref[i])
    end

    return itp.(xq)
end

integrar_trapezio(t, y) = sum(0.5 .* diff(t) .* (y[1:end-1] .+ y[2:end]))

# ---------------------------------------------------------
# Métricas auxiliares
# ---------------------------------------------------------

function detectar_burnout(t, F; limiar_pct=0.05)
    f_max = maximum(F)
    idx = findlast(f -> f >= f_max * limiar_pct, F)
    return isnothing(idx) ? t[end] : t[idx]
end

function detectar_pico_pos_ign(t, y; t_ign=0.15)
    idx_validos = findall(val -> val >= t_ign, t)

    if isempty(idx_validos)
        return t[argmax(y)], maximum(y)
    end

    sub_t = t[idx_validos]
    sub_y = y[idx_validos]

    return sub_t[argmax(sub_y)], maximum(sub_y)
end

function calcular_mae_tailoff(t, y, y_ref, tb)
    t_inicio = 0.8 * tb
    idx = findall(v -> v >= t_inicio && v <= tb, t)
    return isempty(idx) ? 0.0 : mean(abs.(y[idx] .- y_ref[idx]))
end

# ---------------------------------------------------------
# Função principal de comparação
# ---------------------------------------------------------

function comparar_cfd_openmotor(caminho_cfd::String, caminho_ref::String; D_garganta_ini=0.025, m_consumida=0.0)
    # 1. Carregar dados
    t, P, F, Aq = ler_csv_cfd(caminho_cfd)
    tr, Pr, Fr = ler_csv_openmotor(caminho_ref)

    if isnothing(t) || isnothing(tr)
        return nothing
    end

    # 2. Alinhamento temporal
    Pr_i = interp_linear(tr, Pr, t)
    Fr_i = interp_linear(tr, Fr, t)

    # 3. Cálculos CFD
    tb = detectar_burnout(t, F)
    It = integrar_trapezio(t, F)

    tp_pico, P_pico = detectar_pico_pos_ign(t, P)
    tf_pico, F_pico = detectar_pico_pos_ign(t, F)

    At_ini = (pi * D_garganta_ini^2) / 4.0
    Kn_max = maximum(Aq) / At_ini

    idx_pressao = findall(p -> p > 0.1, P)
    P_media_util = isempty(idx_pressao) ? mean(P) : mean(P[idx_pressao])

    c_star = (P_media_util * 1e6 * At_ini * tb) / max(m_consumida, 1e-6)
    cf = It / max(P_media_util * 1e6 * At_ini * tb, 1e-6)

    # 4. Cálculos referência
    It_ref = integrar_trapezio(tr, Fr)

    # 5. Erros
    err_P = 100 * (maximum(P) - maximum(Pr)) / max(maximum(Pr), 1e-9)
    err_F = 100 * (maximum(F) - maximum(Fr)) / max(maximum(Fr), 1e-9)
    err_It = 100 * (It - It_ref) / max(It_ref, 1e-9)

    rmse_F = sqrt(mean((F .- Fr_i).^2))
    mae_tail = calcular_mae_tailoff(t, F, Fr_i, tb)

    # 6. Dicionário final
    metricas = Dict(
        "Pmax" => maximum(P),
        "Pmax_ref" => maximum(Pr),
        "erro_Pmax" => err_P,

        "Fmax" => maximum(F),
        "Fmax_ref" => maximum(Fr),
        "erro_Fmax" => err_F,

        "It" => It,
        "It_ref" => It_ref,
        "erro_It" => err_It,

        "tb" => tb,
        "t_pico_P" => tp_pico,
        "P_pico_pos_ign" => P_pico,
        "t_pico_F" => tf_pico,
        "F_pico_pos_ign" => F_pico,

        "Kn_max" => Kn_max,
        "c_star" => c_star,
        "Cf" => cf,

        "rmse_F" => rmse_F,
        "mae_tailoff" => mae_tail
    )

    return metricas
end

# ---------------------------------------------------------
# Impressão formatada
# ---------------------------------------------------------

function imprimir_metricas(m)
    if isnothing(m)
        println("[Erro] Métricas não disponíveis para comparação.")
        return
    end

    println("\n" * "="^58)
    println("              ANÁLISE DE PERFORMANCE E VALIDAÇÃO")
    println("="^58)
    @printf("Métrica              |    CFD     |   REFER.   | Erro %%\n")
    println("-"^58)
    @printf("Pressão Máx (MPa)    | %10.2f | %10.2f | %7.2f%%\n", m["Pmax"], m["Pmax_ref"], m["erro_Pmax"])
    @printf("Empuxo Máx (N)       | %10.1f | %10.1f | %7.2f%%\n", m["Fmax"], m["Fmax_ref"], m["erro_Fmax"])
    @printf("Impulso Total (Ns)   | %10.1f | %10.1f | %7.2f%%\n", m["It"], m["It_ref"], m["erro_It"])
    @printf("Tempo Queima (s)     | %10.2f |    ---     |    ---\n", m["tb"])
    println("-"^58)
    println("PARÂMETROS DE PROJETO (CFD):")
    @printf(" > Kn Máximo:          %.2f\n", m["Kn_max"])
    @printf(" > C* (m/s):           %.1f\n", m["c_star"])
    @printf(" > Cf (Coef. Emp.):    %.3f\n", m["Cf"])
    @printf(" > t pico P pós-ign:   %.3f s\n", m["t_pico_P"])
    @printf(" > t pico F pós-ign:   %.3f s\n", m["t_pico_F"])
    println("-"^58)
    println("FIDELIDADE DA CURVA:")
    @printf(" > RMSE Empuxo:        %.2f N\n", m["rmse_F"])
    @printf(" > MAE Tail-off:       %.2f N\n", m["mae_tailoff"])
    println("="^58)
end