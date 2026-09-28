# ==============================================================================
# CompararCSV.jl — comparação de curvas P(t) e F(t) entre dois CSVs
# ==============================================================================
# Feito para verificação cruzada código-a-código: sobrepõe uma curva de
# REFERÊNCIA (ex.: OpenMotor) com a do RktPrisma, calcula os desvios e gera a
# figura pronta para o relatório.
#
# Uso rápido (linha de comando):
#   julia --project=. CompararCSV.jl referencia.csv meu_resultado.csv
#
# Uso como biblioteca:
#   include("CompararCSV.jl")
#   comparar_curvas("bates_openmotor.csv", "resultados_v2_GUI_Run.csv";
#                   rotulo_ref = "OpenMotor", rotulo_sim = "RktPrisma",
#                   salvar = "validacao_bates.png")
#
# Formatos aceitos (detectados sozinho):
#   • separador , ; ou TAB  ·  decimal . ou ,  (Excel pt-BR)
#   • com cabeçalho → colunas casadas pelo NOME
#   • sem cabeçalho → assume o layout do resultados_v2_*.csv do solver:
#     t, P[MPa], P_head, F[N], mach_saída, ṁ_saída, ṁ_gerada, A_queima
#
# NOTA SOBRE O TERMO "VALIDAÇÃO": comparar com o OpenMotor é verificação
# CRUZADA entre códigos — os dois resolvem o mesmo modelo. Validação no
# sentido estrito exige P(t) medido em bancada. O cabeçalho do relatório
# impresso diz isso de propósito.
# ==============================================================================

using DelimitedFiles, Printf, Statistics

# ─────────────────────────── LEITURA ROBUSTA ─────────────────────────────────

"""Detecta o separador dominante (,  ;  TAB) a partir da 1ª linha de dados."""
function _detectar_sep(linhas::AbstractVector{<:AbstractString})
    amostra = join(linhas[1:min(5, length(linhas))], "\n")
    cont = Dict(',' => count(==(','), amostra),
                ';' => count(==(';'), amostra),
                '\t' => count(==('\t'), amostra))
    sep, n = ';', cont[';']
    for (k, v) in cont
        v > n && ((sep, n) = (k, v))
    end
    n == 0 && error("CompararCSV: não achei separador (, ; ou TAB) no arquivo.")
    return sep
end

"""
Converte texto em número aceitando decimal `.` ou `,`.
Só troca vírgula por ponto quando a vírgula NÃO é o separador de colunas.
"""
function _num(s::AbstractString, sep::Char)
    t = strip(s)
    (isempty(t) || t in ("NA", "NaN", "nan", "-")) && return NaN
    sep != ',' && (t = replace(t, ',' => '.'))
    v = tryparse(Float64, t)
    return v === nothing ? NaN : v
end

"""true se a linha não parece dado numérico (⇒ é cabeçalho)."""
function _eh_cabecalho(linha::AbstractString, sep::Char)
    campos = split(linha, sep)
    isempty(campos) && return false
    n_num = count(c -> !isnan(_num(c, sep)), campos)
    return n_num < max(1, length(campos) ÷ 2)
end

# Sinônimos aceitos por grandeza (minúsculo, sem espaços)
const _ALIAS = Dict(
    :t  => ["time", "times", "tempo", "t", "ts", "t_s", "time(s)", "tempo(s)", "tempo_s"],
    :P  => ["chamberpressure", "chamberpressure(mpa)", "pressure", "pressao", "pressão",
            "p", "pc", "p_mpa", "pmpa", "p(mpa)", "pressaocamara", "chamberpressure(pa)"],
    :F  => ["thrust", "thrust(n)", "empuxo", "f", "f_n", "fn", "força", "forca",
            "thrust(kn)", "f_corr_n"],
    :Kn => ["kn", "k_n", "klemmung"],
)

_normaliza(s) = lowercase(replace(strip(String(s)), r"[\s_\-]" => ""))

"""Acha o índice da coluna cujo nome casa com um dos apelidos da grandeza."""
function _achar_col(nomes::Vector{String}, grandeza::Symbol)
    alvos = _ALIAS[grandeza]
    nn = _normaliza.(nomes)
    # 1ª passada: igualdade exata
    for (i, n) in enumerate(nn), a in alvos
        n == _normaliza(a) && return i
    end
    # 2ª passada: prefixo (pega "chamberpressure(mpa)" com alias "chamberpressure")
    for (i, n) in enumerate(nn), a in alvos
        (startswith(n, _normaliza(a)) || startswith(_normaliza(a), n)) && return i
    end
    return 0
end

"""
    ler_curva(path; col_t, col_P, col_F, escala_P, escala_F) -> NamedTuple

Lê um CSV e devolve `(t, P, F, Kn, fonte, nomes)`. `P` sai em MPa e `F` em N.
As colunas podem ser forçadas por índice (1-based) se a detecção falhar.
"""
function ler_curva(path::AbstractString;
                   col_t::Int = 0, col_P::Int = 0, col_F::Int = 0,
                   escala_P::Float64 = 1.0, escala_F::Float64 = 1.0)

    isfile(path) || error("CompararCSV: arquivo não encontrado: $path")
    linhas = filter(!isempty, strip.(readlines(path)))
    isempty(linhas) && error("CompararCSV: arquivo vazio: $path")

    sep    = _detectar_sep(linhas)
    tem_cb = _eh_cabecalho(linhas[1], sep)
    nomes  = tem_cb ? String.(strip.(split(linhas[1], sep))) : String[]
    dados_l = tem_cb ? linhas[2:end] : linhas
    isempty(dados_l) && error("CompararCSV: sem linhas de dados em $path")

    ncol = length(split(dados_l[1], sep))
    M = fill(NaN, length(dados_l), ncol)
    for (i, l) in enumerate(dados_l)
        campos = split(l, sep)
        for j in 1:min(ncol, length(campos))
            M[i, j] = _num(campos[j], sep)
        end
    end

    # ── Escolha das colunas ──────────────────────────────────────────────────
    it, iP, iF, iKn = col_t, col_P, col_F, 0
    if tem_cb
        it == 0 && (it = _achar_col(nomes, :t))
        iP == 0 && (iP = _achar_col(nomes, :P))
        iF == 0 && (iF = _achar_col(nomes, :F))
        iKn = _achar_col(nomes, :Kn)
    else
        # Sem cabeçalho: layout do resultados_v2_*.csv do solver
        ncol >= 4 || error("CompararCSV: $path não tem cabeçalho e tem só $ncol " *
                           "coluna(s). Passe col_t/col_P/col_F explicitamente.")
        it == 0 && (it = 1)
        iP == 0 && (iP = 2)
        iF == 0 && (iF = 4)
        @info "CompararCSV: $(basename(path)) sem cabeçalho → assumindo layout " *
              "resultados_v2 (t=1, P=2, F=4). Use col_* para mudar."
    end
    (it == 0 || iP == 0) && error("CompararCSV: não identifiquei as colunas de tempo/pressão em $path. " *
                                  "Cabeçalho lido: $(nomes). Passe col_t/col_P.")

    t = M[:, it]
    P = M[:, iP] .* escala_P
    F = iF > 0 && iF <= ncol ? M[:, iF] .* escala_F : fill(NaN, length(t))
    Kn = iKn > 0 && iKn <= ncol ? M[:, iKn] : fill(NaN, length(t))

    # Descarta linhas com t inválido e ordena por tempo
    ok = .!isnan.(t)
    t, P, F, Kn = t[ok], P[ok], F[ok], Kn[ok]
    ord = sortperm(t)

    # ── Aviso de unidade (não converte sozinho: erro silencioso é pior) ──────
    Pm = maximum(filter(!isnan, P); init = 0.0)
    if Pm > 1e4
        @warn "Pressão de $(basename(path)) chega a $(round(Pm, digits=1)) — parece " *
              "estar em Pa, não MPa. Passe escala_P = 1e-6."
    elseif Pm > 500
        @warn "Pressão de $(basename(path)) chega a $(round(Pm, digits=1)) — parece " *
              "estar em bar/psi. Confira a unidade (escala_P)."
    end

    return (t = t[ord], P = P[ord], F = F[ord], Kn = Kn[ord],
            fonte = basename(path), nomes = nomes, sep = sep, tem_cabecalho = tem_cb)
end

# ─────────────────────────── MÉTRICAS ────────────────────────────────────────

"""Interpola y(x) linearmente em xq (fora do domínio → NaN)."""
function _interp(x::Vector{Float64}, y::Vector{Float64}, xq::Vector{Float64})
    out = similar(xq)
    n = length(x)
    @inbounds for (k, q) in enumerate(xq)
        if q < x[1] || q > x[n]
            out[k] = NaN; continue
        end
        j = searchsortedlast(x, q)
        j >= n && (out[k] = y[n]; continue)
        dx = x[j+1] - x[j]
        out[k] = dx <= 0 ? y[j] : y[j] + (y[j+1] - y[j]) * (q - x[j]) / dx
    end
    return out
end

"""Integral por trapézios, ignorando NaN."""
function _trapz(x::Vector{Float64}, y::Vector{Float64})
    s = 0.0
    @inbounds for k in 2:length(x)
        (isnan(y[k]) || isnan(y[k-1])) && continue
        s += 0.5 * (y[k] + y[k-1]) * (x[k] - x[k-1])
    end
    return s
end

"""
Tempo de ação: primeira e última vez em que o sinal passa de `frac`·máximo.
É a mesma regra aplicada às DUAS curvas — comparar t_burn com definições
diferentes é o erro clássico nesse tipo de tabela.
"""
function _tempo_acao(t::Vector{Float64}, y::Vector{Float64}; frac::Float64 = 0.10)
    v = replace(y, NaN => -Inf)
    ymax = maximum(v)
    ymax <= 0 && return (NaN, NaN, NaN)
    lim = frac * ymax
    i1 = findfirst(>=(lim), v); i2 = findlast(>=(lim), v)
    (i1 === nothing || i2 === nothing) && return (NaN, NaN, NaN)
    return (t[i1], t[i2], t[i2] - t[i1])
end

function _metricas(c; frac_burn::Float64 = 0.10)
    Pv = replace(c.P, NaN => -Inf)
    iPm = argmax(Pv)
    base = isnan(c.F[1]) ? c.P : c.F           # sem empuxo, usa P para o tempo de ação
    t1, t2, tb = _tempo_acao(c.t, base; frac = frac_burn)
    jan = findall(k -> !isnan(c.t[k]) && c.t[k] >= t1 && c.t[k] <= t2, eachindex(c.t))
    P_med = isempty(jan) ? NaN : mean(filter(!isnan, c.P[jan]))
    Fm = all(isnan, c.F) ? NaN : maximum(replace(c.F, NaN => -Inf))
    It = all(isnan, c.F) ? NaN : _trapz(c.t, c.F)
    return (P_max = c.P[iPm], t_Pmax = c.t[iPm], P_med = P_med,
            F_max = Fm, I_total = It, t_burn = tb, t_ini = t1, t_fim = t2)
end

_dif(a, b) = (isnan(a) || isnan(b) || a == 0) ? NaN : 100.0 * (b - a) / a

# ─────────────────────────── COMPARAÇÃO ──────────────────────────────────────

"""
    comparar_curvas(csv_ref, csv_sim; kwargs...) -> NamedTuple

Sobrepõe duas curvas, imprime a tabela de desvios e (se `salvar` for dado)
grava a figura. `csv_ref` é a referência — os erros percentuais são relativos
a ela.

Argumentos opcionais:
  rotulo_ref/rotulo_sim  nomes na legenda
  salvar                 caminho do PNG (ou "" para não salvar)
  titulo                 título da figura
  t_corte                descarta t < t_corte nas DUAS curvas (tira o transiente
                         de ignição, que difere entre códigos e polui o RMSE)
  frac_burn              limiar do tempo de ação (padrão 0,10 = 10% do máximo)
  col_*_ref / col_*_sim  força índices de coluna se a detecção falhar
  escala_P_*/escala_F_*  fatores de unidade (ex.: escala_P_ref = 1e-6 se em Pa)
  csv_saida              grava as curvas alinhadas + resíduo num CSV
"""
function comparar_curvas(csv_ref::AbstractString, csv_sim::AbstractString;
        rotulo_ref::String = "Referência", rotulo_sim::String = "RktPrisma",
        salvar::String = "comparacao.png", titulo::String = "",
        t_corte::Float64 = 0.0, frac_burn::Float64 = 0.10,
        col_t_ref::Int = 0, col_P_ref::Int = 0, col_F_ref::Int = 0,
        col_t_sim::Int = 0, col_P_sim::Int = 0, col_F_sim::Int = 0,
        escala_P_ref::Float64 = 1.0, escala_F_ref::Float64 = 1.0,
        escala_P_sim::Float64 = 1.0, escala_F_sim::Float64 = 1.0,
        csv_saida::String = "", residuo::Bool = false)

    ref = ler_curva(csv_ref; col_t = col_t_ref, col_P = col_P_ref, col_F = col_F_ref,
                    escala_P = escala_P_ref, escala_F = escala_F_ref)
    sim = ler_curva(csv_sim; col_t = col_t_sim, col_P = col_P_sim, col_F = col_F_sim,
                    escala_P = escala_P_sim, escala_F = escala_F_sim)

    if t_corte > 0
        cut(c) = (i = findall(>=(t_corte), c.t);
                  (t = c.t[i], P = c.P[i], F = c.F[i], Kn = c.Kn[i],
                   fonte = c.fonte, nomes = c.nomes, sep = c.sep,
                   tem_cabecalho = c.tem_cabecalho))
        ref = cut(ref); sim = cut(sim)
    end

    mr = _metricas(ref; frac_burn = frac_burn)
    ms = _metricas(sim; frac_burn = frac_burn)

    # ── Resíduo na malha COMUM de tempo (interseção dos domínios) ────────────
    t_lo = max(minimum(ref.t), minimum(sim.t))
    t_hi = min(maximum(ref.t), maximum(sim.t))
    rmse_P = nrmse_P = maxdif_P = NaN
    rmse_F = nrmse_F = NaN
    tg = Float64[]; Pr_i = Float64[]; Ps_i = Float64[]; Fr_i = Float64[]; Fs_i = Float64[]
    if t_hi > t_lo
        tg   = collect(range(t_lo, t_hi; length = 600))
        Pr_i = _interp(ref.t, ref.P, tg); Ps_i = _interp(sim.t, sim.P, tg)
        Fr_i = _interp(ref.t, ref.F, tg); Fs_i = _interp(sim.t, sim.F, tg)
        okP = .!(isnan.(Pr_i) .| isnan.(Ps_i))
        if any(okP)
            d = Ps_i[okP] .- Pr_i[okP]
            rmse_P  = sqrt(mean(d .^ 2))
            faixa   = maximum(Pr_i[okP]) - minimum(Pr_i[okP])
            nrmse_P = faixa > 0 ? 100 * rmse_P / faixa : NaN
            maxdif_P = maximum(abs.(d))
        end
        okF = .!(isnan.(Fr_i) .| isnan.(Fs_i))
        if any(okF)
            d = Fs_i[okF] .- Fr_i[okF]
            rmse_F  = sqrt(mean(d .^ 2))
            faixa   = maximum(Fr_i[okF]) - minimum(Fr_i[okF])
            nrmse_F = faixa > 0 ? 100 * rmse_F / faixa : NaN
        end
    else
        @warn "Os dois arquivos não têm faixa de tempo em comum — sem resíduo."
    end

    # ── Relatório ────────────────────────────────────────────────────────────
    println("\n", "="^78)
    println("  VERIFICAÇÃO CRUZADA DE CURVAS")
    println("  referência : $(ref.fonte)   [$rotulo_ref]")
    println("  comparado  : $(sim.fonte)   [$rotulo_sim]")
    t_corte > 0 && @printf("  corte      : t < %.3f s descartado\n", t_corte)
    println("="^78)
    @printf("%-22s%14s%14s%12s\n", "métrica", rotulo_ref, rotulo_sim, "desvio")
    println("-"^78)
    linha(nm, a, b, un) = @printf("%-22s%14.4g%14.4g%11s\n", nm * " [" * un * "]", a, b,
        isnan(_dif(a, b)) ? "—" : @sprintf("%+.2f%%", _dif(a, b)))
    linha("P_max",    mr.P_max,   ms.P_max,   "MPa")
    linha("t(P_max)", mr.t_Pmax,  ms.t_Pmax,  "s")
    linha("P_média",  mr.P_med,   ms.P_med,   "MPa")
    linha("F_max",    mr.F_max,   ms.F_max,   "N")
    linha("I_total",  mr.I_total, ms.I_total, "N·s")
    linha("t_ação",   mr.t_burn,  ms.t_burn,  "s")
    println("-"^78)
    if !isnan(rmse_P)
        @printf("  RMSE de P      : %.4f MPa   (%.2f%% da amplitude da referência)\n", rmse_P, nrmse_P)
        @printf("  desvio máx. |ΔP|: %.4f MPa\n", maxdif_P)
    end
    !isnan(rmse_F) && @printf("  RMSE de F      : %.1f N     (%.2f%% da amplitude)\n", rmse_F, nrmse_F)
    println("="^78)
    println("  Nota: comparação código-a-código é VERIFICAÇÃO cruzada. Validação")
    println("  no sentido estrito exige P(t) medido em bancada.")
    println("="^78, "\n")

    # ── CSV das curvas alinhadas ─────────────────────────────────────────────
    if !isempty(csv_saida) && !isempty(tg)
        open(csv_saida, "w") do io
            println(io, "t_s;P_ref_MPa;P_sim_MPa;dP_MPa;F_ref_N;F_sim_N;dF_N")
            for k in eachindex(tg)
                vals = (tg[k], Pr_i[k], Ps_i[k], Ps_i[k] - Pr_i[k],
                        Fr_i[k], Fs_i[k], Fs_i[k] - Fr_i[k])
                println(io, join((replace(@sprintf("%.6f", v), "." => ",") for v in vals), ";"))
            end
        end
        println("CSV alinhado (pt-BR): $csv_saida")
    end

    # ── Figura ───────────────────────────────────────────────────────────────
    if !isempty(salvar)
        try
            @eval using Plots
            Base.invokelatest() do
                gr()
                ttl = isempty(titulo) ? "$rotulo_sim × $rotulo_ref" : titulo
                temF = !all(isnan, ref.F) || !all(isnan, sim.F)

                p1 = plot(ref.t, ref.P, lw = 3, color = :black, linestyle = :dash,
                          label = rotulo_ref, ylabel = "Pressão na câmara [MPa]",
                          title = ttl, titlefontsize = 11, legend = :best,
                          grid = true, gridalpha = 0.3,
                          background_color = :white, foreground_color = :black)
                plot!(p1, sim.t, sim.P, lw = 2, color = :crimson, label = rotulo_sim)
                # O RMSE sai só no relatório impresso: na figura ele brigava com a
                # caixa de legenda e poluía o gráfico.

                # Painel do resíduo é opcional (residuo=true); o padrão é a figura
                # enxuta de duas curvas, que é a que vai para o relatório.
                p3 = nothing
                if residuo && !isempty(tg)
                    p3 = plot(tg, Ps_i .- Pr_i, lw = 1.6, color = :steelblue, label = "",
                              ylabel = "ΔP [MPa]", xlabel = "tempo [s]",
                              grid = true, gridalpha = 0.3,
                              background_color = :white, foreground_color = :black)
                    hline!(p3, [0.0], color = :gray60, lw = 1, linestyle = :dot, label = "")
                end

                p2 = nothing
                if temF
                    p2 = plot(ref.t, ref.F ./ 1e3, lw = 3, color = :black, linestyle = :dash,
                              label = "", ylabel = "Empuxo [kN]",
                              xlabel = p3 === nothing ? "tempo [s]" : "",
                              grid = true, gridalpha = 0.3,
                              background_color = :white, foreground_color = :black)
                    plot!(p2, sim.t, sim.F ./ 1e3, lw = 2, color = :crimson, label = "")
                else
                    plot!(p1, xlabel = p3 === nothing ? "tempo [s]" : "")
                end

                paineis = filter(!isnothing, Any[p1, p2, p3])
                if length(paineis) == 3
                    plot(paineis..., layout = grid(3, 1, heights = [0.42, 0.34, 0.24]),
                         size = (900, 850), left_margin = 7Plots.mm)
                elseif length(paineis) == 2
                    plot(paineis..., layout = grid(2, 1, heights = [0.5, 0.5]),
                         size = (900, 680), left_margin = 7Plots.mm)
                else
                    plot(paineis[1], size = (900, 430), left_margin = 7Plots.mm)
                end
                savefig(salvar)
                println("Figura: $salvar")
            end
        catch e
            @warn "Não consegui plotar (Plots indisponível?)" exception = e
        end
    end

    return (ref = mr, sim = ms, rmse_P = rmse_P, nrmse_P = nrmse_P,
            maxdif_P = maxdif_P, rmse_F = rmse_F, nrmse_F = nrmse_F,
            t_comum = (t_lo, t_hi))
end

# ─────────────────────────── CLI ─────────────────────────────────────────────
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 2
        println("""
        Uso: julia --project=. CompararCSV.jl <referencia.csv> <simulacao.csv> [saida.png]

          <referencia.csv>  curva de referência (ex.: export do OpenMotor)
          <simulacao.csv>   curva do RktPrisma (resultados_v2_*.csv ou export do GUI)
          [saida.png]       opcional; padrão comparacao.png

        Exemplo:
          julia --project=. CompararCSV.jl bates_openmotor.csv resultados_v2_GUI_Run.csv
        """)
        exit(1)
    end
    comparar_curvas(ARGS[1], ARGS[2];
        rotulo_ref = "OpenMotor", rotulo_sim = "RktPrisma",
        salvar = length(ARGS) >= 3 ? ARGS[3] : "comparacao.png",
        csv_saida = "comparacao_alinhada.csv")
end
