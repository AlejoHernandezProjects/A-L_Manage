"""Módulo 0 — Generación del banco sintético.

Produce las cinco tablas sobre las que corren los cinco módulos siguientes:

    ref_rate        tasa de política, 120 meses
    curvas          curva cero Nelson-Siegel, 120 meses × 12 tenores
    tasas_deposito  tasa pagada por producto, ajuste parcial asimétrico
    saldos          saldos de depósitos observados (vista, ahorro, plazo)
    instrumentos    ~3.000 cohortes con su contrato completo

y el `ground_truth.json` con los parámetros verdaderos.

**Por qué generar en vez de usar datos reales.** El objetivo del proyecto no es
estimar bien un banco concreto, es demostrar que las estimaciones del Módulo 2 son
correctas. Con datos reales la beta verdadera de los depósitos es inobservable y el
Módulo 2 solo podría reportar un número sin contraste. Aquí la fijamos nosotros, así
que el Módulo 2 puede reportar su **error**, que es lo que un validador de modelos
independiente pediría.

Correlato: hay que ser disciplinado con lo que es observable y lo que no. Las series
latentes (el core verdadero, la tasa latente del Vasicek) viajan en
`DatasetBundle.latentes` y **no** en las tablas observables. Mezclarlas sería
filtrar la respuesta al examen.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.curves import (
    continuous_to_nominal,
    interpolate_zero,
    nelson_siegel,
    solve_nelson_siegel_anchored,
)

__all__ = [
    "DatasetBundle",
    "month_index",
    "curve_tenors",
    "tenor_labels",
    "generate_reference_rate",
    "generate_yield_curves",
    "generate_deposit_rates",
    "generate_deposit_balances",
    "generate_instruments",
    "build_ground_truth",
    "generate_dataset",
]


# ---------------------------------------------------------------------------
# Utilidades de calendario y tenores
# ---------------------------------------------------------------------------

def month_index(cfg) -> pd.DatetimeIndex:
    """Índice de 120 fines de mes consecutivos (§6.2).

    Args:
        cfg: Configuración (``config.params.make_config()``).

    Returns:
        ``DatetimeIndex`` de fines de mes. El último elemento es la fecha de corte.
    """
    return pd.date_range(
        start=cfg.DATES["inicio"], periods=cfg.DATES["n_meses"], freq=cfg.DATES["freq"]
    )


def curve_tenors(cfg) -> np.ndarray:
    """Tenores de la curva en años, en el orden de las columnas."""
    return np.asarray(cfg.CURVE["tenores_a"], dtype=float)


def _tenor_label(a: float) -> str:
    return f"{round(a * 12)}M" if a < 1 else f"{int(round(a))}A"


def tenor_labels(cfg) -> list[str]:
    """Etiquetas de las columnas de la curva (``"3M"``, ``"10A"``, …)."""
    return [_tenor_label(a) for a in cfg.CURVE["tenores_a"]]


def _regime_frame(cfg, fechas: pd.DatetimeIndex) -> pd.DataFrame:
    """Expande `RATE_REGIMES` a una fila por mes, con rampas lineales dentro de
    cada régimen para θ y para la pendiente objetivo de la curva.

    Args:
        cfg: Configuración.
        fechas: Índice mensual.

    Returns:
        DataFrame con columnas ``regimen``, ``theta``, ``kappa``, ``sigma`` y
        ``pendiente``, indexado por ``fechas``.
    """
    filas = []
    for i in range(len(fechas)):
        mes = i + 1
        for reg in cfg.RATE_REGIMES:
            if reg["mes_inicio"] <= mes <= reg["mes_fin"]:
                span = max(1, reg["mes_fin"] - reg["mes_inicio"])
                p = (mes - reg["mes_inicio"]) / span
                filas.append(
                    {
                        "regimen": reg["nombre"],
                        "theta": reg["theta_inicio"] + p * (reg["theta_fin"] - reg["theta_inicio"]),
                        "kappa": reg["kappa"],
                        "sigma": reg["sigma"],
                        "pendiente": reg["pendiente_inicio"]
                        + p * (reg["pendiente_fin"] - reg["pendiente_inicio"]),
                    }
                )
                break
        else:  # pragma: no cover - los regímenes cubren los 120 meses por diseño
            raise ValueError(f"El mes {mes} no está cubierto por ningún régimen")
    return pd.DataFrame(filas, index=fechas)


# ---------------------------------------------------------------------------
# 1. Tasa de referencia (§6.2)
# ---------------------------------------------------------------------------

def generate_reference_rate(cfg, rng) -> pd.Series:
    """Tasa de política mensual: Vasicek latente + decisión de comité.

    El proceso tiene dos capas:

    1.  Un Vasicek discreto con reversión a un objetivo θ que se mueve por tramos
        (§6.2): tasas bajas → ciclo de alzas → meseta alta → normalización.
    2.  Una capa de **decisión de comité** que publica la tasa en escalones de
        25 pb, sólo se mueve cuando el latente se aparta más de la histéresis, y
        tiene un tope de movimiento por reunión.

    **Por qué la segunda capa.** Un Vasicek puro produce una tasa de política que
    se mueve 7 pb cada mes. Ningún banco central hace eso. Y la diferencia no es
    cosmética: la persistencia es lo que hace que un desfase de repreciación se
    traduzca en margen ganado o perdido durante trimestres. Con una tasa que
    oscilara cada mes, el gap se promediaría a cero y el IRRBB sería un problema
    de segundo orden.

    Args:
        cfg: Configuración.
        rng: ``numpy.random.Generator``.

    Returns:
        Serie mensual ``ref_rate`` indexada por fin de mes.
    """
    fechas = month_index(cfg)
    reg = _regime_frame(cfg, fechas)
    pol = cfg.POLICY_RATE

    escalon = pol["escalon_pb"] / 1e4
    histeresis = pol["histeresis_pb"] / 1e4
    tope = pol["movimiento_max_pb"] / 1e4

    latente = pol["r0"]
    publicada = pol["r0"]
    dt = 1.0 / 12.0
    salida = []

    for i in range(len(fechas)):
        theta, kappa, sigma = reg["theta"].iat[i], reg["kappa"].iat[i], reg["sigma"].iat[i]
        latente += kappa * (theta - latente) * dt + sigma * np.sqrt(dt) * rng.standard_normal()
        if abs(latente - publicada) >= histeresis:
            objetivo = round(latente / escalon) * escalon
            movimiento = np.clip(objetivo - publicada, -tope, tope)
            movimiento = round(movimiento / escalon) * escalon
            publicada = max(publicada + movimiento, pol["piso"])
        salida.append(publicada)

    return pd.Series(salida, index=fechas, name="ref_rate")


# ---------------------------------------------------------------------------
# 2. Curva de rendimiento (§6.3)
# ---------------------------------------------------------------------------

def _enforce_monotone_logdf(ceros: np.ndarray, tenores: np.ndarray, piso: float) -> np.ndarray:
    """Garantiza que ``y(τ)·τ`` sea estrictamente creciente (control #6).

    El ruido idiosincrásico por tenor podría, en un caso extremo, invertir el orden
    entre dos nodos vecinos y producir un factor de descuento mayor a plazo largo
    que a plazo corto — es decir, una tasa forward negativa. Se corrige empujando
    el nodo infractor lo mínimo necesario.
    """
    ceros = np.maximum(ceros, piso)
    w = ceros * tenores
    for k in range(1, len(w)):
        if w[k] <= w[k - 1]:
            w[k] = w[k - 1] + 1e-8
    return w / tenores


def generate_yield_curves(ref_rate: pd.Series, cfg, rng) -> pd.DataFrame:
    """Curva cero Nelson-Siegel para cada fecha, anclada a la tasa de referencia.

    Para cada mes se toma la curvatura β₂ de un AR(1) que revierte al nivel del
    régimen, y se resuelve (β₀, β₁) imponiendo dos anclas:

        y(3M)  = ref_rate
        y(10A) = ref_rate + pendiente del régimen

    El ancla corta hace que el control #5 (curva a 3M dentro de ±25 pb de la
    referencia) se cumpla **por construcción**. El ancla larga es la que produce el
    tramo invertido de los meses 55–81, sin el cual el escenario de empinamiento
    del Módulo 4 no tendría nada que revertir.

    Args:
        ref_rate: Serie de tasa de política de :func:`generate_reference_rate`.
        cfg: Configuración.
        rng: ``numpy.random.Generator``.

    Returns:
        DataFrame de 120 filas × 12 tenores con tasas cero en composición continua.
        Columnas etiquetadas ``"1M"``, ``"3M"``, …, ``"30A"``.
    """
    fechas = ref_rate.index
    reg = _regime_frame(cfg, fechas)
    tenores = curve_tenors(cfg)
    cur = cfg.CURVE
    lam = cur["tau_lambda"]
    ruido = cur["ruido_tenor_pb"] / 1e4
    piso = cur["piso_tasa_cero"]

    b2 = cur["curvatura_por_regimen"][reg["regimen"].iat[0]]
    filas = []
    for i in range(len(fechas)):
        objetivo_b2 = cur["curvatura_por_regimen"][reg["regimen"].iat[i]]
        b2 = (
            cur["curvatura_phi"] * b2
            + (1 - cur["curvatura_phi"]) * objetivo_b2
            + cur["curvatura_sigma"] * rng.standard_normal()
        )
        y_corto = float(ref_rate.iat[i])
        y_largo = y_corto + float(reg["pendiente"].iat[i])
        b0, b1 = solve_nelson_siegel_anchored(
            y_corto, y_largo, cur["tenor_ancla_corto_a"], cur["tenor_ancla_largo_a"], b2, lam
        )
        ceros = nelson_siegel(tenores, b0, b1, b2, lam)
        ceros = ceros + ruido * rng.standard_normal(len(tenores))
        filas.append(_enforce_monotone_logdf(ceros, tenores, piso))

    return pd.DataFrame(np.array(filas), index=fechas, columns=tenor_labels(cfg))


# ---------------------------------------------------------------------------
# 3. Tasas pagadas al depósito (§6.4)
# ---------------------------------------------------------------------------

def generate_deposit_rates(ref_rate: pd.Series, cfg, rng) -> pd.DataFrame:
    """Tasa pagada por producto bajo ajuste parcial con traspaso **asimétrico**.

        d*_t = d*_{t-1} + β^± · Δr_t                 traspaso asimétrico
        d*_t = d*_t + κ · (α + β̄·r_t − d*_t)         ancla competitiva de largo plazo
        d_t  = d_{t-1} + λ · (d*_t − d_{t-1}) + ε_t   ajuste parcial

    con β⁺ cuando la referencia sube y β⁻ cuando baja, y β̄ = (β⁺+β⁻)/2. La asimetría
    β⁻ > β⁺ es el hecho estilizado central del negocio de depósitos: el banco
    traslada las bajadas rápido y las subidas lento.

    **Por qué el traspaso va sobre Δr y no sobre el nivel.** La fórmula literal de
    §6.4 aplica β^± al nivel objetivo, y eso invierte el signo: con r = 5% el
    objetivo salta de 1,30% (régimen de subida) a 2,80% (régimen de bajada), de modo
    que un recorte de tasas *subiría* la remuneración del depositante. Generando así,
    la beta estimada de vista sale negativa. La asimetría pertenece al traspaso del
    cambio — que es además cómo la industria usa la palabra "beta": *de una subida de
    100 pb trasladamos 25*.

    Consecuencia para el proyecto: una beta estimada por OLS simple sobre estos datos
    está **mal especificada**. Devuelve un promedio ponderado por la frecuencia de
    subidas y bajadas de la muestra, no un parámetro estructural. Por eso el control
    #8 es WARNING y no ERROR — esperamos que avise, y el Módulo 2 debe explicar por
    qué.

    Args:
        ref_rate: Tasa de política mensual.
        cfg: Configuración.
        rng: ``numpy.random.Generator``.

    Returns:
        DataFrame con columnas ``vista``, ``ahorro``, ``plazo``.
    """
    productos = ["vista", "ahorro", "plazo"]
    dr = cfg.DEPOSIT_RATES
    kappa = dr["kappa_ancla"]
    r = ref_rate.to_numpy()
    delta_r = np.diff(r, prepend=r[0])

    salida = {}
    for prod in productos:
        p = dr[prod]
        beta_bar = 0.5 * (p["beta_up"] + p["beta_down"])
        objetivo = p["alpha"] + p["beta_up"] * r[0]
        d = objetivo
        serie = []
        for i in range(len(r)):
            beta = p["beta_up"] if delta_r[i] >= 0 else p["beta_down"]
            objetivo += beta * delta_r[i]
            objetivo += kappa * (p["alpha"] + beta_bar * r[i] - objetivo)
            d += p["lambda_"] * (objetivo - d) + dr["sigma_eps"] * rng.standard_normal()
            d = max(d, dr["piso"])
            serie.append(d)
        salida[prod] = serie

    return pd.DataFrame(salida, index=ref_rate.index)


# ---------------------------------------------------------------------------
# 4. Saldos de depósitos (§6.5)
# ---------------------------------------------------------------------------

def _perfil_estres(cfg, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Perfil del episodio de estrés de liquidez del mes 96.

    Devuelve dos series separadas porque son dos fenómenos distintos:

    * ``transitorio``: la corrida propiamente dicha, que se recupera.
    * ``permanente``: la erosión del core que **no** vuelve. Clientes que se
      fueron a otro banco y no regresan.

    Separarlas importa: el Módulo 2 debe estimar la vida promedio del core sobre
    una serie que contiene un salto permanente de nivel, que es exactamente el tipo
    de contaminación que rompe una estimación ingenua de persistencia.
    """
    est = cfg.NMD_PARAMS["estres"]
    inicio = est["mes"] - 1  # a índice base 0
    transitorio = np.zeros(n)
    permanente = np.zeros(n)
    erosion = est["erosion_permanente_core"]
    caidas = est["caidas"]
    caida_total = sum(caidas)
    residual = caida_total * (1.0 - est["fraccion_recuperada"])  # p.ej. −3%
    m_rec = est["meses_recuperacion"]

    for k in range(n - inicio):
        t = inicio + k
        if k < len(caidas):
            total = sum(caidas[: k + 1])
        elif k < len(caidas) + m_rec:
            p = (k - len(caidas) + 1) / m_rec
            total = caida_total + p * (residual - caida_total)
        elif k < len(caidas) + 2 * m_rec:
            p = (k - len(caidas) - m_rec + 1) / m_rec
            total = residual + p * (-erosion - residual)
        else:
            total = -erosion
        perm = min(erosion, erosion * (k + 1) / max(1, len(caidas)))
        permanente[t] = perm
        transitorio[t] = total + perm

    return transitorio, permanente


def generate_deposit_balances(
    ref_rate: pd.Series, dep_rates: pd.DataFrame, cfg, rng
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Saldos de depósitos con descomposición estructural ``B_t = C_t + V_t``.

    El **core** se genera como stock con runoff y originación nueva::

        C_t = C_{t-1} · exp(−λ_d/12) + N_t,    λ_d = 1 / vida_promedio_a

    con ``N_t`` calibrado para que la tendencia neta observada sea +4,5% anual.
    No es una simple tendencia decreciente, y la diferencia es de identificación:
    si el core sólo fuera una tendencia, la vida promedio no estaría en los datos y
    el Módulo 2 estaría "recuperando" un parámetro que nunca se usó para generar.

    El **volátil** recoge estacionalidad multiplicativa, un AR(1) de media cero, la
    migración hacia plazo cuando el diferencial ``r − d`` se abre, y el episodio de
    estrés del mes 96.

    La migración conserva el fondeo: lo que sale de vista y ahorro **entra a
    plazo**. En la mesa esto se llama *deposit mix shift*, y es media razón por la
    que el costo de fondeo sube en un ciclo de alzas aunque ninguna beta por
    producto se mueva — el banco no subió precios, cambió de producto.

    Los niveles se anclan al final: la última observación de cada producto se
    escala al share de §5, de modo que la serie y la tabla de instrumentos digan lo
    mismo en la fecha de corte.

    Args:
        ref_rate: Tasa de política mensual.
        dep_rates: Tasas pagadas por producto.
        cfg: Configuración.
        rng: ``numpy.random.Generator``.

    Returns:
        ``(saldos, latentes)``. ``saldos`` es lo **observable**: columnas ``vista``,
        ``ahorro``, ``plazo``. ``latentes`` lleva el core verdadero y las
        componentes, y no debe usarse fuera de la validación y del cálculo de error
        del Módulo 2.
    """
    fechas = ref_rate.index
    n = len(fechas)
    nmd = cfg.NMD_PARAMS
    total = cfg.BANK_PROFILE["activos_totales_musd"]
    transitorio, permanente = _perfil_estres(cfg, n)
    meses_cal = np.array([f.month for f in fechas])

    observables, latentes = {}, {}
    salidas_a_plazo = np.zeros(n)

    for prod in ("vista", "ahorro"):
        p = nmd[prod]
        lam_d = 1.0 / p["vida_promedio_a"]
        runoff = np.exp(-lam_d / 12.0)
        g = p["crecimiento_anual"]

        # --- core: runoff + originación nueva, con tendencia neta +g anual
        core = np.zeros(n)
        c = 1.0
        core[0] = c
        for t in range(1, n):
            objetivo = np.exp(g * t / 12.0) * (1.0 - permanente[t])
            nuevo = objetivo - core[t - 1] * runoff
            nuevo *= 1.0 + nmd["sigma_originacion"] * rng.standard_normal()
            core[t] = core[t - 1] * runoff + max(nuevo, 0.0)

        # --- volátil: estacionalidad + AR(1) + migración + estrés
        estacional = np.array([nmd["estacionalidad"][m] for m in meses_cal])
        ar = np.zeros(n)
        for t in range(1, n):
            ar[t] = nmd["ar1_phi"] * ar[t - 1] + p["sigma_volatil"] * rng.standard_normal()

        exceso = np.maximum(0.0, (ref_rate.to_numpy() - dep_rates[prod].to_numpy()) - nmd["spread_normal"])
        mig = np.zeros(n)
        for t in range(1, n):
            mig[t] = nmd["phi_migracion"] * mig[t - 1] + (1 - nmd["phi_migracion"]) * (
                -p["gamma_migracion"] * exceso[t]
            )

        v = estacional + ar + mig + transitorio
        base = core / p["core_share"]
        saldo = base * (1.0 + v)

        ancla = total * cfg.INSTRUMENT_SPECS[prod]["share"] / saldo[-1]
        observables[prod] = saldo * ancla
        latentes[f"{prod}_core"] = core * ancla / p["core_share"] * p["core_share"]
        latentes[f"{prod}_volatil"] = (saldo - core) * ancla
        latentes[f"{prod}_migracion"] = mig
        salidas_a_plazo += -np.minimum(mig, 0.0) * base * ancla

    # --- plazo: tendencia propia + lo que migró desde vista y ahorro
    g_plazo = nmd["vista"]["crecimiento_anual"]
    base_plazo = np.exp(g_plazo * np.arange(n) / 12.0)
    ar_plazo = np.zeros(n)
    for t in range(1, n):
        ar_plazo[t] = nmd["ar1_phi"] * ar_plazo[t - 1] + 0.008 * rng.standard_normal()
    saldo_plazo = base_plazo * (1.0 + ar_plazo + 0.5 * transitorio)
    saldo_plazo = saldo_plazo / saldo_plazo[-1] * (
        total * cfg.INSTRUMENT_SPECS["plazo"]["share"] - salidas_a_plazo[-1]
    ) + salidas_a_plazo
    observables["plazo"] = saldo_plazo

    saldos = pd.DataFrame(observables, index=fechas)
    lat = pd.DataFrame(latentes, index=fechas)
    return saldos, lat


# ---------------------------------------------------------------------------
# 5. Instrumentos (§6.6)
# ---------------------------------------------------------------------------

def _fin_de_mes(fecha: pd.Timestamp, meses: int) -> pd.Timestamp:
    """Desplaza ``meses`` fines de mes desde ``fecha`` (que ya es fin de mes)."""
    return fecha + pd.offsets.MonthEnd(meses)


def _fraccion_principal_vivo(
    amortizacion: str, tasa_anual: float, plazo_meses: int, edad_meses: int,
    frecuencia_pago_meses: int
) -> float:
    """Fracción del principal original que sigue viva tras ``edad_meses``.

    Para una cohorte con cuota nivelada (sistema francés) el saldo insoluto es

        S(k)/S(0) = ((1+i)^n − (1+i)^k) / ((1+i)^n − 1)

    con ``i`` la tasa del período y ``n`` el número total de pagos. Para un bullet
    la fracción es 1 hasta el vencimiento.

    **Por qué importa.** Sin este factor, una cohorte hipotecaria originada hace
    ocho años pesa en el balance por su monto original, cuando en realidad ya
    devolvió cerca de un tercio del principal. El efecto agregado es que el libro
    parece más viejo — y por tanto más corto — de lo que es, y la duración del
    activo sale sesgada a la baja. Es un detalle de contabilidad de cohortes, no
    de finanzas, y es de los que se olvidan al construir balances sintéticos.

    Args:
        amortizacion: ``"frances"`` o ``"bullet"``.
        tasa_anual: Tasa nominal anual de la cohorte.
        plazo_meses: Plazo original en meses.
        edad_meses: Meses transcurridos desde la originación.
        frecuencia_pago_meses: Meses entre pagos.

    Returns:
        Fracción en (0, 1].
    """
    if amortizacion != "frances" or plazo_meses <= 0:
        return 1.0
    freq = max(1, int(frecuencia_pago_meses))
    n = max(1, int(np.ceil(plazo_meses / freq)))
    k = min(n, int(edad_meses // freq))
    i = tasa_anual * freq / 12.0
    if i <= 0:
        return max(0.0, 1.0 - k / n)
    return float(((1 + i) ** n - (1 + i) ** k) / ((1 + i) ** n - 1))


def _tasa_fija_originacion(
    curvas: pd.DataFrame, tenores: np.ndarray, fecha: pd.Timestamp, plazo_meses: int,
    spread: float, pagos_por_anio: int
) -> float:
    """Tasa contractual de una cohorte fija: curva cero del mes de originación al
    plazo del contrato, convertida a nominal, más el spread de crédito."""
    ceros = curvas.loc[fecha].to_numpy()
    y = float(interpolate_zero(tenores, ceros, max(plazo_meses, 1) / 12.0))
    return float(continuous_to_nominal(y, pagos_por_anio)) + spread


def generate_instruments(
    cfg,
    curvas: pd.DataFrame,
    ref_rate: pd.Series,
    dep_rates: pd.DataFrame,
    rng,
) -> pd.DataFrame:
    """Inventario de ~3.000 cohortes (vintages) vivas en la fecha de corte.

    Cada fila es una **cohorte de producto**, no un contrato individual: el conjunto
    de créditos hipotecarios a 20 años originados en marzo de 2018, por ejemplo.
    Lo que importa es que las fechas de originación estén repartidas en los 120
    meses. Si todo se originara el mismo día, el balance entero repactaría en una
    sola banda temporal y tanto el gap del Módulo 1 como el EVE del Módulo 4 serían
    artefactos del generador.

    **La decisión de diseño que más importa**: la tasa contractual de cada cohorte
    fija se fija con la curva **de su mes de originación**, no con la de hoy. Es el
    mecanismo económico central del proyecto — la cartera hipotecaria originada en
    los meses 1–30, con la referencia al 1,5%, sigue viva hoy con la referencia al
    3%, y es esa cartera la que produce la pérdida de EVE ante +200 pb. Si todas
    las cohortes se originaran a la tasa de hoy, el deterioro de EVE sería un
    artefacto y no un resultado.

    Los saldos se reescalan al share exacto de §5 por categoría, de modo que
    Activos = Pasivos + Patrimonio (control #1) se cumple por construcción.

    Nota sobre la firma: §6.7 la propone como ``(cfg, curve_cutoff, rng)``. Recibe
    la historia completa de la curva porque, por lo dicho arriba, la tasa de una
    cohorte depende de su mes de originación y no sólo del corte.

    Args:
        cfg: Configuración.
        curvas: Curvas cero mensuales.
        ref_rate: Tasa de política mensual.
        dep_rates: Tasas pagadas por producto.
        rng: ``numpy.random.Generator``.

    Returns:
        DataFrame de cohortes con contrato completo, indexado por ``id``.
    """
    fechas = curvas.index
    n_meses = len(fechas)
    corte = fechas[-1]
    tenores = curve_tenors(cfg)
    total = cfg.BANK_PROFILE["activos_totales_musd"]

    filas: list[dict] = []
    for categoria, spec in cfg.INSTRUMENT_SPECS.items():
        filas.extend(
            _cohortes_de_categoria(
                categoria, spec, cfg, fechas, curvas, ref_rate, dep_rates, tenores, rng
            )
        )

    df = pd.DataFrame(filas)

    # --- reescalado exacto al mix de §5
    for categoria, spec in cfg.INSTRUMENT_SPECS.items():
        m = df["categoria"] == categoria
        objetivo = total * spec["share"]
        df.loc[m, "saldo"] = df.loc[m, "peso"] / df.loc[m, "peso"].sum() * objetivo

    df = df.drop(columns=["peso"]).set_index("id")
    assert (df["fecha_repreciacion"] <= df["fecha_vencimiento"]).all()
    return df


def _cohortes_de_categoria(
    categoria, spec, cfg, fechas, curvas, ref_rate, dep_rates, tenores, rng
) -> list[dict]:
    """Genera las cohortes vivas de una categoría.

    Se sobre-genera y se filtra por supervivencia en lugar de muestrear
    directamente la distribución condicional de cohortes vivas. Es más lento pero
    reproduce la composición correcta: en un producto de plazo corto sólo
    sobreviven las cohortes recientes, y ese sesgo de supervivencia es real — la
    cartera de depósitos a plazo de un banco está toda originada en los últimos
    dos años, y por eso repacta rápido.
    """
    corte = fechas[-1]
    n_meses = len(fechas)
    plazo_min, plazo_max = spec["plazo_meses"]
    n_obj = spec["n_cohortes"]
    lado = spec["lado"]
    g = cfg.NMD_PARAMS["vista"]["crecimiento_anual"]

    if categoria in ("vista", "ahorro"):
        tasa = float(dep_rates[categoria].iat[-1])
        return [
            {
                "id": f"{categoria.upper()}-0001",
                "categoria": categoria,
                "lado": lado,
                "peso": 1.0,
                "saldo": np.nan,
                "tipo_tasa": "administrada",
                "tasa": tasa,
                "factor_devengo": 1.0,
                "indice_referencia": None,
                "spread_sobre_referencia": np.nan,
                "fecha_originacion": fechas[0],
                "fecha_repreciacion": _fin_de_mes(corte, 1),
                "fecha_vencimiento": _fin_de_mes(corte, 1),
                "meses_a_repreciacion": 1,
                "meses_a_vencimiento": 1,
                "frecuencia_repricing_m": 1,
                "frecuencia_pago_meses": 1,
                "amortizacion": "bullet",
                "sensible": True,
                "es_nmd": True,
                "moneda": cfg.CONVENTIONS["moneda"],
            }
        ]

    if spec["tipo_tasa"] == "no_sensible" or categoria == "patrimonio":
        return [
            {
                "id": f"{categoria.upper()}-{k + 1:04d}",
                "categoria": categoria,
                "lado": lado,
                "peso": 1.0,
                "saldo": np.nan,
                "tipo_tasa": "no_sensible",
                "tasa": 0.0,
                "factor_devengo": 0.0,
                "indice_referencia": None,
                "spread_sobre_referencia": np.nan,
                "fecha_originacion": fechas[0],
                "fecha_repreciacion": _fin_de_mes(corte, 1),
                "fecha_vencimiento": _fin_de_mes(corte, 1),
                "meses_a_repreciacion": 1,
                "meses_a_vencimiento": 1,
                "frecuencia_repricing_m": None,
                "frecuencia_pago_meses": 1,
                "amortizacion": "bullet",
                "sensible": False,
                "es_nmd": False,
                "moneda": cfg.CONVENTIONS["moneda"],
            }
            for k in range(spec["n_cohortes"])
        ]

    # --- sobre-generación y filtro de supervivencia
    plazo_medio = (plazo_min + plazo_max) / 2
    factor = int(np.ceil(n_meses / plazo_medio)) + 2
    n_bruto = n_obj * factor

    mes_orig = rng.integers(1, n_meses + 1, size=n_bruto)
    plazo = rng.integers(plazo_min, plazo_max + 1, size=n_bruto)
    vivas = (mes_orig + plazo) > n_meses
    mes_orig, plazo = mes_orig[vivas], plazo[vivas]
    if len(mes_orig) > n_obj:
        sel = rng.choice(len(mes_orig), size=n_obj, replace=False)
        mes_orig, plazo = mes_orig[sel], plazo[sel]

    frac_fija = spec["frac_fija"]
    es_fija = rng.random(len(mes_orig)) < frac_fija
    spread = (spec["spread_pb"] or 0.0) / 1e4
    freq_pago = spec["frecuencia_pago_meses"]
    rep_m = spec["repricing_meses"]
    pagos_anio = max(1, 12 // freq_pago)

    filas = []
    for k in range(len(mes_orig)):
        o = int(mes_orig[k])
        fecha_o = fechas[o - 1]
        restante = int(o + plazo[k] - n_meses)
        venc = _fin_de_mes(corte, restante)

        if categoria == "plazo":
            # Un depósito a plazo fija su tasa en la emisión y no repacta.
            tasa = float(dep_rates["plazo"].iat[o - 1])
            tipo, idx, spr = "fija", None, np.nan
            m_rep = restante
        elif es_fija[k]:
            tasa = _tasa_fija_originacion(curvas, tenores, fecha_o, int(plazo[k]), spread, pagos_anio)
            tipo, idx, spr = "fija", None, spread
            m_rep = restante
        else:
            # Variable: el último reset ya ocurrió; la tasa vigente es la de ese día.
            transcurridos = n_meses - o
            desde_reset = transcurridos % rep_m
            m_rep = min(rep_m - desde_reset if desde_reset else rep_m, restante)
            mes_reset = max(0, n_meses - 1 - desde_reset)
            tasa = float(continuous_to_nominal(ref_rate.iat[mes_reset], pagos_anio)) + spread
            tipo, idx, spr = "variable", "REF_3M", spread

        if spec["tipo_tasa"] == "administrada":
            tipo = "administrada"

        # El saldo vivo de una cohorte no es lo que se originó: una cohorte
        # amortizable de hace ocho años ya devolvió buena parte del principal.
        # Sin este factor, las añadas viejas pesan de más y el libro parece más
        # corto y más viejo de lo que es.
        peso = np.exp(g * o / 12.0) * np.exp(0.25 * rng.standard_normal())
        peso *= _fraccion_principal_vivo(
            spec["amortizacion"], tasa, int(plazo[k]), n_meses - o, freq_pago
        )
        filas.append(
            {
                "id": f"{categoria.upper()[:4]}-{k + 1:04d}",
                "categoria": categoria,
                "lado": lado,
                "peso": peso,
                "saldo": np.nan,
                "tipo_tasa": tipo,
                "tasa": tasa,
                "factor_devengo": spec.get("revolving_ratio", 1.0),
                "indice_referencia": idx,
                "spread_sobre_referencia": spr,
                "fecha_originacion": fecha_o,
                "fecha_repreciacion": _fin_de_mes(corte, m_rep),
                "fecha_vencimiento": venc,
                "meses_a_repreciacion": m_rep,
                "meses_a_vencimiento": restante,
                "frecuencia_repricing_m": None if tipo == "fija" else rep_m,
                "frecuencia_pago_meses": freq_pago,
                "amortizacion": spec["amortizacion"],
                "sensible": True,
                "es_nmd": False,
                "moneda": cfg.CONVENTIONS["moneda"],
            }
        )
    return filas


# ---------------------------------------------------------------------------
# 6. Ground truth
# ---------------------------------------------------------------------------

def build_ground_truth(cfg) -> dict:
    """Parámetros verdaderos, para que el Módulo 2 pueda medir su error.

    Registra explícitamente **vida promedio y half-life** de cada NMD. Bajo
    decaimiento exponencial difieren en un factor 1/ln2 ≈ 1,443, y confundirlas
    tiene consecuencia regulatoria directa: IRRBB capea la *vida promedio* de
    repreciación del core en 5 años para minorista transaccional, 4,5 para
    minorista no transaccional y 4 para mayorista. Con 4,2 años de vida promedio el
    banco queda holgado; si esos 4,2 fueran half-life, la vida promedio sería 6,06
    años y el core verdadero excedería el tope.

    Args:
        cfg: Configuración.

    Returns:
        Diccionario serializable a JSON.
    """
    ln2 = float(np.log(2.0))
    nmd = {}
    for prod in ("vista", "ahorro"):
        p = cfg.NMD_PARAMS[prod]
        vida = p["vida_promedio_a"]
        nmd[prod] = {
            "vida_promedio_a": vida,
            "half_life_a": vida * ln2,
            "lambda_decaimiento_anual": 1.0 / vida,
            "core_share": p["core_share"],
            "crecimiento_anual": p["crecimiento_anual"],
            "gamma_migracion": p["gamma_migracion"],
            "tope_irrbb_a": 5.0 if prod == "vista" else 4.5,
        }

    betas = {
        prod: {
            "beta_up": cfg.DEPOSIT_RATES[prod]["beta_up"],
            "beta_down": cfg.DEPOSIT_RATES[prod]["beta_down"],
            "beta_largo_plazo": 0.5
            * (cfg.DEPOSIT_RATES[prod]["beta_up"] + cfg.DEPOSIT_RATES[prod]["beta_down"]),
            "lambda_ajuste": cfg.DEPOSIT_RATES[prod]["lambda_"],
            "alpha": cfg.DEPOSIT_RATES[prod]["alpha"],
            "kappa_ancla": cfg.DEPOSIT_RATES["kappa_ancla"],
        }
        for prod in ("vista", "ahorro", "plazo")
    }

    return {
        "_nota": (
            "Ground truth del generador. 'vida_promedio_a' es vida promedio (mean "
            "life), NO half-life; difieren en 1/ln2 y el tope IRRBB aplica a la "
            "primera. Este archivo es la respuesta del examen: el Módulo 2 sólo "
            "debe leerlo para reportar su error de estimación, nunca para estimar."
        ),
        "semilla": cfg.SEED,
        "fechas": {"inicio": cfg.DATES["inicio"], "n_meses": cfg.DATES["n_meses"]},
        "convenciones": cfg.CONVENTIONS,
        "banco": cfg.BANK_PROFILE,
        "betas_deposito": betas,
        "nmd": nmd,
        "estres_liquidez": cfg.NMD_PARAMS["estres"],
        "regimenes_tasa": cfg.RATE_REGIMES,
        "spreads_pb": {
            k: v["spread_pb"] for k, v in cfg.INSTRUMENT_SPECS.items() if v.get("spread_pb") is not None
        },
        "shares_balance": {k: v["share"] for k, v in cfg.INSTRUMENT_SPECS.items()},
        "revolving_ratio_tarjetas": cfg.INSTRUMENT_SPECS["tarjetas"]["revolving_ratio"],
        "objetivos_calibracion": cfg.CALIBRATION_TARGETS,
    }


# ---------------------------------------------------------------------------
# 7. Orquestador
# ---------------------------------------------------------------------------

@dataclass
class DatasetBundle:
    """Las cinco tablas del Módulo 0 más el ground truth.

    Attributes:
        ref_rate: Tasa de política mensual.
        curvas: Curvas cero, 120 × 12.
        tasas_deposito: Tasa pagada por producto.
        saldos: Saldos de depósitos **observables**.
        instrumentos: Cohortes vivas en la fecha de corte.
        latentes: Series latentes (core verdadero, migración). No observables.
        ground_truth: Parámetros verdaderos.
    """

    ref_rate: pd.Series
    curvas: pd.DataFrame
    tasas_deposito: pd.DataFrame
    saldos: pd.DataFrame
    instrumentos: pd.DataFrame
    latentes: pd.DataFrame
    ground_truth: dict = field(default_factory=dict)

    @property
    def fecha_corte(self) -> pd.Timestamp:
        """Fecha desde la que se proyectan NII y EVE."""
        return self.curvas.index[-1]

    def hash(self) -> str:
        """SHA-256 sobre una serialización canónica de las tablas (control #11).

        El formato numérico se fija a 10 decimales para que el hash no dependa de
        la representación en coma flotante por defecto de la versión de pandas.
        """
        h = hashlib.sha256()
        for obj in (self.ref_rate, self.curvas, self.tasas_deposito, self.saldos, self.instrumentos):
            h.update(obj.to_csv(float_format="%.10f").encode("utf-8"))
        h.update(json.dumps(self.ground_truth, sort_keys=True, default=str).encode("utf-8"))
        return h.hexdigest()

    def write(self, directorio: Path) -> None:
        """Escribe las tablas a CSV y el ground truth a JSON.

        Args:
            directorio: Carpeta de salida; se crea si no existe.
        """
        directorio = Path(directorio)
        directorio.mkdir(parents=True, exist_ok=True)
        self.ref_rate.to_csv(directorio / "ref_rate.csv", float_format="%.10f")
        self.curvas.to_csv(directorio / "curvas.csv", float_format="%.10f")
        self.tasas_deposito.to_csv(directorio / "tasas_deposito.csv", float_format="%.10f")
        self.saldos.to_csv(directorio / "saldos_deposito.csv", float_format="%.10f")
        self.latentes.to_csv(directorio / "latentes_nmd.csv", float_format="%.10f")
        self.instrumentos.to_csv(directorio / "instrumentos.csv", float_format="%.10f")
        gt = dict(self.ground_truth)
        gt["hash_dataset"] = self.hash()
        (directorio / "ground_truth.json").write_text(
            json.dumps(gt, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )


def generate_dataset(cfg, seed: int | None = None) -> DatasetBundle:
    """Orquestador del Módulo 0: genera el banco sintético completo.

    El orden importa y no es arbitrario — cada bloque consume el anterior, que es
    la forma de garantizar la regla §9.3 (una sola curva alimenta todo):

        1. tasa de política
        2. curva cero, anclada a esa tasa
        3. tasas de depósito, dirigidas por esa tasa con betas asimétricas
        4. saldos de depósito, que reaccionan al diferencial ``r − d``
        5. instrumentos, con la tasa de cada cohorte tomada de la curva de su
           mes de originación

    Args:
        cfg: Configuración (``config.params.make_config()``).
        seed: Semilla; si es ``None`` se usa ``cfg.SEED``.

    Returns:
        :class:`DatasetBundle` listo para validar.
    """
    rng = np.random.default_rng(cfg.SEED if seed is None else seed)

    ref_rate = generate_reference_rate(cfg, rng)
    curvas = generate_yield_curves(ref_rate, cfg, rng)
    tasas_dep = generate_deposit_rates(ref_rate, cfg, rng)
    saldos, latentes = generate_deposit_balances(ref_rate, tasas_dep, cfg, rng)
    instrumentos = generate_instruments(cfg, curvas, ref_rate, tasas_dep, rng)

    return DatasetBundle(
        ref_rate=ref_rate,
        curvas=curvas,
        tasas_deposito=tasas_dep,
        saldos=saldos,
        instrumentos=instrumentos,
        latentes=latentes,
        ground_truth=build_ground_truth(cfg),
    )
