"""Utilidades de curva: construcción Nelson-Siegel, interpolación, descuento,
flujos contractuales y duración.

Este módulo existe porque las mismas dos operaciones — interpolar un cero a un
plazo arbitrario y descontar un flujo — se usan en el Módulo 0 (fijar la tasa de
originación de cada cohorte con la curva vigente ese mes) y otra vez en el Módulo 4
(valorar el balance bajo los seis escenarios). Duplicarlas rompería la regla de
consistencia §9.3: *la curva que descuenta en el Módulo 4 es la misma que alimenta
las tasas del Módulo 1*.

Convención: **tasas cero en composición continua**, `DF(τ) = exp(−y(τ)·τ)`.
Consecuencia que conviene tener presente: bajo composición continua la duración
modificada **es igual** a la de Macaulay. El divisor `(1 + y/m)` del libro de texto
aparece sólo bajo composición discreta.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "year_fraction_30_360",
    "nelson_siegel",
    "solve_nelson_siegel_anchored",
    "interpolate_zero",
    "discount_factors",
    "schedule_cashflows",
    "present_value",
    "macaulay_duration",
    "modified_duration",
    "par_rate",
    "price_fixed_bond",
    "continuous_to_nominal",
]


def continuous_to_nominal(y, pagos_por_anio: int = 12):
    """Convierte una tasa continua a tasa nominal anual con ``m`` pagos al año.

        r_nom = m · (e^{y/m} − 1)

    La curva se guarda en composición continua porque es lo cómodo para descontar,
    pero un contrato de crédito no se escribe así: se escribe con una tasa nominal
    y una frecuencia de pago. Convertir en la frontera evita el error clásico de
    tratar ambas como intercambiables — a 5% y 12 pagos la diferencia es de 1 pb,
    despreciable, pero a 20% son 17 pb y ya se nota en la cartera de tarjetas.

    Args:
        y: Tasa continua (escalar o arreglo).
        pagos_por_anio: Número de pagos al año.

    Returns:
        Tasa nominal anual equivalente.
    """
    m = float(pagos_por_anio)
    return m * (np.exp(np.asarray(y, dtype=float) / m) - 1.0)


# ---------------------------------------------------------------------------
# Convención de días
# ---------------------------------------------------------------------------

def year_fraction_30_360(d1, d2) -> float:
    """Fracción de año entre dos fechas bajo base 30/360 (US).

    Cada mes vale 30 días y el año 360. Es la convención de §6.6 y la habitual en
    banca comercial: hace que las cuotas mensuales de un crédito sean idénticas
    entre sí, lo que a su vez hace que el cuadro de amortización sea el mismo que
    ve el cliente en su contrato.

    Args:
        d1: Fecha inicial (``datetime``/``pd.Timestamp``).
        d2: Fecha final.

    Returns:
        Fracción de año, positiva si ``d2 > d1``.
    """
    y1, m1, dd1 = d1.year, d1.month, min(d1.day, 30)
    y2, m2, dd2 = d2.year, d2.month, d2.day
    if dd1 == 30 and dd2 == 31:
        dd2 = 30
    return ((y2 - y1) * 360 + (m2 - m1) * 30 + (dd2 - dd1)) / 360.0


# ---------------------------------------------------------------------------
# Nelson-Siegel
# ---------------------------------------------------------------------------

def _ns_loadings(tau: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Cargas (loadings) de pendiente y curvatura de Nelson-Siegel.

    Se evalúan con un límite explícito en ``τ → 0`` porque ``(1−e^{−x})/x`` es
    indeterminado en cero y ahí vive el ancla corta.
    """
    tau = np.asarray(tau, dtype=float)
    x = tau / lam
    with np.errstate(divide="ignore", invalid="ignore"):
        l1 = np.where(x < 1e-8, 1.0, (1.0 - np.exp(-x)) / np.where(x < 1e-8, 1.0, x))
    l2 = l1 - np.exp(-x)
    return l1, l2


def nelson_siegel(tau, b0: float, b1: float, b2: float, lam: float) -> np.ndarray:
    """Tasa cero Nelson-Siegel a los plazos ``tau`` (en años).

        y(τ) = β₀ + β₁·L₁(τ) + β₂·L₂(τ)

    con β₀ el nivel de largo plazo, β₁ la pendiente (``y(0) = β₀ + β₁``) y β₂ la
    curvatura, cuya carga es máxima alrededor de ``τ ≈ 1,79·λ``.

    Args:
        tau: Plazo o arreglo de plazos, en años.
        b0, b1, b2: Nivel, pendiente y curvatura.
        lam: Parámetro de decaimiento τλ, en años.

    Returns:
        Tasas cero en composición continua.
    """
    l1, l2 = _ns_loadings(tau, lam)
    return b0 + b1 * l1 + b2 * l2


def solve_nelson_siegel_anchored(
    y_corto: float,
    y_largo: float,
    tau_corto: float,
    tau_largo: float,
    b2: float,
    lam: float,
) -> tuple[float, float]:
    """Resuelve (β₀, β₁) para que la curva pase exactamente por dos anclas.

    Dados β₂ y λ, imponer ``y(τ_corto) = y_corto`` e ``y(τ_largo) = y_largo`` deja
    un sistema lineal 2×2 en (β₀, β₁):

        β₀ + β₁·L₁(τ_c) = y_c − β₂·L₂(τ_c)
        β₀ + β₁·L₁(τ_l) = y_l − β₂·L₂(τ_l)

    **Por qué anclar en vez de calibrar libremente.** El control #5 exige que la
    curva a 3M quede dentro de ±25 pb de la tasa de referencia en *cada* una de las
    120 fechas. Ajustar Nelson-Siegel por mínimos cuadrados y después rezar para que
    el ancla corta caiga en rango es frágil. Anclando, el control pasa por
    construcción y β₂ queda libre para dar forma al tramo intermedio.

    Económicamente el anclaje corto es correcto: el tramo corto de la curva es,
    salvo primas pequeñas, la trayectoria esperada de la tasa de política.

    Args:
        y_corto: Tasa cero objetivo en el ancla corta.
        y_largo: Tasa cero objetivo en el ancla larga.
        tau_corto: Plazo del ancla corta, en años.
        tau_largo: Plazo del ancla larga, en años.
        b2: Curvatura, tomada como dada.
        lam: Parámetro de decaimiento τλ.

    Returns:
        Tupla ``(β₀, β₁)``.
    """
    l1, l2 = _ns_loadings(np.array([tau_corto, tau_largo]), lam)
    a = np.array([[1.0, l1[0]], [1.0, l1[1]]])
    rhs = np.array([y_corto - b2 * l2[0], y_largo - b2 * l2[1]])
    b0, b1 = np.linalg.solve(a, rhs)
    return float(b0), float(b1)


# ---------------------------------------------------------------------------
# Interpolación y descuento
# ---------------------------------------------------------------------------

def interpolate_zero(tenores: np.ndarray, ceros: np.ndarray, tau) -> np.ndarray:
    """Interpola la tasa cero al plazo ``tau``, linealmente en ``y(τ)·τ``.

    **Por qué en ``y·τ`` y no en ``y``.** ``y(τ)·τ`` es menos el logaritmo del
    factor de descuento. Interpolar linealmente ahí equivale a interpolar el log-DF,
    lo que preserva la monotonía decreciente de los factores de descuento y hace que
    las tasas forward implícitas entre nodos sean constantes en lugar de saltar.
    Interpolar linealmente en ``y`` puede generar forwards negativos entre nodos
    cuando la curva está invertida — exactamente el régimen que este proyecto
    genera a propósito en los meses 55–81.

    Fuera del rango de tenores se extrapola plano (la tasa del nodo extremo), que es
    la práctica de mesa habitual.

    Args:
        tenores: Plazos de los nodos, en años, crecientes.
        ceros: Tasas cero en los nodos, composición continua.
        tau: Plazo o arreglo de plazos a interpolar, en años.

    Returns:
        Tasas cero interpoladas, con la misma forma que ``tau``.
    """
    tenores = np.asarray(tenores, dtype=float)
    ceros = np.asarray(ceros, dtype=float)
    tau = np.asarray(tau, dtype=float)

    tau_c = np.clip(tau, tenores[0], tenores[-1])
    w = np.interp(tau_c, tenores, ceros * tenores)
    return w / tau_c


def discount_factors(tenores: np.ndarray, ceros: np.ndarray, tau) -> np.ndarray:
    """Factores de descuento ``DF(τ) = exp(−y(τ)·τ)``.

    Args:
        tenores: Plazos de los nodos de la curva, en años.
        ceros: Tasas cero en los nodos, composición continua.
        tau: Plazos a descontar, en años.

    Returns:
        Factores de descuento. ``DF(0) = 1``.
    """
    tau = np.asarray(tau, dtype=float)
    y = interpolate_zero(tenores, ceros, tau)
    return np.exp(-y * tau)


# ---------------------------------------------------------------------------
# Flujos contractuales
# ---------------------------------------------------------------------------

def schedule_cashflows(
    nominal: float,
    tasa_anual: float,
    plazo_meses: int,
    amortizacion: str = "bullet",
    frecuencia_pago_meses: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Calendario de flujos contractuales de un instrumento.

    Args:
        nominal: Saldo vivo hoy, en las unidades del balance (USD M).
        tasa_anual: Tasa nominal anual del contrato.
        plazo_meses: Meses hasta el vencimiento desde hoy.
        amortizacion: ``"bullet"`` (interés periódico y principal al final) o
            ``"frances"`` (cuota constante que amortiza principal e interés).
        frecuencia_pago_meses: Meses entre pagos.

    Returns:
        ``(tiempos, montos)`` con los tiempos en años (base 30/360) y los montos
        incluyendo principal e interés.

    Notas:
        El sistema francés (cuota nivelada) es el estándar de la banca minorista y
        es la razón por la que la duración de una hipoteca a 20 años ronda los 6–8
        años y no los 20: el principal se devuelve gradualmente, así que el flujo
        está mucho más adelantado en el tiempo de lo que sugiere el plazo final.
        Tratar la cartera hipotecaria como bullet sobrestima el riesgo de tasa de
        forma severa.
    """
    if plazo_meses <= 0:
        return np.array([]), np.array([])

    freq = max(1, int(frecuencia_pago_meses))
    n_pagos = max(1, int(np.ceil(plazo_meses / freq)))
    tiempos = np.array([(k + 1) * freq / 12.0 for k in range(n_pagos)])
    tasa_periodo = tasa_anual * freq / 12.0

    if amortizacion == "frances" and n_pagos > 1 and tasa_periodo > 0:
        cuota = nominal * tasa_periodo / (1.0 - (1.0 + tasa_periodo) ** (-n_pagos))
        montos = np.full(n_pagos, cuota)
    elif amortizacion == "frances" and n_pagos > 1:
        montos = np.full(n_pagos, nominal / n_pagos)
    else:  # bullet
        montos = np.full(n_pagos, nominal * tasa_periodo)
        montos[-1] += nominal

    return tiempos, montos


# ---------------------------------------------------------------------------
# Valoración y sensibilidad
# ---------------------------------------------------------------------------

def present_value(
    tiempos: np.ndarray, montos: np.ndarray, tenores: np.ndarray, ceros: np.ndarray
) -> float:
    """Valor presente de un calendario de flujos descontado a la curva cero.

    Args:
        tiempos: Tiempos de los flujos, en años.
        montos: Montos de los flujos.
        tenores: Plazos de los nodos de la curva, en años.
        ceros: Tasas cero de la curva, composición continua.

    Returns:
        Valor presente.
    """
    if len(tiempos) == 0:
        return 0.0
    return float(np.sum(montos * discount_factors(tenores, ceros, tiempos)))


def macaulay_duration(
    tiempos: np.ndarray, montos: np.ndarray, tenores: np.ndarray, ceros: np.ndarray
) -> float:
    """Duración de Macaulay: plazo medio ponderado por valor presente.

    Args:
        tiempos: Tiempos de los flujos, en años.
        montos: Montos de los flujos.
        tenores: Plazos de los nodos de la curva, en años.
        ceros: Tasas cero de la curva.

    Returns:
        Duración en años. Cero si no hay flujos o el valor presente es nulo.
    """
    if len(tiempos) == 0:
        return 0.0
    pvs = montos * discount_factors(tenores, ceros, tiempos)
    total = pvs.sum()
    if abs(total) < 1e-12:
        return 0.0
    return float(np.sum(tiempos * pvs) / total)


def modified_duration(
    tiempos: np.ndarray, montos: np.ndarray, tenores: np.ndarray, ceros: np.ndarray
) -> float:
    """Duración modificada: sensibilidad relativa del VP a un choque paralelo.

    Bajo composición continua ``−(1/VP)·∂VP/∂y`` coincide exactamente con la
    duración de Macaulay; el divisor ``(1 + y/m)`` del libro de texto es un artefacto
    de la composición discreta. Se mantiene como función aparte porque el Módulo 4
    la usa con nombre propio y porque la equivalencia merece estar documentada, no
    escondida en un alias.

    Args:
        tiempos: Tiempos de los flujos, en años.
        montos: Montos de los flujos.
        tenores: Plazos de los nodos de la curva, en años.
        ceros: Tasas cero de la curva.

    Returns:
        Duración modificada en años.
    """
    return macaulay_duration(tiempos, montos, tenores, ceros)


def par_rate(
    tenores: np.ndarray,
    ceros: np.ndarray,
    plazo_a: float,
    frecuencia_pago_meses: int = 6,
) -> float:
    """Tasa cupón que hace que un bono bullet cotice exactamente a la par.

        c* = (1 − DF(T)) / (Δ · Σ_k DF(t_k))

    Args:
        tenores: Plazos de los nodos de la curva, en años.
        ceros: Tasas cero de la curva.
        plazo_a: Plazo del bono, en años.
        frecuencia_pago_meses: Meses entre cupones.

    Returns:
        Tasa cupón anual a la par.
    """
    delta = frecuencia_pago_meses / 12.0
    n = max(1, int(round(plazo_a / delta)))
    tiempos = np.array([(k + 1) * delta for k in range(n)])
    dfs = discount_factors(tenores, ceros, tiempos)
    return float((1.0 - dfs[-1]) / (delta * dfs.sum()))


def price_fixed_bond(
    nominal: float,
    cupon_anual: float,
    plazo_a: float,
    tenores: np.ndarray,
    ceros: np.ndarray,
    frecuencia_pago_meses: int = 6,
) -> float:
    """Precio de un bono bullet a tasa fija descontado a la curva cero.

    Args:
        nominal: Valor nominal.
        cupon_anual: Tasa cupón anual.
        plazo_a: Plazo al vencimiento, en años.
        tenores: Plazos de los nodos de la curva, en años.
        ceros: Tasas cero de la curva.
        frecuencia_pago_meses: Meses entre cupones.

    Returns:
        Precio (valor presente).
    """
    tiempos, montos = schedule_cashflows(
        nominal,
        cupon_anual,
        int(round(plazo_a * 12)),
        amortizacion="bullet",
        frecuencia_pago_meses=frecuencia_pago_meses,
    )
    return present_value(tiempos, montos, tenores, ceros)
