"""Módulo 0 — Controles de calidad del dataset (§6.8).

Cada control devuelve un :class:`Check`. **ERROR bloquea el avance al Módulo 1;
WARNING sólo informa.** La distinción no es de comodidad: un ERROR señala que el
dataset viola una identidad contable o una consistencia interna, y cualquier
resultado calculado sobre él sería basura. Un WARNING señala que el dataset es
válido pero que algún supuesto quedó tenso, y eso es información que va al informe,
no un motivo para parar.

Este archivo es el equivalente artesanal de lo que en un banco haría la función de
validación independiente de modelos (SR 11-7 en EE. UU., TRIM en la eurozona): un
conjunto de pruebas escritas *antes* de mirar los resultados, que se corren siempre
y cuyo veredicto no se negocia. La diferencia entre un proyecto de datos y un modelo
que un supervisor acepta está casi toda aquí.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.curves import discount_factors, macaulay_duration, schedule_cashflows
from src.data_gen import curve_tenors

__all__ = ["Check", "calibration_metrics", "run_all_checks", "report", "assert_no_errors"]


# ---------------------------------------------------------------------------
# Estructura de un control
# ---------------------------------------------------------------------------

@dataclass
class Check:
    """Resultado de un control de calidad.

    Attributes:
        numero: Número del control en la tabla de §6.8.
        nombre: Descripción corta.
        valor: Valor medido.
        umbral: Umbral contra el que se compara.
        severidad: ``"ERROR"`` o ``"WARNING"``.
        passed: Si el control se superó.
        detalle: Explicación legible, sobre todo cuando falla.
    """

    numero: int
    nombre: str
    valor: Any
    umbral: Any
    severidad: str
    passed: bool
    detalle: str = ""


# ---------------------------------------------------------------------------
# Métricas de calibración (§6.1)
# ---------------------------------------------------------------------------

def _duracion_instrumento(fila, tenores: np.ndarray, ceros: np.ndarray) -> float:
    """Duración modificada de una cohorte según su tipo de tasa.

    * **Fija**: duración de los flujos contractuales descontados a la curva.
    * **Variable / administrada**: el plazo hasta la próxima repreciación. Un
      instrumento a tasa flotante devuelve su valor al par en cada reset, así que su
      sensibilidad a la tasa se agota ahí, por más lejos que esté el vencimiento.
      Esta es la razón mecánica de que 16.200 M de cartera comercial variable
      aporten casi nada al riesgo de EVE del banco.
    """
    if fila["tipo_tasa"] == "fija":
        tiempos, montos = schedule_cashflows(
            float(fila["saldo"]),
            float(fila["tasa"]),
            int(fila["meses_a_vencimiento"]),
            str(fila["amortizacion"]),
            int(fila["frecuencia_pago_meses"]),
        )
        return macaulay_duration(tiempos, montos, tenores, ceros)
    return float(fila["meses_a_repreciacion"]) / 12.0


def _duracion_nmd(cfg, prod: str, y_corto: float) -> tuple[float, float]:
    """Duración del core de un NMD bajo el modelo de runoff exponencial.

    Con runoff a tasa λ y descuento a la tasa ``y``, la duración del core es
    ``1/(λ + y)``: el flujo esperado es ``λe^{−λt}`` y su plazo medio ponderado por
    valor presente sale de esa integral. La porción volátil se trata como overnight.

    Devuelve dos números que **no** son intercambiables:

    * *contractual/runoff*: sin ajustar por beta. Es la convención de los objetivos
      de §6.1.
    * *efectiva*: multiplicada por ``(1 − β⁺)``, que reconoce que el banco sólo
      traslada una fracción del choque y por tanto el pasivo se comporta como si
      fuera más corto.

    Mezclar ambas es el error que hace que dos áreas del mismo banco reporten
    duraciones de pasivo distintas por un factor de dos y nadie sepa cuál usar.
    """
    p = cfg.NMD_PARAMS[prod]
    lam = 1.0 / p["vida_promedio_a"]
    d_core = 1.0 / (lam + max(y_corto, 1e-6))
    d_runoff = p["core_share"] * d_core
    beta = cfg.DEPOSIT_RATES[prod]["beta_up"]
    return d_runoff, d_runoff * (1.0 - beta)


def calibration_metrics(bundle, cfg) -> dict:
    """Métricas de §6.1 medidas sobre el dataset generado.

    Las de EVE y NII son **aproximaciones de primer orden** deliberadas: el ΔEVE
    sale de la duración del gap y el ΔNII de un gap estático a 12 meses. Sirven
    como diagnóstico de calibración en el Módulo 0. Los Módulos 3 y 4 los
    recalculan con proyección y revaluación completas, y **la diferencia entre
    ambos métodos es uno de los entregables** — es donde se ve dónde falla la
    aproximación lineal ante choques grandes.

    Args:
        bundle: :class:`~src.data_gen.DatasetBundle`.
        cfg: Configuración.

    Returns:
        Diccionario de métricas.
    """
    inst = bundle.instrumentos
    tenores = curve_tenors(cfg)
    ceros = bundle.curvas.iloc[-1].to_numpy()
    y_corto = float(bundle.curvas["3M"].iat[-1])
    tier1 = cfg.BANK_PROFILE["tier1_musd"]

    sens = inst[inst["sensible"]].copy()
    activos = sens[sens["lado"] == "activo"]
    pasivos = sens[sens["lado"] == "pasivo"]

    # --- duración de activos
    dur_a = activos.apply(lambda f: _duracion_instrumento(f, tenores, ceros), axis=1)
    a_total = float(activos["saldo"].sum())
    d_activos = float((dur_a * activos["saldo"]).sum() / a_total)

    # --- duración de pasivos, tratando los NMD con el modelo de runoff
    num_runoff = num_efectiva = 0.0
    for _, f in pasivos.iterrows():
        if f["es_nmd"]:
            d_r, d_e = _duracion_nmd(cfg, f["categoria"], y_corto)
        else:
            d_r = d_e = _duracion_instrumento(f, tenores, ceros)
        num_runoff += d_r * f["saldo"]
        num_efectiva += d_e * f["saldo"]
    p_total = float(pasivos["saldo"].sum())
    d_pasivos = num_runoff / p_total
    d_pasivos_efectiva = num_efectiva / p_total

    gap_duracion = d_activos - d_pasivos * (p_total / a_total)

    # --- NIM sobre activos productivos
    productivos = inst[(inst["lado"] == "activo") & inst["sensible"]]
    ingreso = float((productivos["saldo"] * productivos["tasa"] * productivos["factor_devengo"]).sum())
    costo = float((pasivos["saldo"] * pasivos["tasa"]).sum())
    nim = (ingreso - costo) / float(productivos["saldo"].sum())

    # --- ΔNII 12m ante +200 pb, gap estático
    shock = 0.02
    def _beta_traslado(f):
        if f["es_nmd"]:
            return cfg.DEPOSIT_RATES[f["categoria"]]["beta_up"]
        return 1.0

    d_nii = 0.0
    for _, f in sens.iterrows():
        k = int(f["meses_a_repreciacion"])
        if k >= 12:
            continue
        fraccion = (12 - k) / 12.0
        impacto = f["saldo"] * shock * fraccion * f["factor_devengo"]
        if f["lado"] == "activo":
            d_nii += impacto
        else:
            d_nii -= f["saldo"] * shock * fraccion * _beta_traslado(f)
    nii_base = ingreso - costo
    delta_nii_pct = d_nii / nii_base

    # --- ΔEVE ante +200 pb, primer orden
    delta_eve = -shock * (d_activos * a_total - d_pasivos * p_total)
    delta_eve_sobre_tier1 = delta_eve / tier1

    return {
        "duracion_activos_sensibles_a": d_activos,
        "duracion_pasivos_sensibles_a": d_pasivos,
        "duracion_pasivos_efectiva_a": d_pasivos_efectiva,
        "gap_duracion_a": gap_duracion,
        "nim": nim,
        "nii_base_musd": nii_base,
        "delta_nii_12m_up200": delta_nii_pct,
        "delta_eve_up200_musd": delta_eve,
        "delta_eve_peor_sobre_tier1": delta_eve_sobre_tier1,
        "activos_sensibles_musd": a_total,
        "pasivos_sensibles_musd": p_total,
    }


# ---------------------------------------------------------------------------
# Controles 1 a 11
# ---------------------------------------------------------------------------

def _check_01_balance(bundle, cfg) -> Check:
    inst = bundle.instrumentos
    a = float(inst.loc[inst["lado"] == "activo", "saldo"].sum())
    pk = float(inst.loc[inst["lado"] != "activo", "saldo"].sum())
    err = abs(a - pk) / a
    tol = cfg.VALIDATION_THRESHOLDS["balance_tol_relativa"]
    return Check(
        1, "Activos = Pasivos + Patrimonio", err, tol, "ERROR", err <= tol,
        f"A = {a:,.2f} M; P+K = {pk:,.2f} M; error relativo {err:.2e}",
    )


def _check_02_series(bundle, cfg) -> Check:
    n_obj = cfg.VALIDATION_THRESHOLDS["n_meses_esperado"]
    problemas = []
    tablas = {
        "ref_rate": bundle.ref_rate.to_frame(),
        "curvas": bundle.curvas,
        "tasas_deposito": bundle.tasas_deposito,
        "saldos": bundle.saldos,
    }
    for nombre, t in tablas.items():
        if len(t) != n_obj:
            problemas.append(f"{nombre}: {len(t)} filas")
        if t.index.has_duplicates:
            problemas.append(f"{nombre}: fechas duplicadas")
        if t.isna().to_numpy().any():
            problemas.append(f"{nombre}: contiene NaN")
        esperado = pd.date_range(t.index[0], periods=len(t), freq=cfg.DATES["freq"])
        if not t.index.equals(esperado):
            problemas.append(f"{nombre}: fechas no consecutivas")
    return Check(
        2, "120 meses consecutivos, sin NaN ni duplicados", len(bundle.ref_rate), n_obj,
        "ERROR", not problemas, "; ".join(problemas) or "las cuatro series están completas",
    )


def _check_03_fechas(bundle, cfg) -> Check:
    inst = bundle.instrumentos
    mal_orden = int((inst["fecha_repreciacion"] > inst["fecha_vencimiento"]).sum())
    fijas = inst[inst["tipo_tasa"] == "fija"]
    mal_fija = int((fijas["fecha_repreciacion"] != fijas["fecha_vencimiento"]).sum())
    ok = mal_orden == 0 and mal_fija == 0
    return Check(
        3, "Repreciación ≤ vencimiento; fija ⇒ repricing = vencimiento",
        mal_orden + mal_fija, 0, "ERROR", ok,
        f"{mal_orden} con repreciación posterior al vencimiento; "
        f"{mal_fija} cohortes fijas cuyo repricing no coincide con el vencimiento",
    )


def _check_04_spreads(bundle, cfg) -> Check:
    inst = bundle.instrumentos
    pasivos = inst[(inst["lado"] == "pasivo") & inst["sensible"]]
    costo_fondeo = float((pasivos["saldo"] * pasivos["tasa"]).sum() / pasivos["saldo"].sum())
    activos = inst[(inst["lado"] == "activo") & inst["sensible"]]
    fallos = []
    peor = np.inf
    for cat, d in activos.groupby("categoria"):
        tasa = float((d["saldo"] * d["tasa"] * d["factor_devengo"]).sum() / d["saldo"].sum())
        peor = min(peor, tasa - costo_fondeo)
        if tasa <= costo_fondeo:
            fallos.append(f"{cat} {tasa:.2%} ≤ fondeo {costo_fondeo:.2%}")
    return Check(
        4, "Tasa activa por producto > costo de fondeo", peor * 1e4,
        cfg.VALIDATION_THRESHOLDS["spread_minimo_pb"], "ERROR", not fallos,
        "; ".join(fallos) or f"costo de fondeo {costo_fondeo:.2%}; menor spread {peor * 1e4:.0f} pb",
    )


def _check_05_curva_corta(bundle, cfg) -> Check:
    dif_pb = (bundle.curvas["3M"] - bundle.ref_rate).abs().max() * 1e4
    umbral = cfg.VALIDATION_THRESHOLDS["curva_3m_vs_ref_pb"]
    return Check(
        5, "Curva 3M dentro de ±25 pb de la referencia", float(dif_pb), umbral,
        "ERROR", dif_pb <= umbral, f"desvío máximo {dif_pb:.1f} pb sobre 120 fechas",
    )


def _check_06_descuento(bundle, cfg) -> Check:
    tenores = curve_tenors(cfg)
    taus = np.linspace(0.02, float(tenores[-1]), 300)
    peor_min, fechas_malas = 1.0, 0
    for _, fila in bundle.curvas.iterrows():
        dfs = discount_factors(tenores, fila.to_numpy(), taus)
        peor_min = min(peor_min, float(dfs.min()))
        if dfs.min() <= 0 or np.any(np.diff(dfs) >= 0):
            fechas_malas += 1
    return Check(
        6, "Factores de descuento positivos y decrecientes", fechas_malas, 0,
        "ERROR", fechas_malas == 0,
        f"{fechas_malas} fechas con DF no monótono; DF mínimo observado {peor_min:.4f}",
    )


def _regresion_rezagos(dd: np.ndarray, dr: np.ndarray, n_rezagos: int) -> tuple[float, float, float]:
    """Regresión de Δd sobre Δr y sus rezagos.

    Devuelve ``(Σβ̂, t de Σβ̂, R²)``. El estadístico t es el de la **suma** de
    coeficientes, no el del contemporáneo: la hipótesis que interesa es que el
    traspaso acumulado sea significativo, y se contrasta con
    ``Var(c'β̂) = c'·Var(β̂)·c`` para el vector ``c`` que suma los rezagos.
    """
    idx = np.arange(n_rezagos, len(dd))
    x = np.column_stack([np.ones(len(idx))] + [dr[idx - k] for k in range(n_rezagos)])
    y = dd[idx]
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    gl = len(idx) - x.shape[1]
    s2 = float(resid @ resid) / gl
    cov = s2 * np.linalg.inv(x.T @ x)
    c = np.zeros(x.shape[1])
    c[1:] = 1.0
    suma = float(c @ beta)
    t = suma / float(np.sqrt(c @ cov @ c))
    r2 = 1.0 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())
    return suma, t, r2


def _check_07_regresion(bundle, cfg) -> Check:
    dr = np.diff(bundle.ref_rate.to_numpy())
    umb_t = cfg.VALIDATION_THRESHOLDS["regresion_t_stat_min"]
    umb_r2 = cfg.VALIDATION_THRESHOLDS["regresion_r2_min"]
    n_rezagos = cfg.VALIDATION_THRESHOLDS["regresion_rezagos"]

    detalles, peor_t, peor_r2 = [], np.inf, np.inf
    for prod in ("vista", "ahorro", "plazo"):
        dd = np.diff(bundle.tasas_deposito[prod].to_numpy())
        suma, t, r2 = _regresion_rezagos(dd, dr, n_rezagos)
        peor_t, peor_r2 = min(peor_t, abs(t)), min(peor_r2, r2)
        contemporanea = stats.linregress(dr, dd)
        detalles.append(
            f"{prod}: Σβ̂={suma:.2f} (t={t:.1f}, R²={r2:.2f}); "
            f"contemporánea R²={contemporanea.rvalue ** 2:.2f}"
        )
    ok = peor_t > umb_t and peor_r2 > umb_r2
    return Check(
        7, f"Regresión Δd ~ Δr con {n_rezagos} rezagos: t > 3 y R² > 0,5",
        f"t≥{peor_t:.1f}, R²≥{peor_r2:.2f}", f"t>{umb_t}, R²>{umb_r2}", "ERROR", ok,
        "; ".join(detalles)
        + " | Se regresa sobre Δr y sus rezagos, no sólo sobre Δr contemporáneo: con "
        "ajuste parcial (λ=0,30 en vista) sólo un tercio de la respuesta ocurre en el "
        "mes del movimiento y el resto llega después, cuando Δr ya vale cero. Medido "
        "con ruido cero, la regresión contemporánea de vista topa en R²=0,50 — el "
        "umbral era inalcanzable por construcción del propio modelo de §6.4. Σβ̂ es el "
        "traspaso acumulado, que es el número que un ALCO pide.",
    )


def _check_08_beta_ols(bundle, cfg) -> Check:
    r = bundle.ref_rate.to_numpy()
    dr = np.diff(r, prepend=r[0])
    subidas = dr > 0
    bajadas = dr < 0
    n_mov = int(subidas.sum() + bajadas.sum())
    w_up = subidas.sum() / n_mov if n_mov else 0.5

    detalles, peor = [], 0.0
    for prod in ("vista", "ahorro", "plazo"):
        p = cfg.DEPOSIT_RATES[prod]
        referencia = w_up * p["beta_up"] + (1 - w_up) * p["beta_down"]
        beta_ols = stats.linregress(r, bundle.tasas_deposito[prod].to_numpy()).slope
        err = abs(beta_ols - referencia)
        peor = max(peor, err)
        detalles.append(f"{prod}: OLS {beta_ols:.2f} vs ponderada {referencia:.2f} (err {err:+.2f})")
    tol = cfg.VALIDATION_THRESHOLDS["beta_ols_tolerancia"]
    return Check(
        8, "Beta OLS ingenua ≈ promedio ponderado de β⁺/β⁻", peor, tol, "WARNING", peor <= tol,
        "; ".join(detalles)
        + f" | {w_up:.0%} de los movimientos fueron subidas. "
        "Se ESPERA que este control roce el umbral: una regresión simétrica sobre un "
        "proceso asimétrico está mal especificada. Que avise es información, no un defecto.",
    )


def _check_09_saldos(bundle, cfg) -> Check:
    umbral = cfg.VALIDATION_THRESHOLDS["salto_saldo_max"]
    exentos = set(cfg.VALIDATION_THRESHOLDS["meses_exentos_salto"])
    negativos = int((bundle.saldos <= 0).to_numpy().sum())
    saltos = []
    for col in bundle.saldos.columns:
        var = bundle.saldos[col].pct_change().abs()
        for i, v in enumerate(var.to_numpy()):
            if i + 1 in exentos or not np.isfinite(v):
                continue
            if v > umbral:
                saltos.append(f"{col} mes {i + 1}: {v:.1%}")
    ok = negativos == 0 and not saltos
    return Check(
        9, "Saldos > 0 y sin saltos > 15% fuera del estrés", len(saltos) + negativos, 0,
        "WARNING", ok,
        "; ".join(saltos[:5]) or f"{negativos} saldos no positivos; ningún salto anómalo "
        f"(la ventana del estrés, meses {sorted(exentos)}, está exenta por diseño)",
    )


def _check_10_calibracion(bundle, cfg, metricas: dict) -> list[Check]:
    checks = []
    for i, (clave, (lo, hi)) in enumerate(cfg.CALIBRATION_TARGETS.items()):
        valor = metricas.get(clave)
        if valor is None:
            continue
        ok = lo <= valor <= hi
        checks.append(
            Check(10, f"Calibración · {clave}", valor, (lo, hi), "WARNING", ok,
                  f"medido {valor:.4f}, objetivo [{lo}, {hi}]" + ("" if ok else "  ← fuera de rango"))
        )
    return checks


# ---------------------------------------------------------------------------
# Orquestación e informe
# ---------------------------------------------------------------------------

def run_all_checks(bundle, cfg, hash_repetido: str | None = None) -> list[Check]:
    """Corre los 11 controles de §6.8 sobre un dataset generado.

    Args:
        bundle: :class:`~src.data_gen.DatasetBundle` a validar.
        cfg: Configuración.
        hash_repetido: Hash de una segunda generación con la misma semilla. Si se
            omite, el control #11 se marca como no ejecutado (y por tanto fallido):
            un control de reproducibilidad que se saltea silenciosamente no es un
            control.

    Returns:
        Lista de :class:`Check` en el orden de la tabla de §6.8.
    """
    metricas = calibration_metrics(bundle, cfg)
    checks = [
        _check_01_balance(bundle, cfg),
        _check_02_series(bundle, cfg),
        _check_03_fechas(bundle, cfg),
        _check_04_spreads(bundle, cfg),
        _check_05_curva_corta(bundle, cfg),
        _check_06_descuento(bundle, cfg),
        _check_07_regresion(bundle, cfg),
        _check_08_beta_ols(bundle, cfg),
        _check_09_saldos(bundle, cfg),
    ]
    checks += _check_10_calibracion(bundle, cfg, metricas)

    propio = bundle.hash()
    ok11 = hash_repetido is not None and hash_repetido == propio
    checks.append(
        Check(11, "Reproducibilidad: misma semilla ⇒ mismo hash", propio[:16],
              (hash_repetido or "no ejecutado")[:16], "ERROR", ok11,
              "hashes idénticos" if ok11 else "no se comparó contra una segunda corrida")
    )
    return checks


def report(checks: list[Check], metricas: dict | None = None) -> str:
    """Informe legible de los controles, en el formato que iría al comité.

    Args:
        checks: Resultados de :func:`run_all_checks`.
        metricas: Métricas de :func:`calibration_metrics`, opcional.

    Returns:
        Texto en Markdown.
    """
    lineas = ["# Módulo 0 — Informe de validación", ""]
    if metricas:
        lineas += ["## Métricas de calibración", "", "| Métrica | Valor |", "|---|---|"]
        for k, v in metricas.items():
            fmt = f"{v:,.4f}" if abs(v) < 100 else f"{v:,.1f}"
            lineas.append(f"| {k} | {fmt} |")
        lineas.append("")

    lineas += ["## Controles", "", "| # | Control | Valor | Umbral | Sev. | Estado |", "|---|---|---|---|---|---|"]
    for c in checks:
        val = f"{c.valor:.4f}" if isinstance(c.valor, float) else str(c.valor)
        estado = "PASA" if c.passed else ("**FALLA**" if c.severidad == "ERROR" else "avisa")
        lineas.append(f"| {c.numero} | {c.nombre} | {val} | {c.umbral} | {c.severidad} | {estado} |")

    lineas += ["", "## Detalle", ""]
    for c in checks:
        if c.detalle:
            lineas.append(f"- **#{c.numero} {c.nombre}** — {c.detalle}")

    errores = [c for c in checks if not c.passed and c.severidad == "ERROR"]
    avisos = [c for c in checks if not c.passed and c.severidad == "WARNING"]
    lineas += [
        "",
        "## Veredicto",
        "",
        f"- ERROR fallidos: **{len(errores)}** (bloquean el avance al Módulo 1)",
        f"- WARNING activos: {len(avisos)} (informan, no bloquean)",
    ]
    return "\n".join(lineas)


def assert_no_errors(checks: list[Check]) -> None:
    """Levanta ``AssertionError`` si algún control de severidad ERROR falló.

    Args:
        checks: Resultados de :func:`run_all_checks`.

    Raises:
        AssertionError: Si hay al menos un ERROR fallido.
    """
    errores = [c for c in checks if not c.passed and c.severidad == "ERROR"]
    if errores:
        detalle = "\n".join(f"  #{c.numero} {c.nombre}: {c.detalle}" for c in errores)
        raise AssertionError(f"{len(errores)} control(es) ERROR fallaron:\n{detalle}")
