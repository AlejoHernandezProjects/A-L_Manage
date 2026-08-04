"""Módulo 1 — Balance sintético y brecha de repreciación.

El Módulo 0 dejó el inventario de cohortes. Este módulo lo lee por el eje que
importa en gestión de activos y pasivos: **cuándo repacta cada peso, no cuándo
vence**.

La distinción no es sutil. Sobre el mismo balance los dos ejes dan tablas distintas:
16.200 M de cartera comercial vencen entre 3 y 7 años pero repactan cada trimestre.
Clasificar por vencimiento haría ver largo un activo que es corto, y llevaría al
comité a cubrir un riesgo que no tiene mientras deja abierto el que sí tiene. El gap
de repreciación existe exactamente para no cometer ese error, y por eso este módulo
publica las dos tablas lado a lado.

**Advertencia que el informe repite y conviene tener presente al leer el código.** El
gap de repreciación contractual asigna los depósitos sin vencimiento a la banda más
corta, porque contractualmente el cliente retira mañana. Eso equivale a suponer que el
banco traslada el 100% de cualquier movimiento de tasas a esos depósitos. No es
cierto: la beta verdadera de vista es 0,25. El resultado es que el gap de este banco
dice "fuertemente pasivo-sensible" mientras el margen financiero mejora cuando suben
las tasas. Los dos números son correctos; el que está mal especificado es el gap.

Este módulo **no aplica ninguna beta ni supuesto conductual**. Es deliberado: la beta
se estima en el Módulo 2, y usarla aquí sería importar el resultado antes de
producirlo. Lo que el Módulo 1 entrega es la vista contractual y el diagnóstico de
por qué no basta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.curves import amortization_schedule

__all__ = [
    "bandas_frame",
    "asignar_banda",
    "perfil_repreciacion",
    "gap_repreciacion",
    "gap_vencimiento",
    "indicadores_gap",
    "reconciliacion",
    "informe_alco",
]


# ---------------------------------------------------------------------------
# Bandas
# ---------------------------------------------------------------------------

def bandas_frame(cfg) -> pd.DataFrame:
    """Las 19 bandas IRRBB como tabla, con su agrupamiento para el comité.

    Args:
        cfg: Configuración.

    Returns:
        DataFrame indexado por ``codigo`` con ``etiqueta``, ``mes_min``, ``mes_max``,
        ``punto_medio_a``, ``orden`` y ``banda_alco``.
    """
    filas = [
        {
            "codigo": c,
            "etiqueta": e,
            "mes_min": lo,
            "mes_max": hi,
            "punto_medio_a": pm,
            "orden": i,
        }
        for i, (c, e, lo, hi, pm) in enumerate(cfg.IRRBB_BANDS)
    ]
    df = pd.DataFrame(filas).set_index("codigo")
    mapa = {cod: alco for alco, codigos in cfg.BANDAS_ALCO.items() for cod in codigos}
    df["banda_alco"] = df.index.map(mapa)
    if df["banda_alco"].isna().any():
        faltan = list(df.index[df["banda_alco"].isna()])
        raise ValueError(f"BANDAS_ALCO no cubre las bandas: {faltan}")
    return df


def asignar_banda(meses, cfg) -> np.ndarray:
    """Asigna cada plazo (en meses) a su banda IRRBB.

    Los intervalos son **abiertos por abajo y cerrados por arriba**: un plazo de 3,0
    meses cae en ``1M–3M``, no en ``3M–6M``. La única excepción es la banda
    overnight, cerrada por ambos lados para que el plazo cero tenga dónde caer.

    Args:
        meses: Escalar o arreglo de plazos en meses.
        cfg: Configuración.

    Returns:
        Arreglo de códigos de banda.
    """
    bandas = bandas_frame(cfg)
    topes = bandas["mes_max"].to_numpy()
    codigos = bandas.index.to_numpy()
    m = np.asarray(meses, dtype=float)
    idx = np.searchsorted(topes, m, side="left")
    idx = np.clip(idx, 0, len(codigos) - 1)
    return codigos[idx]


# ---------------------------------------------------------------------------
# Reparto del saldo por banda
# ---------------------------------------------------------------------------

def _horizonte(fila, eje: str) -> int:
    """Meses hasta el evento que define la banda del instrumento."""
    if eje == "vencimiento":
        return int(fila["meses_a_vencimiento"])
    return int(fila["meses_a_repreciacion"])


def perfil_repreciacion(
    instrumentos: pd.DataFrame, cfg, metodo: str = "amortizado", eje: str = "repreciacion"
) -> pd.DataFrame:
    """Reparte el saldo de cada instrumento entre bandas temporales.

    Regla del método ``"amortizado"``::

        horizonte = meses a repreciación   (o a vencimiento, según el eje)
        el principal devuelto en t < horizonte  →  banda de t
        el saldo remanente en el horizonte      →  banda del horizonte

    **Por qué el amortizado y no el bullet.** Una hipoteca francesa a 20 años no
    expone el 100% de su saldo dentro de 20 años: devuelve principal todos los meses.
    Asignarla entera a la banda 15–20A sobrestima el plazo del activo de forma
    grosera. Con 16.200 M de cartera amortizable en este balance (hipotecario,
    consumo, vehículos) la diferencia entre los dos métodos es material, y por eso el
    informe muestra ambos.

    El método ``"bullet"`` asigna todo el saldo a la banda del horizonte. Es el
    tratamiento ingenuo, y se conserva como contraste explícito.

    Los instrumentos no sensibles (otros activos, otros pasivos, patrimonio) quedan
    **fuera**: no repactan nunca y meterlos en una banda sería inventar exposición.
    Aparecen en :func:`reconciliacion`.

    Args:
        instrumentos: Inventario del Módulo 0.
        cfg: Configuración.
        metodo: ``"amortizado"`` o ``"bullet"``.
        eje: ``"repreciacion"`` o ``"vencimiento"``.

    Returns:
        DataFrame largo con ``id``, ``categoria``, ``lado``, ``banda``,
        ``saldo_en_banda``.
    """
    if metodo not in ("amortizado", "bullet"):
        raise ValueError(f"método desconocido: {metodo!r}")
    if eje not in ("repreciacion", "vencimiento"):
        raise ValueError(f"eje desconocido: {eje!r}")

    sensibles = instrumentos[instrumentos["sensible"]]
    filas: list[dict] = []

    for id_, fila in sensibles.iterrows():
        horizonte = _horizonte(fila, eje)
        saldo = float(fila["saldo"])
        tramos: list[tuple[float, float]] = []

        if metodo == "amortizado" and fila["amortizacion"] == "frances":
            meses, principal = amortization_schedule(
                saldo,
                float(fila["tasa"]),
                int(fila["meses_a_vencimiento"]),
                str(fila["amortizacion"]),
                int(fila["frecuencia_pago_meses"]),
            )
            antes = meses < horizonte
            tramos.extend(zip(meses[antes], principal[antes]))
            remanente = saldo - float(principal[antes].sum())
            if remanente > 0:
                tramos.append((float(horizonte), remanente))
        else:
            tramos.append((float(horizonte), saldo))

        for mes, monto in tramos:
            filas.append(
                {
                    "id": id_,
                    "categoria": fila["categoria"],
                    "lado": fila["lado"],
                    "mes": mes,
                    "saldo_en_banda": monto,
                }
            )

    perfil = pd.DataFrame(filas)
    perfil["banda"] = asignar_banda(perfil["mes"].to_numpy(), cfg)
    return perfil


# ---------------------------------------------------------------------------
# Tablas de gap
# ---------------------------------------------------------------------------

def _tabla_gap(perfil: pd.DataFrame, cfg, agrupar: bool) -> pd.DataFrame:
    """Convierte un perfil largo en la tabla de gap por banda."""
    bandas = bandas_frame(cfg)
    pivote = (
        perfil.pivot_table(
            index="banda", columns="lado", values="saldo_en_banda", aggfunc="sum", observed=False
        )
        .reindex(bandas.index)
        .fillna(0.0)
    )
    for lado in ("activo", "pasivo"):
        if lado not in pivote.columns:
            pivote[lado] = 0.0

    tabla = pd.DataFrame(
        {
            "etiqueta": bandas["etiqueta"],
            "punto_medio_a": bandas["punto_medio_a"],
            "banda_alco": bandas["banda_alco"],
            "activos": pivote["activo"],
            "pasivos": pivote["pasivo"],
        }
    )

    if agrupar:
        orden = list(cfg.BANDAS_ALCO)
        tabla = (
            tabla.groupby("banda_alco", observed=False)[["activos", "pasivos"]]
            .sum()
            .reindex(orden)
            .fillna(0.0)
        )
        tabla.index.name = "banda"
    else:
        tabla = tabla.drop(columns=["banda_alco"])

    total_activos = cfg.BANK_PROFILE["activos_totales_musd"]
    tabla["gap"] = tabla["activos"] - tabla["pasivos"]
    tabla["gap_acumulado"] = tabla["gap"].cumsum()
    tabla["gap_pct_activos"] = tabla["gap"] / total_activos
    tabla["gap_acum_pct_activos"] = tabla["gap_acumulado"] / total_activos
    with np.errstate(divide="ignore", invalid="ignore"):
        tabla["rsa_rsl"] = np.where(tabla["pasivos"] > 0, tabla["activos"] / tabla["pasivos"], np.nan)
    return tabla


def gap_repreciacion(
    instrumentos: pd.DataFrame, cfg, metodo: str = "amortizado", agrupar: bool = False
) -> pd.DataFrame:
    """Brecha de repreciación: activos menos pasivos que repactan en cada banda.

    Es la tabla central del Módulo 1. Un gap positivo en una banda significa que en
    ese tramo repacta más activo que pasivo, de modo que una subida de tasas mejora
    el margen; uno negativo, lo contrario.

    Args:
        instrumentos: Inventario del Módulo 0.
        cfg: Configuración.
        metodo: ``"amortizado"`` (recomendado) o ``"bullet"``.
        agrupar: Si es ``True``, agrega a las 8 bandas de presentación al comité.

    Returns:
        DataFrame por banda con activos, pasivos, gap, gap acumulado y ratios.
    """
    perfil = perfil_repreciacion(instrumentos, cfg, metodo=metodo, eje="repreciacion")
    return _tabla_gap(perfil, cfg, agrupar)


def gap_vencimiento(
    instrumentos: pd.DataFrame, cfg, metodo: str = "amortizado", agrupar: bool = False
) -> pd.DataFrame:
    """La misma tabla clasificada por **vencimiento**, como contraste.

    No es una métrica de riesgo de tasa: es el error que el gap de repreciación
    existe para no cometer. Se publica al lado porque la diferencia entre las dos
    tablas se entiende mucho mejor vista que explicada — la cartera comercial
    aparece en 3–7 años aquí y en 1–3 meses allá, y es el mismo dinero.

    Sí es, en cambio, la tabla relevante para **riesgo de liquidez**, que es un
    problema distinto y no es el de este proyecto.

    Args:
        instrumentos: Inventario del Módulo 0.
        cfg: Configuración.
        metodo: ``"amortizado"`` o ``"bullet"``.
        agrupar: Si es ``True``, agrega a las 8 bandas de presentación.

    Returns:
        DataFrame por banda de vencimiento.
    """
    perfil = perfil_repreciacion(instrumentos, cfg, metodo=metodo, eje="vencimiento")
    return _tabla_gap(perfil, cfg, agrupar)


# ---------------------------------------------------------------------------
# Indicadores y reconciliación
# ---------------------------------------------------------------------------

def indicadores_gap(tabla: pd.DataFrame, cfg) -> dict:
    """Indicadores que un ALCO mira en la reunión, con su veredicto contra política.

    El más importante es el **gap acumulado a 12 meses sobre activos totales**: la
    porción del balance que quedaría expuesta a un movimiento de tasas dentro del
    horizonte de presupuesto. Se compara contra los umbrales de apetito de riesgo de
    ``GAP_THRESHOLDS``, que son internos, no regulatorios.

    Args:
        tabla: Salida de :func:`gap_repreciacion` **sin agrupar** (19 bandas).
        cfg: Configuración.

    Returns:
        Diccionario de indicadores.
    """
    bandas = bandas_frame(cfg)
    hasta_12m = bandas["mes_max"] <= 12.0
    total_activos = cfg.BANK_PROFILE["activos_totales_musd"]

    rsa = float(tabla.loc[hasta_12m, "activos"].sum())
    rsl = float(tabla.loc[hasta_12m, "pasivos"].sum())
    gap_12m = rsa - rsl
    ratio = gap_12m / total_activos

    umbrales = cfg.GAP_THRESHOLDS
    if abs(ratio) >= umbrales["gap_12m_sobre_activos_alerta"]:
        estado = "ALERTA"
    elif abs(ratio) >= umbrales["gap_12m_sobre_activos_aviso"]:
        estado = "AVISO"
    else:
        estado = "DENTRO DE POLÍTICA"

    lo, hi = umbrales["rsa_rsl_12m_banda"]
    rsa_rsl = rsa / rsl if rsl else float("nan")
    peor = tabla["gap"].abs().idxmax()

    return {
        "rsa_12m_musd": rsa,
        "rsl_12m_musd": rsl,
        "gap_acumulado_12m_musd": gap_12m,
        "gap_12m_sobre_activos": ratio,
        "estado_politica": estado,
        "rsa_rsl_12m": rsa_rsl,
        "rsa_rsl_dentro_de_banda": bool(lo <= rsa_rsl <= hi),
        "pct_activos_repactan_12m": rsa / total_activos,
        "pct_pasivos_repactan_12m": rsl / total_activos,
        "banda_mayor_descalce": str(peor),
        "gap_mayor_descalce_musd": float(tabla.loc[peor, "gap"]),
    }


def reconciliacion(instrumentos: pd.DataFrame, cfg) -> pd.DataFrame:
    """Puente del balance contable a los saldos sensibles del informe de gap.

    Un informe de gap que no cuadra con el balance es un informe que nadie audita.
    Esta tabla existe para que cualquiera pueda sumar y llegar a los 50.000 M.

    Args:
        instrumentos: Inventario del Módulo 0.
        cfg: Configuración.

    Returns:
        DataFrame con una fila por concepto y su saldo en USD M.
    """
    inst = instrumentos
    def _suma(lado, sensible=None):
        m = inst["lado"] == lado
        if sensible is not None:
            m &= inst["sensible"] == sensible
        return float(inst.loc[m, "saldo"].sum())

    activos_s, activos_ns = _suma("activo", True), _suma("activo", False)
    pasivos_s, pasivos_ns = _suma("pasivo", True), _suma("pasivo", False)
    patrimonio = _suma("patrimonio")

    filas = [
        ("Activos sensibles a tasa (RSA)", activos_s),
        ("Activos no sensibles", activos_ns),
        ("TOTAL ACTIVOS", activos_s + activos_ns),
        ("Pasivos sensibles a tasa (RSL)", pasivos_s),
        ("Pasivos no sensibles", pasivos_ns),
        ("Patrimonio", patrimonio),
        ("TOTAL PASIVO + PATRIMONIO", pasivos_s + pasivos_ns + patrimonio),
        ("Descuadre", (activos_s + activos_ns) - (pasivos_s + pasivos_ns + patrimonio)),
    ]
    return pd.DataFrame(filas, columns=["concepto", "musd"]).set_index("concepto")


# ---------------------------------------------------------------------------
# Informe
# ---------------------------------------------------------------------------

def _tabla_md(tabla: pd.DataFrame, columnas: dict[str, str]) -> list[str]:
    encabezado = "| Banda | " + " | ".join(columnas.values()) + " |"
    sep = "|---|" + "---|" * len(columnas)
    lineas = [encabezado, sep]
    for banda, fila in tabla.iterrows():
        celdas = []
        for col, _ in columnas.items():
            v = fila[col]
            if pd.isna(v):
                celdas.append("—")
            elif col.endswith("pct_activos"):
                celdas.append(f"{v:+.1%}")
            elif col == "rsa_rsl":
                celdas.append(f"{v:.2f}")
            else:
                celdas.append(f"{v:,.0f}")
        lineas.append(f"| {banda} | " + " | ".join(celdas) + " |")
    return lineas


def informe_alco(
    instrumentos: pd.DataFrame,
    cfg,
    delta_nii_modulo0: float | None = None,
) -> str:
    """Informe de brecha de repreciación en el formato que iría al comité.

    Args:
        instrumentos: Inventario del Módulo 0.
        cfg: Configuración.
        delta_nii_modulo0: ΔNII a 12 meses ante +200 pb medido en el Módulo 0. Si se
            entrega, el informe contrasta explícitamente el signo del gap contra el
            del margen — que es el punto del módulo.

    Returns:
        Texto en Markdown.
    """
    rep_alco = gap_repreciacion(instrumentos, cfg, agrupar=True)
    rep_19 = gap_repreciacion(instrumentos, cfg, agrupar=False)
    ven_alco = gap_vencimiento(instrumentos, cfg, agrupar=True)
    bullet_19 = gap_repreciacion(instrumentos, cfg, metodo="bullet", agrupar=False)
    ind = indicadores_gap(rep_19, cfg)
    ind_bullet = indicadores_gap(bullet_19, cfg)
    rec = reconciliacion(instrumentos, cfg)

    cols = {
        "activos": "Activos",
        "pasivos": "Pasivos",
        "gap": "Gap",
        "gap_acumulado": "Gap acum.",
        "gap_acum_pct_activos": "Gap acum. / activos",
        "rsa_rsl": "RSA/RSL",
    }

    L = ["# Módulo 1 — Brecha de repreciación", "", "Cifras en USD millones.", ""]

    L += ["## 1. Reconciliación con el balance", "", "| Concepto | USD M |", "|---|---|"]
    for concepto, fila in rec.iterrows():
        L.append(f"| {concepto} | {fila['musd']:,.0f} |")
    L += [
        "",
        "Los saldos no sensibles y el patrimonio quedan fuera de las tablas de gap: no "
        "repactan nunca, y asignarles una banda sería inventar exposición.",
        "",
    ]

    L += [
        "## 2. Gap de repreciación — clasificado por cuándo repacta",
        "",
        "Método de reparto: **amortizado**. El principal devuelto antes del horizonte de "
        "repreciación se asigna a la banda en que se cobra, y sólo el remanente va a la "
        "banda del repricing.",
        "",
    ]
    L += _tabla_md(rep_alco, cols)

    L += [
        "",
        "## 3. Contraste — la misma cartera clasificada por vencimiento",
        "",
        "Esta tabla **no** es una métrica de riesgo de tasa; es el error que el gap de "
        "repreciación existe para no cometer. Se publica al lado porque la diferencia se "
        "entiende mejor vista que explicada: la cartera comercial aparece en 2–5 años aquí "
        "y en 1–3 meses arriba, y es el mismo dinero. (Sí es la tabla relevante para "
        "riesgo de **liquidez**, que es otro problema.)",
        "",
    ]
    L += _tabla_md(ven_alco, cols)

    L += [
        "",
        "## 4. Indicadores contra política de ALM",
        "",
        "| Indicador | Valor | Referencia |",
        "|---|---|---|",
        f"| RSA a 12 meses | {ind['rsa_12m_musd']:,.0f} | — |",
        f"| RSL a 12 meses | {ind['rsl_12m_musd']:,.0f} | — |",
        f"| **Gap acumulado a 12 meses** | **{ind['gap_acumulado_12m_musd']:,.0f}** | — |",
        f"| Gap 12m / activos totales | {ind['gap_12m_sobre_activos']:+.1%} | "
        f"aviso {cfg.GAP_THRESHOLDS['gap_12m_sobre_activos_aviso']:.0%}, "
        f"alerta {cfg.GAP_THRESHOLDS['gap_12m_sobre_activos_alerta']:.0%} |",
        f"| Estado | **{ind['estado_politica']}** | — |",
        f"| RSA/RSL a 12 meses | {ind['rsa_rsl_12m']:.2f} | banda "
        f"{cfg.GAP_THRESHOLDS['rsa_rsl_12m_banda'][0]:.2f}–"
        f"{cfg.GAP_THRESHOLDS['rsa_rsl_12m_banda'][1]:.2f} |",
        f"| Banda de mayor descalce | {ind['banda_mayor_descalce']} | "
        f"{ind['gap_mayor_descalce_musd']:,.0f} M |",
        "",
        "Estos umbrales son de **apetito de riesgo interno**, no regulatorios. El único "
        "umbral que Basilea fija es el del *outlier test* de EVE (15% del Tier 1), que se "
        "evalúa en el Módulo 4.",
        "",
    ]

    L += [
        "## 5. Sensibilidad al método de reparto",
        "",
        "| Método | Gap acum. 12m | Sobre activos |",
        "|---|---|---|",
        f"| Amortizado | {ind['gap_acumulado_12m_musd']:,.0f} | "
        f"{ind['gap_12m_sobre_activos']:+.1%} |",
        f"| Bullet (ingenuo) | {ind_bullet['gap_acumulado_12m_musd']:,.0f} | "
        f"{ind_bullet['gap_12m_sobre_activos']:+.1%} |",
        "",
        "La diferencia sale de los 16.200 M de cartera amortizable. Bajo el método bullet "
        "una hipoteca a 20 años expone su saldo íntegro dentro de 20 años, cuando en "
        "realidad devuelve principal todos los meses.",
        "",
    ]

    L += ["## 6. Lectura del resultado", ""]
    if delta_nii_modulo0 is not None:
        L += [
            f"El gap acumulado a 12 meses es **{ind['gap_12m_sobre_activos']:+.1%}** de los "
            f"activos. Leído literalmente, un gap negativo de esta magnitud predice que el "
            f"margen financiero **cae** cuando suben las tasas.",
            "",
            f"El diagnóstico del Módulo 0 dice lo contrario: ante +200 pb el NII a 12 meses "
            f"**mejora {delta_nii_modulo0:+.2%}**.",
            "",
            "**Los dos números son correctos. El que está mal especificado es el gap.**",
            "",
            "La causa es identificable y está en una sola línea del tratamiento: los "
            "depósitos a la vista y de ahorro —22.000 M, el 44% del fondeo— entran en la "
            "banda más corta porque contractualmente el cliente puede retirar mañana. Eso "
            "equivale a suponer que el banco traslada el **100%** de cualquier movimiento "
            "de tasas a esos depósitos. No lo hace: traslada del orden de una cuarta parte "
            "en las cuentas transaccionales.",
            "",
            "El gap contractual está midiendo un banco que no existe. Ésta es, "
            "históricamente, la razón por la que la industria dejó de usar el análisis de "
            "brechas como herramienta única y pasó a simulación completa del margen.",
            "",
            "Qué resuelve cada módulo siguiente:",
            "",
            "- **Módulo 2** estima la beta y el perfil de runoff del core, y reasigna los "
            "NMD a bandas según comportamiento en vez de según contrato. Con eso el gap "
            "deja de estar mal especificado.",
            "- **Módulo 3** proyecta el margen a 12 meses aplicando esa beta, en lugar de "
            "inferirlo del signo de una brecha.",
            "",
            "Nótese que el problema es doble y el Módulo 2 arregla las dos mitades: la "
            "**cuantía** del traspaso (la beta) y el **momento** en que ocurre (el plazo "
            "conductual del core). Corregir sólo la primera dejaría el EVE del Módulo 4 "
            "igual de mal.",
        ]
    else:
        L.append(
            f"Gap acumulado a 12 meses: {ind['gap_12m_sobre_activos']:+.1%} de los activos."
        )

    return "\n".join(L)
