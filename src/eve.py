"""Módulo 4 — Valor económico del patrimonio (EVE).

    EVE = VP(activos) − VP(pasivos)

El Módulo 3 mostró que el margen mejora cuando suben las tasas. Este módulo mide la
otra mitad, y es la que da sentido al proyecto: **el mismo choque deteriora el valor
económico**. Un banco puede estar ganando margen este año mientras destruye valor a
largo plazo, y con una sola métrica no se ve. Por eso Basilea exige las dos.

**Convención de valoración: calibración a la par.** Para cada instrumento se resuelve
el spread constante ``s`` tal que ``VP(flujos, curva + s) = saldo`` en la fecha de
corte. Tres consecuencias, y las tres importan:

1. ``EVE(t=0)`` reproduce exactamente el valor en libros. La convención no inventa
   plusvalías ni minusvalías latentes.
2. El choque mueve **sólo la parte libre de riesgo**, así que el ΔEVE es riesgo de
   tasa puro y no una mezcla con riesgo de crédito.
3. El spread capturado —el margen comercial de cada producto— queda explícito en vez
   de escondido en la curva de descuento.

Descontar flujos con spread de crédito usando una curva libre de riesgo inflaría el
activo muy por encima de su valor en libros y haría que el ΔEVE midiera dos riesgos a
la vez. Es un error frecuente y difícil de detectar, porque el número resultante
*parece* razonable.

**Los NMD se valoran con el modelo del Módulo 2**, no con su contrato. Ahí cobra todo
lo anterior: un depósito a la vista tratado como exigible mañana vale su nominal y no
aporta nada al descalce; tratado con su perfil conductual, su valor cae cuando suben
las tasas y amortigua el deterioro del EVE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src.curves import amortization_schedule, interpolate_zero, schedule_cashflows
from src.data_gen import curve_tenors
from src.scenarios import ESCENARIOS, ETIQUETAS, aplicar_escenario

__all__ = [
    "cartera_valorable",
    "valor_presente",
    "eve",
    "tabla_delta_eve",
    "duracion_convexidad",
    "tabla_aproximaciones",
    "sensibilidad_eve",
    "informe_eve",
]

_H_BUMP = 1e-4  # 1 pb, para las derivadas numéricas


# ---------------------------------------------------------------------------
# Preparación de la cartera
# ---------------------------------------------------------------------------

def _flujos(fila, cfg) -> tuple[np.ndarray, np.ndarray]:
    """Flujos de **repreciación** de un instrumento, en años y montos.

    La distinción con los flujos contractuales es la que decide todo el módulo. Un
    crédito comercial a cinco años que repacta cada trimestre **no tiene duración de
    cinco años**: en cada reset vuelve a valer su nominal, porque su cupón se reajusta
    al mercado. Su exposición al riesgo de tasa se agota en el próximo repricing.

    Por eso el horizonte de los flujos es:

    * el **vencimiento** para los instrumentos a tasa fija;
    * la **próxima repreciación** para los variables y administrados, con el principal
      amortizado antes del reset en sus fechas y el saldo vivo restante al reset.

    Es exactamente la misma regla que usa el gap del Módulo 1
    (:func:`~src.balance.perfil_repreciacion`), y esa coherencia no es opcional: si el
    gap y el EVE clasificaran el mismo instrumento con horizontes distintos, las dos
    métricas del informe describirían balances diferentes.

    Tratar los 16.200 M de cartera comercial variable como si fueran bonos a cinco años
    infla la duración del activo cerca de un año entero y triplica el ΔEVE. El número
    resultante *parece* razonable, que es lo que hace peligroso el error.
    """
    venc = int(fila["meses_a_vencimiento"])
    if not fila["sensible"] or venc <= 0:
        return np.array([]), np.array([])

    saldo = float(fila["saldo"])
    tasa = float(fila["tasa"])
    freq = int(fila["frecuencia_pago_meses"])
    amort = str(fila["amortizacion"])

    es_fija = fila["tipo_tasa"] == "fija"
    horizonte = venc if es_fija else int(fila["meses_a_repreciacion"])
    horizonte = max(1, min(horizonte, venc))

    if amort != "frances":
        return schedule_cashflows(saldo, tasa, horizonte, "bullet", freq)

    tiempos, montos = schedule_cashflows(saldo, tasa, venc, "frances", freq)
    dentro = tiempos * 12.0 <= horizonte + 1e-9
    tiempos, montos = tiempos[dentro], montos[dentro].copy()

    meses_p, principal = amortization_schedule(saldo, tasa, venc, "frances", freq)
    vivo = saldo - float(principal[meses_p <= horizonte + 1e-9].sum())
    if vivo <= 1e-12:
        return tiempos, montos

    if len(tiempos) and abs(tiempos[-1] * 12.0 - horizonte) < 1e-9:
        montos[-1] += vivo
        return tiempos, montos
    return np.append(tiempos, horizonte / 12.0), np.append(montos, vivo)


def _vp(tiempos, montos, tenores, ceros, spread: float) -> float:
    if len(tiempos) == 0:
        return 0.0
    y = interpolate_zero(tenores, ceros, tiempos) + spread
    return float(np.sum(montos * np.exp(-y * tiempos)))


def cartera_valorable(instrumentos: pd.DataFrame, ceros, cfg) -> list[dict]:
    """Prepara la cartera para valorar: flujos y spread calibrado a la par.

    El patrimonio queda fuera: el EVE **es** el valor del patrimonio económico, así
    que incluirlo entre los pasivos sería contarlo dos veces. Los instrumentos no
    sensibles entran a valor en libros, porque no responden a la curva.

    El spread se calibra una sola vez sobre la curva base y se reutiliza en los seis
    escenarios. Eso es precisamente lo que hace que el ΔEVE aísle el riesgo de tasa:
    el margen comercial se mantiene fijo y sólo se mueve el suelo libre de riesgo.

    Args:
        instrumentos: Inventario, contractual o conductual.
        ceros: Curva cero de la fecha de corte.
        cfg: Configuración.

    Returns:
        Lista de diccionarios con flujos, spread, saldo y lado.
    """
    tenores = curve_tenors(cfg)
    ceros = np.asarray(ceros, dtype=float)
    cartera = []

    for id_, fila in instrumentos.iterrows():
        if fila["lado"] == "patrimonio":
            continue
        saldo = float(fila["saldo"])
        tiempos, montos = _flujos(fila, cfg)

        if len(tiempos) == 0:
            cartera.append(
                {"id": id_, "lado": fila["lado"], "categoria": fila["categoria"],
                 "tiempos": tiempos, "montos": montos, "spread": 0.0,
                 "saldo": saldo, "sensible": False}
            )
            continue

        def _error(s, t=tiempos, m=montos):
            return _vp(t, m, tenores, ceros, s) - saldo

        try:
            spread = brentq(_error, -0.50, 2.00, xtol=1e-12, rtol=1e-14)
        except ValueError:  # pragma: no cover - sólo con datos degenerados
            spread = 0.0

        cartera.append(
            {"id": id_, "lado": fila["lado"], "categoria": fila["categoria"],
             "tiempos": tiempos, "montos": montos, "spread": spread,
             "saldo": saldo, "sensible": True}
        )
    return cartera


# ---------------------------------------------------------------------------
# Valoración
# ---------------------------------------------------------------------------

def valor_presente(cartera: list[dict], ceros, cfg) -> pd.Series:
    """Valor presente de cada instrumento bajo una curva dada.

    Args:
        cartera: Salida de :func:`cartera_valorable`.
        ceros: Curva cero a usar.
        cfg: Configuración.

    Returns:
        Serie de valores presentes indexada por id.
    """
    tenores = curve_tenors(cfg)
    ceros = np.asarray(ceros, dtype=float)
    valores = {}
    for it in cartera:
        valores[it["id"]] = (
            _vp(it["tiempos"], it["montos"], tenores, ceros, it["spread"])
            if it["sensible"] else it["saldo"]
        )
    return pd.Series(valores, name="vp")


def eve(cartera: list[dict], ceros, cfg) -> dict:
    """EVE bajo una curva: valor presente de activos menos el de pasivos.

    Args:
        cartera: Salida de :func:`cartera_valorable`.
        ceros: Curva cero a usar.
        cfg: Configuración.

    Returns:
        Diccionario con ``eve``, ``vp_activos``, ``vp_pasivos`` y el desglose por
        categoría.
    """
    vp = valor_presente(cartera, ceros, cfg)
    lados = pd.Series({it["id"]: it["lado"] for it in cartera})
    cats = pd.Series({it["id"]: it["categoria"] for it in cartera})

    vp_a = float(vp[lados == "activo"].sum())
    vp_p = float(vp[lados == "pasivo"].sum())
    por_cat = vp.groupby([lados, cats], observed=False).sum()
    por_cat.index.names = ["lado", "categoria"]

    return {"eve": vp_a - vp_p, "vp_activos": vp_a, "vp_pasivos": vp_p, "por_categoria": por_cat}


def tabla_delta_eve(cartera: list[dict], ceros, cfg, escenarios=None) -> pd.DataFrame:
    """ΔEVE bajo cada escenario, contra el capital de nivel 1.

    El **peor de los seis** es el resultado que se compara con el umbral de alerta
    supervisora del 15% del Tier 1 — el *outlier test*. No es un límite duro que
    prohíba operar: es el disparador que obliga al banco a explicarse ante el
    supervisor, y en la práctica a traer un plan.

    Args:
        cartera: Salida de :func:`cartera_valorable`.
        ceros: Curva cero base.
        cfg: Configuración.
        escenarios: Subconjunto; por defecto los seis.

    Returns:
        DataFrame por escenario con EVE, ΔEVE, ΔEVE/Tier 1 y si supera el umbral.
    """
    tenores = curve_tenors(cfg)
    base = eve(cartera, ceros, cfg)
    tier1 = cfg.BANK_PROFILE["tier1_musd"]

    filas = []
    for nombre in (escenarios if escenarios is not None else ESCENARIOS):
        ceros_e = aplicar_escenario(ceros, tenores, nombre, cfg)
        r = eve(cartera, ceros_e, cfg)
        delta = r["eve"] - base["eve"]
        filas.append(
            {
                "escenario": nombre,
                "etiqueta": ETIQUETAS[nombre],
                "eve_musd": r["eve"],
                "delta_musd": delta,
                "delta_sobre_tier1": delta / tier1,
                "supera_umbral": delta / tier1 <= -0.15,
            }
        )
    tabla = pd.DataFrame(filas).set_index("escenario")
    tabla.attrs["eve_base"] = base["eve"]
    return tabla


# ---------------------------------------------------------------------------
# Duración, convexidad y los límites de la aproximación lineal
# ---------------------------------------------------------------------------

def duracion_convexidad(cartera: list[dict], ceros, cfg) -> dict:
    """Duración y convexidad **efectivas**, por diferencias finitas.

    Se calculan desplazando la curva ±1 pb y midiendo la respuesta del valor
    presente, en lugar de con la fórmula cerrada de Macaulay::

        D = −(VP₊ − VP₋) / (2·h·VP)        C = (VP₊ + VP₋ − 2·VP) / (h²·VP)

    **Por qué efectivas y no analíticas.** La duración analítica supone que los flujos
    no cambian con la tasa. Aquí ya no es cierto para los NMD, cuyo perfil sale de un
    modelo de comportamiento, y dejaría de serlo del todo si el proyecto incorporara
    prepago. La duración efectiva mide lo que realmente pasa cuando se mueve la curva,
    que es lo que un comité necesita.

    Args:
        cartera: Salida de :func:`cartera_valorable`.
        ceros: Curva cero base.
        cfg: Configuración.

    Returns:
        Diccionario con duración y convexidad de activos y pasivos, sus valores
        presentes y la duración del gap ajustada por apalancamiento.
    """
    tenores = curve_tenors(cfg)
    ceros = np.asarray(ceros, dtype=float)
    lados = pd.Series({it["id"]: it["lado"] for it in cartera})

    def _vps(despl):
        vp = valor_presente(cartera, ceros + despl, cfg)
        return float(vp[lados == "activo"].sum()), float(vp[lados == "pasivo"].sum())

    a0, p0 = _vps(0.0)
    a_up, p_up = _vps(_H_BUMP)
    a_dn, p_dn = _vps(-_H_BUMP)

    def _derivadas(v0, v_up, v_dn):
        """Duración y convexidad efectivas; cero si el lado está vacío."""
        if abs(v0) < 1e-12:
            return 0.0, 0.0
        return (
            -(v_up - v_dn) / (2 * _H_BUMP * v0),
            (v_up + v_dn - 2 * v0) / (_H_BUMP ** 2 * v0),
        )

    d_a, c_a = _derivadas(a0, a_up, a_dn)
    d_p, c_p = _derivadas(p0, p_up, p_dn)

    return {
        "vp_activos": a0,
        "vp_pasivos": p0,
        "duracion_activos": d_a,
        "duracion_pasivos": d_p,
        "convexidad_activos": c_a,
        "convexidad_pasivos": c_p,
        "gap_duracion": d_a - d_p * (p0 / a0) if abs(a0) > 1e-12 else 0.0,
    }


def tabla_aproximaciones(cartera: list[dict], ceros, cfg, choques_pb=(50, 100, 200, 400, 800)) -> pd.DataFrame:
    """Revaluación completa contra las aproximaciones de primer y segundo orden.

    Para un choque paralelo de tamaño ``Δy``::

        primer orden   = −(D_A·A − D_L·L) · Δy
        segundo orden  = primer orden + ½ · (C_A·A − C_L·L) · Δy²

    El error de la aproximación lineal crece con el **cuadrado** del choque, así que
    para ±50 pb es despreciable y para ±800 pb es grande. Ése es el argumento
    cuantitativo de por qué un límite de ALM expresado sólo en duración es insuficiente
    para escenarios de estrés: mide bien el riesgo del día a día y subestima
    precisamente el que motiva tener límites.

    Args:
        cartera: Salida de :func:`cartera_valorable`.
        ceros: Curva cero base.
        cfg: Configuración.
        choques_pb: Magnitudes paralelas a evaluar, en pb (se prueban en ambos signos).

    Returns:
        DataFrame con la revaluación completa, ambas aproximaciones y sus errores.
    """
    dc = duracion_convexidad(cartera, ceros, cfg)
    ceros = np.asarray(ceros, dtype=float)
    base = eve(cartera, ceros, cfg)["eve"]

    dur_neta = dc["duracion_activos"] * dc["vp_activos"] - dc["duracion_pasivos"] * dc["vp_pasivos"]
    conv_neta = dc["convexidad_activos"] * dc["vp_activos"] - dc["convexidad_pasivos"] * dc["vp_pasivos"]

    filas = []
    for pb in choques_pb:
        for signo in (1, -1):
            dy = signo * pb / 1e4
            completa = eve(cartera, ceros + dy, cfg)["eve"] - base
            orden1 = -dur_neta * dy
            orden2 = orden1 + 0.5 * conv_neta * dy ** 2
            filas.append(
                {
                    "choque_pb": signo * pb,
                    "revaluacion_completa": completa,
                    "primer_orden": orden1,
                    "segundo_orden": orden2,
                    "error_primer_orden": orden1 - completa,
                    "error_segundo_orden": orden2 - completa,
                    "error_relativo_primer_orden": (orden1 - completa) / abs(completa) if completa else np.nan,
                }
            )
    return pd.DataFrame(filas).sort_values("choque_pb").set_index("choque_pb")


# ---------------------------------------------------------------------------
# Sensibilidad
# ---------------------------------------------------------------------------

def sensibilidad_eve(bundle, cfg, modelo_base: dict) -> dict:
    """Barrido sobre los supuestos que el EVE hereda del Módulo 2.

    El EVE es mucho más sensible que el NII al plazo conductual de los NMD, porque
    descuenta a veinte años en vez de proyectar doce meses. Y ese plazo es justo el
    parámetro que el Módulo 2 demostró **no identificable** desde el saldo agregado.

    De ahí que el resultado que interesa no sea un ΔEVE puntual sino si la conclusión
    regulatoria —¿outlier o no?— aguanta en todo el rango del supuesto. Si cambiara de
    veredicto dentro del rango, el banco tendría que decir eso al supervisor, no elegir
    la punta que le conviene.

    Args:
        bundle: :class:`~src.data_gen.DatasetBundle`.
        cfg: Configuración.
        modelo_base: Salida de :func:`~src.deposits.modelo_nmd`.

    Returns:
        Diccionario de DataFrames: ``vida``, ``estable``, ``beta`` y ``tratamiento``.
    """
    from src.deposits import PRODUCTOS_NMD, instrumentos_conductuales, modelo_nmd

    ceros = bundle.curvas.iloc[-1].to_numpy()
    tier1 = cfg.BANK_PROFILE["tier1_musd"]

    def _peor(instrumentos):
        cartera = cartera_valorable(instrumentos, ceros, cfg)
        tabla = tabla_delta_eve(cartera, ceros, cfg)
        peor = tabla["delta_sobre_tier1"].idxmin()
        return {
            "eve_base_musd": tabla.attrs["eve_base"],
            "peor_escenario": ETIQUETAS[peor],
            "peor_delta_musd": tabla.loc[peor, "delta_musd"],
            "peor_sobre_tier1": tabla.loc[peor, "delta_sobre_tier1"],
            "supera_umbral": bool(tabla.loc[peor, "supera_umbral"]),
            "delta_paralelo_arriba": tabla.loc["paralelo_arriba", "delta_sobre_tier1"],
        }

    filas_vida = []
    for vida in cfg.NMD_ESTIMATION["sensibilidad_vida_a"]:
        m = modelo_nmd(bundle, cfg, vida_supuesta={p: vida for p in PRODUCTOS_NMD})
        inst = instrumentos_conductuales(bundle.instrumentos, m, cfg)
        filas_vida.append({
            "vida_supuesta_a": vida,
            "vida_aplicada_vista_a": m["nucleos"]["vista"]["vida_a"],
            "tope_muerde": any(n["tope_vida_muerde"] for n in m["nucleos"].values()),
            **_peor(inst),
        })

    filas_estable = []
    for delta in cfg.NMD_ESTIMATION["sensibilidad_estable_delta"]:
        m = modelo_nmd(bundle, cfg, ajuste_estable=delta)
        inst = instrumentos_conductuales(bundle.instrumentos, m, cfg)
        filas_estable.append({"ajuste_estable": delta, **_peor(inst)})

    filas_beta = []
    for clave in ("beta_up", "beta_largo"):
        m = modelo_nmd(bundle, cfg, beta_clave=clave)
        inst = instrumentos_conductuales(bundle.instrumentos, m, cfg)
        filas_beta.append({"beta_usada": clave, **_peor(inst)})

    inst_cond = instrumentos_conductuales(bundle.instrumentos, modelo_base, cfg)
    tratamiento = pd.DataFrame([
        {"tratamiento_nmd": "contractual (Módulo 1)", **_peor(bundle.instrumentos)},
        {"tratamiento_nmd": "conductual (Módulo 2)", **_peor(inst_cond)},
    ]).set_index("tratamiento_nmd")

    return {
        "vida": pd.DataFrame(filas_vida).set_index("vida_supuesta_a"),
        "estable": pd.DataFrame(filas_estable).set_index("ajuste_estable"),
        "beta": pd.DataFrame(filas_beta).set_index("beta_usada"),
        "tratamiento": tratamiento,
    }


# ---------------------------------------------------------------------------
# Informe
# ---------------------------------------------------------------------------

def sensibilidad_mix_hipotecario(cfg, shares=None) -> pd.DataFrame:
    """§11.1: ¿con qué mix de balance este banco se vuelve *outlier*?

    Es el barrido que quedó declarado en ``SENSITIVITY_GRID`` desde el Módulo 0, y la
    razón por la que allí se decidió **no** subir el mix hipotecario para acercar el
    banco al umbral: los objetivos de calibración se pre-registraron, y mover el
    balance hasta que el número saliera dramático es exactamente lo que esa
    pre-registración existe para impedir. El 25% se muestra como sensibilidad.

    Para mantener el balance cuadrado, lo que gana el hipotecario lo cede el crédito
    comercial. No es un artificio: es el traslado que un banco haría de verdad si
    decidiera crecer en minorista de largo plazo, y es justo el que más cambia su
    perfil de riesgo de tasa — cambia cartera indexada que repacta cada trimestre por
    cartera fija a quince años.

    Args:
        cfg: Configuración base.
        shares: Participaciones de hipotecario a probar; por defecto las de
            ``SENSITIVITY_GRID``.

    Returns:
        DataFrame por participación con duración, peor escenario y ΔEVE / Tier 1.
    """
    from config.params import make_config
    from src.data_gen import generate_dataset
    from src.deposits import instrumentos_conductuales, modelo_nmd

    shares = list(shares if shares is not None else cfg.SENSITIVITY_GRID["share_hipotecario"])
    base_hip = cfg.INSTRUMENT_SPECS["hipotecario"]["share"]
    base_com = cfg.INSTRUMENT_SPECS["comercial"]["share"]

    filas = []
    for s in shares:
        cfg_s = make_config(**{
            "INSTRUMENT_SPECS.hipotecario.share": s,
            "INSTRUMENT_SPECS.comercial.share": base_com - (s - base_hip),
        })
        b = generate_dataset(cfg_s)
        m = modelo_nmd(b, cfg_s)
        ceros = b.curvas.iloc[-1].to_numpy()
        cart = cartera_valorable(instrumentos_conductuales(b.instrumentos, m, cfg_s), ceros, cfg_s)
        tabla = tabla_delta_eve(cart, ceros, cfg_s)
        dc = duracion_convexidad(cart, ceros, cfg_s)
        peor = tabla["delta_sobre_tier1"].idxmin()
        filas.append({
            "share_hipotecario": s,
            "share_comercial": cfg_s.INSTRUMENT_SPECS["comercial"]["share"],
            "duracion_activos": dc["duracion_activos"],
            "gap_duracion": dc["gap_duracion"],
            "peor_escenario": ETIQUETAS[peor],
            "peor_sobre_tier1": tabla.loc[peor, "delta_sobre_tier1"],
            "supera_umbral": bool(tabla.loc[peor, "supera_umbral"]),
        })
    return pd.DataFrame(filas).set_index("share_hipotecario")


def informe_eve(bundle, cfg, modelo: dict, tabla_nii: pd.DataFrame | None = None) -> str:
    """Informe de valor económico en formato de comité.

    Args:
        bundle: :class:`~src.data_gen.DatasetBundle`.
        cfg: Configuración.
        modelo: Salida de :func:`~src.deposits.modelo_nmd`.
        tabla_nii: Salida de :func:`~src.nii.tabla_delta_nii`, para la tabla conjunta.

    Returns:
        Texto en Markdown.
    """
    from src.deposits import instrumentos_conductuales

    ceros = bundle.curvas.iloc[-1].to_numpy()
    inst = instrumentos_conductuales(bundle.instrumentos, modelo, cfg)
    cartera = cartera_valorable(inst, ceros, cfg)

    tabla = tabla_delta_eve(cartera, ceros, cfg)
    dc = duracion_convexidad(cartera, ceros, cfg)
    aprox = tabla_aproximaciones(cartera, ceros, cfg)
    sens = sensibilidad_eve(bundle, cfg, modelo)

    tier1 = cfg.BANK_PROFILE["tier1_musd"]
    umbral = 0.15 * tier1
    peor = tabla["delta_sobre_tier1"].idxmin()

    L = [
        "# Módulo 4 — Valor económico del patrimonio (EVE)",
        "",
        f"EVE base **{tabla.attrs['eve_base']:,.0f} M** · Tier 1 **{tier1:,.0f} M** · "
        f"umbral del *outlier test* **−{umbral:,.0f} M (15%)**. Cifras en USD millones.",
        "",
        "Valoración con **calibración a la par**: para cada instrumento se resuelve el "
        "spread que hace que su valor presente iguale su saldo en libros hoy. Así el "
        "EVE de partida no inventa plusvalías latentes y el choque mueve sólo la parte "
        "libre de riesgo — el ΔEVE es riesgo de tasa puro, no una mezcla con crédito.",
        "",
        "## 1. ΔEVE bajo los seis escenarios",
        "",
        "| Escenario | EVE | ΔEVE | ΔEVE / Tier 1 | ¿Outlier? |",
        "|---|---|---|---|---|",
    ]
    for nombre, f in tabla.iterrows():
        marca = " ⟵ **peor**" if nombre == peor else ""
        L.append(
            f"| {f['etiqueta']}{marca} | {f['eve_musd']:,.0f} | {f['delta_musd']:+,.0f} | "
            f"**{f['delta_sobre_tier1']:+.1%}** | {'**SÍ**' if f['supera_umbral'] else 'no'} |"
        )

    peor_f = tabla.loc[peor]
    L += [
        "",
        f"**Resultado headline: {peor_f['etiqueta']}, {peor_f['delta_musd']:+,.0f} M = "
        f"{peor_f['delta_sobre_tier1']:+.1%} del Tier 1.**",
        "",
        "El *outlier test* no es una prohibición: es el disparador que obliga al banco a "
        "explicarse ante el supervisor y, en la práctica, a traer un plan. Un banco puede "
        "operar por encima del umbral si sabe defender por qué.",
        "",
    ]

    if tabla_nii is not None:
        L += [
            "## 2. La tabla que va al comité — NII y EVE juntos",
            "",
            "| Escenario | ΔNII 12m | ΔEVE / Tier 1 |",
            "|---|---|---|",
        ]
        for nombre, f in tabla.iterrows():
            L.append(
                f"| {f['etiqueta']} | {tabla_nii.loc[nombre, 'delta_pct']:+.2%} | "
                f"{f['delta_sobre_tier1']:+.1%} |"
            )
        arriba_nii = tabla_nii.loc["paralelo_arriba", "delta_pct"]
        arriba_eve = tabla.loc["paralelo_arriba", "delta_sobre_tier1"]
        L += [
            "",
            f"**Ésta es la tensión que el proyecto existe para demostrar.** Ante +200 pb el "
            f"margen a doce meses **mejora {arriba_nii:+.2%}** y el valor económico "
            f"**se deteriora {arriba_eve:+.1%} del Tier 1**.",
            "",
            "No es una contradicción: son dos horizontes. El NII mira doce meses y ahí manda "
            "quién repacta antes; el EVE descuenta toda la vida de los flujos y ahí manda "
            "quién tiene más duración. Este banco cobra rápido en el activo indexado y paga "
            "despacio en sus depósitos —gana margen— mientras carga 9.000 M de hipotecas a "
            "tasa fija originadas cuando la referencia estaba en 1,5% —pierde valor—.",
            "",
            "Gestionar sólo por NII llevaría a este banco a celebrar una subida de tasas que "
            "le está destruyendo patrimonio económico. Que las dos métricas estén en la misma "
            "tabla, calculadas sobre los mismos seis escenarios definidos una sola vez, es "
            "todo el argumento.",
            "",
        ]

    L += [
        "## 3. Duración, convexidad y dónde falla lo lineal",
        "",
        f"Duración efectiva del activo **{dc['duracion_activos']:.2f} a**, del pasivo "
        f"**{dc['duracion_pasivos']:.2f} a**, gap ajustado por apalancamiento "
        f"**{dc['gap_duracion']:+.2f} a**. Convexidad del activo "
        f"{dc['convexidad_activos']:.1f}, del pasivo {dc['convexidad_pasivos']:.1f}.",
        "",
        "| Choque | Revaluación completa | Primer orden | Segundo orden | Error lineal |",
        "|---|---|---|---|---|",
    ]
    for pb, f in aprox.iterrows():
        L.append(
            f"| {pb:+,} pb | {f['revaluacion_completa']:+,.0f} | {f['primer_orden']:+,.0f} | "
            f"{f['segundo_orden']:+,.0f} | {f['error_relativo_primer_orden']:+.1%} |"
        )

    err_chico = abs(aprox.loc[50, "error_relativo_primer_orden"])
    err_grande = abs(aprox.loc[800, "error_relativo_primer_orden"])
    L += [
        "",
        f"El error de la aproximación lineal crece con el **cuadrado** del choque: "
        f"{err_chico:.1%} a 50 pb y {err_grande:.1%} a 800 pb. Es el argumento cuantitativo "
        "de por qué un límite de ALM expresado sólo en duración es insuficiente para "
        "escenarios de estrés: mide bien el riesgo del día a día y subestima precisamente "
        "el que motiva tener límites. El término de convexidad recupera casi todo.",
        "",
        "## 4. Cuánto cambia el tratamiento de los NMD",
        "",
        "| Tratamiento | Peor escenario | ΔEVE | / Tier 1 | ¿Outlier? |",
        "|---|---|---|---|---|",
    ]
    for idx, f in sens["tratamiento"].iterrows():
        L.append(
            f"| {idx} | {f['peor_escenario']} | {f['peor_delta_musd']:+,.0f} | "
            f"{f['peor_sobre_tier1']:+.1%} | {'**SÍ**' if f['supera_umbral'] else 'no'} |"
        )
    L += [
        "",
        "Tratar los depósitos como exigibles mañana los hace valer su nominal pase lo que "
        "pase: no aportan nada que compense la duración del activo. El modelo conductual les "
        "da el plazo que su comportamiento justifica, y con él absorben parte del choque. "
        "**Todo el trabajo del Módulo 2 se cobra en esta tabla.**",
        "",
        "### Reconciliación con el diagnóstico del Módulo 0",
        "",
        "El Módulo 0 estimaba **−21,9%** del Tier 1 con una aproximación de duración de primer "
        "orden. Aquí sale menos malo porque la aproximación lineal sobrestima el deterioro "
        "cerca de un 8% a 200 pb, como muestra la tabla anterior.",
        "",
        "Más interesante es de dónde venía el número que ese diagnóstico daba **antes** de la "
        "revisión final: −11,5%, con el que el banco «cumplía» el objetivo pre-registrado. "
        "Usaba la duración de pasivo sin ajustar por beta —el núcleo de **volumen** en lugar "
        "del de **repreciación**— cuando la propia función ya calculaba la versión correcta "
        "al lado. La diferencia es de definición de núcleo, y merece explicarse porque es el "
        "error conceptual que el proyecto entero persigue.",
        "",
        "Aquel diagnóstico usaba el núcleo **de volumen** —la porción del saldo que no se "
        "va— como si fuera el núcleo de repreciación. IRRBB define el núcleo como la porción "
        "que **no repacta**, que es el volumen estable multiplicado por (1 − β). Para vista: "
        "0,90 de volumen estable, pero sólo 0,65 de núcleo regulatorio.",
        "",
        "Un cuarto del depósito a la vista es dinero que se queda **y aun así sigue a la tasa "
        "de mercado**. Contarlo como núcleo le atribuye un plazo largo que no tiene y hace "
        "parecer al banco más cubierto de lo que está. Ésa es exactamente la nota conceptual "
        "de §6.5 del proyecto, y aquí se ve cuánto cuesta ignorarla: casi seis puntos "
        "porcentuales de Tier 1, la diferencia entre pasar el *outlier test* y no pasarlo.",
        "",
        "## 5. Sensibilidad al supuesto no identificado",
        "",
        "El plazo conductual del núcleo pesa mucho más aquí que en el margen, porque el EVE "
        "descuenta a veinte años en vez de proyectar doce meses. Y es justo el parámetro que "
        "el Módulo 2 demostró **no estimable** desde el saldo agregado.",
        "",
        "| Vida supuesta | Aplicada | ¿Tope muerde? | Peor escenario | ΔEVE / Tier 1 | ¿Outlier? |",
        "|---|---|---|---|---|---|",
    ]
    for vida, f in sens["vida"].iterrows():
        L.append(
            f"| {vida:.1f} a | {f['vida_aplicada_vista_a']:.1f} a | "
            f"{'sí' if f['tope_muerde'] else 'no'} | {f['peor_escenario']} | "
            f"**{f['peor_sobre_tier1']:+.1%}** | {'**SÍ**' if f['supera_umbral'] else 'no'} |"
        )

    rango = sens["vida"]["peor_sobre_tier1"]
    veredictos = set(sens["vida"]["supera_umbral"])
    L += [
        "",
        f"El supuesto mueve el resultado entre **{rango.min():+.1%}** y **{rango.max():+.1%}** "
        "del Tier 1.",
        "",
    ]
    L.append(
        "**El veredicto regulatorio cambia dentro del rango del supuesto.** Eso es lo que "
        "el banco tiene que decir al supervisor, no elegir la punta que le conviene: la "
        "conclusión no está determinada por los datos sino por una hipótesis que los datos "
        "no pueden respaldar."
        if len(veredictos) > 1 else
        "**El veredicto regulatorio aguanta en todo el rango del supuesto.** Es el mejor "
        "resultado posible cuando un parámetro no está identificado: la conclusión no "
        "depende de la hipótesis que no se puede verificar."
    )
    L += [
        "",
        "| Ajuste a la proporción estable | ΔEVE / Tier 1 | ¿Outlier? |",
        "|---|---|---|",
    ]
    for delta, f in sens["estable"].iterrows():
        L.append(
            f"| {delta:+.0%} | {f['peor_sobre_tier1']:+.1%} | "
            f"{'**SÍ**' if f['supera_umbral'] else 'no'} |"
        )

    mix = sensibilidad_mix_hipotecario(cfg)
    L += [
        "",
        "### Mix de balance (§11.1, declarado en `SENSITIVITY_GRID` desde el Módulo 0)",
        "",
        "| Hipotecario | Comercial | Dur. activo | Gap dur. | ΔEVE / Tier 1 | ¿Outlier? |",
        "|---|---|---|---|---|---|",
    ]
    for s, f in mix.iterrows():
        L.append(
            f"| {s:.0%} | {f['share_comercial']:.1%} | {f['duracion_activos']:.2f} a | "
            f"{f['gap_duracion']:+.2f} a | **{f['peor_sobre_tier1']:+.1%}** | "
            f"{'**SÍ**' if f['supera_umbral'] else 'no'} |"
        )
    L += [
        "",
        "Lo que gana el hipotecario lo cede el comercial, así que el balance sigue "
        "cuadrando. No es un artificio: es el traslado que un banco haría de verdad al "
        "crecer en minorista de largo plazo, y es el que más cambia su perfil — cambia "
        "cartera indexada que repacta cada trimestre por cartera fija a quince años.",
        "",
        "En el Módulo 0 se decidió **no** subir este mix para acercar el banco al umbral, "
        "porque los objetivos se habían pre-registrado y moverlo habría sido acomodar el "
        "balance al resultado deseado. Aparece aquí como sensibilidad, que es su sitio.",
        "",
        "## 6. Qué queda fuera",
        "",
        "El Módulo 5 documenta el alcance con honestidad. Lo que más pesaría aquí: la "
        "**opcionalidad de prepago** de la cartera hipotecaria, que acortaría el activo justo "
        "cuando las tasas bajan y lo alargaría cuando suben —convexidad negativa, el efecto "
        "va en contra del banco en ambos sentidos— y el **retiro anticipado** de los depósitos "
        "a plazo, que es la opción simétrica del lado del pasivo.",
    ]
    return "\n".join(L)
