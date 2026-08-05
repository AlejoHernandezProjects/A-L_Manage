"""Módulo 3 — Margen financiero (NII) proyectado a 12 meses.

El Módulo 2 dejó el gap bien especificado y demostró que aun así no puede predecir el
signo del margen: cualquier análisis de brechas trata cada peso que repacta dentro de
su banda como si trasladara el 100% del choque. Este módulo responde proyectando.

**Supuesto rector: balance constante y margen constante.** Es el estándar regulatorio
para ΔNII. El balance no crece ni cambia de composición, y cada instrumento conserva
su propio spread sobre la curva — lo que vence se reinvierte en el mismo producto al
mismo spread. Sólo se mueve la curva. Conviene tenerlo presente al leer los números:
un banco real respondería al choque cambiando de estrategia comercial, y el ΔNII no
pretende predecir eso. Mide la sensibilidad del balance **actual**, que es lo que un
comité necesita saber antes de decidir si quiere cambiarlo.

La mecánica de cada instrumento::

    tasa(m) = tasa_original + β · Δcurva(plazo relevante) · traspaso(m)

con ``traspaso(m) = 0`` mientras el instrumento no haya repactado. En los depósitos a
la vista y de ahorro, β es la del Módulo 2 —β̂⁺ si el choque sube, β̂⁻ si baja— y el
traspaso no es instantáneo: usa el perfil mensual que los propios coeficientes de
rezago estimados describen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.curves import continuous_to_nominal, interpolate_zero
from src.data_gen import curve_tenors
from src.scenarios import ESCENARIOS, ETIQUETAS, aplicar_escenario

__all__ = [
    "perfil_traspaso",
    "proyectar_nii",
    "tabla_delta_nii",
    "descomposicion_nii",
    "sensibilidad_nii",
    "informe_nii",
]


# ---------------------------------------------------------------------------
# Perfil temporal del traspaso
# ---------------------------------------------------------------------------

def perfil_traspaso(modelo: dict, producto: str, sube: bool, horizonte: int) -> np.ndarray:
    """Fracción del traspaso total ya materializada en cada mes del horizonte.

    Los coeficientes de rezago que el Módulo 2 estimó **son** el perfil temporal del
    traspaso: el coeficiente del rezago *k* es cuánto de la subida de hace *k* meses
    se refleja hoy en la tasa pagada. Su suma acumulada, normalizada, dice qué
    fracción del traspaso total ha ocurrido al mes *m*.

    Usar esto en vez de aplicar la beta entera desde el primer mes no es un refinamiento
    cosmético. Con λ = 0,30 en vista, suponer traspaso inmediato sobrestima el costo de
    fondeo del primer trimestre de forma apreciable, y el ΔNII a 12 meses sale
    sistemáticamente sesgado a la baja.

    Args:
        modelo: Salida de :func:`~src.deposits.modelo_nmd`.
        producto: Producto NMD.
        sube: ``True`` si el choque es al alza (usa β⁺), ``False`` si a la baja.
        horizonte: Meses del horizonte de proyección.

    Returns:
        Arreglo de longitud ``horizonte``, no decreciente y acotado en [0, 1].
    """
    coef = modelo["betas"].loc[producto, "coef_up" if sube else "coef_down"]
    coef = np.asarray(coef, dtype=float)
    total = coef.sum()
    if abs(total) < 1e-12:
        return np.ones(horizonte)

    acumulado = np.cumsum(coef) / total
    if len(acumulado) < horizonte:
        acumulado = np.concatenate([acumulado, np.ones(horizonte - len(acumulado))])
    acumulado = np.clip(acumulado[:horizonte], 0.0, 1.0)
    # El traspaso no se deshace: un banco no vuelve a bajar la tasa que ya subió sólo
    # porque un coeficiente de rezago salga negativo por ruido muestral.
    return np.maximum.accumulate(acumulado)


# ---------------------------------------------------------------------------
# Proyección
# ---------------------------------------------------------------------------

def _tenor_relevante(instrumentos: pd.DataFrame, cfg) -> np.ndarray:
    """Plazo de la curva al que repacta cada instrumento, en años.

    Un crédito comercial que repacta trimestralmente toma la tasa a 3 meses, no la de
    su vencimiento a cinco años. Un depósito a plazo que vence y se renueva toma la
    tasa del plazo típico del producto. Y los NMD siguen al tramo corto, que es lo que
    la tasa de política arrastra.
    """
    plazos_tipicos = {
        cat: float(np.mean(spec["plazo_meses"])) / 12.0
        for cat, spec in cfg.INSTRUMENT_SPECS.items()
    }
    ancla_corta = cfg.CURVE["tenor_ancla_corto_a"]

    tenores = np.empty(len(instrumentos), dtype=float)
    for i, (_, fila) in enumerate(instrumentos.iterrows()):
        if fila["es_nmd"]:
            tenores[i] = ancla_corta
        elif fila["tipo_tasa"] == "fija":
            tenores[i] = plazos_tipicos.get(fila["categoria"], 1.0)
        else:
            freq = fila["frecuencia_repricing_m"]
            tenores[i] = (float(freq) / 12.0) if freq and not pd.isna(freq) else ancla_corta
    return tenores


def proyectar_nii(
    instrumentos: pd.DataFrame,
    ceros_base,
    cfg,
    modelo: dict,
    escenario: str | None = None,
    horizonte: int | None = None,
    regla_balance: str | None = None,
    crecimiento: float | None = None,
    traspaso: str | None = None,
    beta_simetrica: bool = False,
    beta_activos: float | None = None,
) -> dict:
    """Proyecta el margen financiero sobre el horizonte bajo un escenario.

    Args:
        instrumentos: Inventario del Módulo 0.
        ceros_base: Curva cero en la fecha de corte.
        cfg: Configuración.
        modelo: Salida de :func:`~src.deposits.modelo_nmd`, de donde salen las betas.
        escenario: Uno de :data:`~src.scenarios.ESCENARIOS`; ``None`` para el base.
        horizonte: Meses; por defecto ``NII_PARAMS``.
        regla_balance: ``"constante"`` o ``"crecimiento"``.
        crecimiento: Tasa anual si la regla es de crecimiento.
        traspaso: ``"rezagos"`` o ``"inmediato"``.
        beta_simetrica: Si es ``True``, usa la misma beta para subidas y bajadas. Es
            el contrafactual que aísla cuánto vale la asimetría estimada.
        beta_activos: Override del traspaso en el activo.

    Returns:
        Diccionario con ``nii``, ``ingreso``, ``costo``, la senda mensual y el
        desglose por categoría.
    """
    p = cfg.NII_PARAMS
    horizonte = horizonte or p["horizonte_meses"]
    regla_balance = regla_balance or p["regla_balance"]
    crecimiento = p["crecimiento_anual"] if crecimiento is None else crecimiento
    traspaso = traspaso or p["traspaso"]
    beta_act = p["beta_activos"] if beta_activos is None else beta_activos

    tenores_curva = curve_tenors(cfg)
    ceros_base = np.asarray(ceros_base, dtype=float)
    ceros_esc = (
        aplicar_escenario(ceros_base, tenores_curva, escenario, cfg)
        if escenario else ceros_base
    )

    sens = instrumentos[instrumentos["sensible"]].copy()
    n = len(sens)
    taus = _tenor_relevante(sens, cfg)
    pagos = sens["frecuencia_pago_meses"].to_numpy(dtype=float)
    pagos_anio = np.maximum(1.0, 12.0 / np.where(pagos > 0, pagos, 1.0))

    y_base = interpolate_zero(tenores_curva, ceros_base, taus)
    y_esc = interpolate_zero(tenores_curva, ceros_esc, taus)
    delta = continuous_to_nominal(y_esc, pagos_anio) - continuous_to_nominal(y_base, pagos_anio)

    # El signo del choque se juzga en el tramo corto: es lo que decide si el banco
    # está en régimen de subida (β⁺) o de bajada (β⁻).
    ancla = cfg.CURVE["tenor_ancla_corto_a"]
    sube = float(
        interpolate_zero(tenores_curva, ceros_esc, ancla)
        - interpolate_zero(tenores_curva, ceros_base, ancla)
    ) >= 0

    es_nmd = sens["es_nmd"].to_numpy()
    es_activo = (sens["lado"] == "activo").to_numpy()
    categorias = sens["categoria"].to_numpy()

    betas = np.where(es_activo, beta_act, p["beta_mercado"]).astype(float)
    meses = np.arange(1, horizonte + 1)
    perfil = np.zeros((n, horizonte))

    k = np.ceil(sens["meses_a_repreciacion"].to_numpy(dtype=float)).astype(int)
    for i in range(n):
        if es_nmd[i]:
            prod = categorias[i]
            fila = modelo["betas"].loc[prod]
            if beta_simetrica:
                b = 0.5 * (fila["beta_up_est"] + fila["beta_down_est"])
            else:
                b = fila["beta_up_est"] if sube else fila["beta_down_est"]
            betas[i] = b
            # Una tasa administrada puede moverse el mismo mes: el retraso que
            # importa no es contractual, es el del propio traspaso.
            perfil[i] = (
                perfil_traspaso(modelo, prod, sube, horizonte)
                if traspaso == "rezagos" else np.ones(horizonte)
            )
        else:
            perfil[i] = (meses > k[i]).astype(float)

    saldos = sens["saldo"].to_numpy(dtype=float)
    if regla_balance == "crecimiento":
        factor = (1.0 + crecimiento) ** (meses / 12.0)
    else:
        factor = np.ones(horizonte)

    tasas = sens["tasa"].to_numpy(dtype=float)[:, None] + (betas * delta)[:, None] * perfil

    # Un banco no puede cobrarle al minorista por depositar: la tasa pagada se
    # detiene en cero. Es la restricción que apaga la ganancia asimétrica en
    # escenarios de bajada — β⁻ sólo paga mientras quede tasa que recortar.
    piso = p["piso_tasa_pasivo"]
    hay_pasivo = bool((~es_activo).any())
    if hay_pasivo:
        tasas[~es_activo] = np.maximum(tasas[~es_activo], piso)
    holgura_piso = float(tasas[~es_activo].min() - piso) if hay_pasivo else float("inf")

    devengo = (
        saldos[:, None] * tasas * sens["factor_devengo"].to_numpy(dtype=float)[:, None]
        * factor[None, :] / 12.0
    )

    ingreso_mes = devengo[es_activo].sum(axis=0)
    costo_mes = devengo[~es_activo].sum(axis=0)
    por_categoria = (
        pd.DataFrame({"categoria": categorias, "lado": sens["lado"].to_numpy(),
                      "devengo": devengo.sum(axis=1)})
        .groupby(["lado", "categoria"], observed=False)["devengo"].sum()
    )

    return {
        "escenario": escenario or "base",
        "nii": float(ingreso_mes.sum() - costo_mes.sum()),
        "ingreso": float(ingreso_mes.sum()),
        "costo": float(costo_mes.sum()),
        "senda_nii": ingreso_mes - costo_mes,
        "por_categoria": por_categoria,
        "sube": sube,
        "horizonte": horizonte,
        "holgura_piso_pasivo": holgura_piso,
    }


# ---------------------------------------------------------------------------
# Tablas
# ---------------------------------------------------------------------------

def tabla_delta_nii(bundle, cfg, modelo: dict, escenarios=None, **kwargs) -> pd.DataFrame:
    """ΔNII por escenario contra el caso base.

    Args:
        bundle: :class:`~src.data_gen.DatasetBundle`.
        cfg: Configuración.
        modelo: Salida de :func:`~src.deposits.modelo_nmd`.
        escenarios: Subconjunto; por defecto los seis.
        **kwargs: Se pasan a :func:`proyectar_nii` (regla de balance, traspaso…).

    Returns:
        DataFrame indexado por escenario con NII base, NII del escenario, Δ y Δ%.
    """
    ceros = bundle.curvas.iloc[-1].to_numpy()
    base = proyectar_nii(bundle.instrumentos, ceros, cfg, modelo, escenario=None, **kwargs)

    filas = []
    for nombre in (escenarios if escenarios is not None else ESCENARIOS):
        r = proyectar_nii(bundle.instrumentos, ceros, cfg, modelo, escenario=nombre, **kwargs)
        filas.append(
            {
                "escenario": nombre,
                "etiqueta": ETIQUETAS[nombre],
                "nii_base_musd": base["nii"],
                "nii_escenario_musd": r["nii"],
                "delta_musd": r["nii"] - base["nii"],
                "delta_pct": (r["nii"] - base["nii"]) / base["nii"],
            }
        )
    return pd.DataFrame(filas).set_index("escenario")


def descomposicion_nii(bundle, cfg, modelo: dict, escenario: str) -> pd.DataFrame:
    """De dónde sale cada peso del ΔNII, por producto.

    Args:
        bundle: :class:`~src.data_gen.DatasetBundle`.
        cfg: Configuración.
        modelo: Salida de :func:`~src.deposits.modelo_nmd`.
        escenario: Escenario a descomponer.

    Returns:
        DataFrame por lado y categoría con devengo base, del escenario y su diferencia.
    """
    ceros = bundle.curvas.iloc[-1].to_numpy()
    base = proyectar_nii(bundle.instrumentos, ceros, cfg, modelo)
    shock = proyectar_nii(bundle.instrumentos, ceros, cfg, modelo, escenario=escenario)

    tabla = pd.DataFrame(
        {"base_musd": base["por_categoria"], "escenario_musd": shock["por_categoria"]}
    ).fillna(0.0)
    tabla["delta_musd"] = tabla["escenario_musd"] - tabla["base_musd"]
    # El costo entra con signo positivo en el devengo del pasivo; para leer el aporte
    # al margen hay que invertirlo.
    signo = tabla.index.get_level_values("lado").map({"activo": 1.0, "pasivo": -1.0})
    tabla["aporte_al_margen"] = tabla["delta_musd"] * signo
    return tabla.sort_values("aporte_al_margen")


# ---------------------------------------------------------------------------
# Sensibilidad
# ---------------------------------------------------------------------------

def _trasladar_entorno(instrumentos: pd.DataFrame, modelo: dict, cfg, desplazamiento: float):
    """El mismo balance, pero en un entorno de tasas desplazado.

    Traslada la curva y, con ella, las tasas contractuales: los activos y el pasivo de
    mercado siguen al desplazamiento completo, y los NMD sólo la fracción que su beta
    de bajada permite, con el piso de cero.

    Sirve para responder una pregunta que el ΔNII estándar no responde: *¿cuánto de la
    ganancia de este banco ante bajadas depende de que hoy haya tasa que recortar?*
    """
    inst = instrumentos.copy()
    piso = cfg.NII_PARAMS["piso_tasa_pasivo"]
    tasas = inst["tasa"].to_numpy(dtype=float).copy()

    for i, (_, fila) in enumerate(inst.iterrows()):
        if not fila["sensible"]:
            continue
        if fila["es_nmd"]:
            beta = float(modelo["betas"].loc[fila["categoria"], "beta_down_est"])
        else:
            beta = 1.0
        nueva = tasas[i] + beta * desplazamiento
        tasas[i] = nueva if fila["lado"] == "activo" else max(nueva, piso)

    inst["tasa"] = tasas
    return inst


def sensibilidad_entorno(bundle, cfg, modelo: dict, desplazamientos_pb=(0, -100, -200, -300)) -> pd.DataFrame:
    """ΔNII del mismo banco partiendo de niveles de tasa cada vez más bajos.

    Es el exhibit que acota el resultado principal. La ganancia del banco ante bajadas
    viene de que traslada el recorte a sus depositantes más rápido de lo que pierde en
    el activo. Ese mecanismo **necesita que quede tasa que recortar**: la remuneración
    de un depósito se detiene en cero.

    Por eso las franquicias de depósitos valen mucho menos en un entorno de tasas cero,
    y por eso la banca europea y japonesa pasó una década sin poder monetizarlas.

    Args:
        bundle: :class:`~src.data_gen.DatasetBundle`.
        cfg: Configuración.
        modelo: Salida de :func:`~src.deposits.modelo_nmd`.
        desplazamientos_pb: Niveles de partida a probar, en pb sobre el actual.

    Returns:
        DataFrame con el ΔNII de cada paralelo y la holgura al piso.
    """
    ceros = bundle.curvas.iloc[-1].to_numpy()
    filas = []
    for pb in desplazamientos_pb:
        despl = pb / 1e4
        inst = _trasladar_entorno(bundle.instrumentos, modelo, cfg, despl)
        base = proyectar_nii(inst, ceros + despl, cfg, modelo)
        fila = {
            "nivel_3m": float(interpolate_zero(curve_tenors(cfg), ceros + despl, 0.25)),
            "nii_base_musd": base["nii"],
        }
        for esc in cfg.ESCENARIOS_NII:
            r = proyectar_nii(inst, ceros + despl, cfg, modelo, escenario=esc)
            fila[ETIQUETAS[esc]] = (r["nii"] - base["nii"]) / base["nii"]
            if not r["sube"]:
                fila["holgura_piso_pb"] = r["holgura_piso_pasivo"] * 1e4
        filas.append(fila)
    return pd.DataFrame(filas, index=pd.Index(desplazamientos_pb, name="desplazamiento_pb"))


def sensibilidad_nii(bundle, cfg, modelo: dict) -> dict:
    """Barrido sobre los supuestos que más mueven el resultado.

    Args:
        bundle: :class:`~src.data_gen.DatasetBundle`.
        cfg: Configuración.
        modelo: Salida de :func:`~src.deposits.modelo_nmd`.

    Returns:
        Diccionario de DataFrames: ``asimetria``, ``traspaso``, ``balance``,
        ``beta_activos``.
    """
    par = list(cfg.ESCENARIOS_NII)

    def _delta(**kw):
        t = tabla_delta_nii(bundle, cfg, modelo, escenarios=par, **kw)
        return {ETIQUETAS[e]: t.loc[e, "delta_pct"] for e in par}

    asimetria = pd.DataFrame(
        [
            {"beta": "asimétrica (β̂⁺ / β̂⁻)", **_delta(beta_simetrica=False)},
            {"beta": "simétrica (promedio)", **_delta(beta_simetrica=True)},
        ]
    ).set_index("beta")

    traspaso = pd.DataFrame(
        [
            {"traspaso": "con rezagos", **_delta(traspaso="rezagos")},
            {"traspaso": "inmediato", **_delta(traspaso="inmediato")},
        ]
    ).set_index("traspaso")

    balance = pd.DataFrame(
        [
            {"regla": "constante", **_delta(regla_balance="constante")},
            {"regla": "crecimiento 4,5%", **_delta(regla_balance="crecimiento", crecimiento=0.045)},
        ]
    ).set_index("regla")

    beta_act = pd.DataFrame(
        [{"beta_activos": b, **_delta(beta_activos=b)} for b in (0.80, 0.90, 1.00)]
    ).set_index("beta_activos")

    return {
        "asimetria": asimetria,
        "traspaso": traspaso,
        "balance": balance,
        "beta_activos": beta_act,
        "entorno": sensibilidad_entorno(bundle, cfg, modelo),
    }


# ---------------------------------------------------------------------------
# Informe
# ---------------------------------------------------------------------------

def informe_nii(bundle, cfg, modelo: dict, delta_nii_modulo0: float | None = None) -> str:
    """Informe de margen financiero en formato de comité.

    Args:
        bundle: :class:`~src.data_gen.DatasetBundle`.
        cfg: Configuración.
        modelo: Salida de :func:`~src.deposits.modelo_nmd`.
        delta_nii_modulo0: Aproximación de gap estático del Módulo 0, para reconciliar.

    Returns:
        Texto en Markdown.
    """
    tabla = tabla_delta_nii(bundle, cfg, modelo)
    sens = sensibilidad_nii(bundle, cfg, modelo)
    desc = descomposicion_nii(bundle, cfg, modelo, "paralelo_arriba")
    base = float(tabla["nii_base_musd"].iloc[0])
    arriba = float(tabla.loc["paralelo_arriba", "delta_pct"])
    abajo = float(tabla.loc["paralelo_abajo", "delta_pct"])

    L = [
        "# Módulo 3 — Margen financiero proyectado",
        "",
        f"Horizonte {cfg.NII_PARAMS['horizonte_meses']} meses · balance constante · "
        f"margen constante · NII base **{base:,.0f} M**. Cifras en USD millones.",
        "",
        "## 1. ΔNII bajo los seis escenarios",
        "",
        "| Escenario | NII | Δ | Δ% |",
        "|---|---|---|---|",
    ]
    for _, f in tabla.iterrows():
        L.append(
            f"| {f['etiqueta']} | {f['nii_escenario_musd']:,.0f} | "
            f"{f['delta_musd']:+,.0f} | **{f['delta_pct']:+.2%}** |"
        )
    L += [
        "",
        "Por convención el NII se titula con los dos paralelos (§8): el margen a 12 meses "
        "depende casi por completo del tramo corto, que es donde repacta el balance dentro "
        "del horizonte, y los escenarios rotacionales se distinguen sobre todo en el tramo "
        "largo, que apenas entra. Los seis se calculan igualmente.",
        "",
        "## 2. El resultado central: el banco gana en las dos direcciones",
        "",
        f"Ante **+200 pb el margen mejora {arriba:+.2%}**. Es la mitad de la tensión que el "
        "proyecto debe demostrar: el Módulo 4 mostrará que el mismo choque **deteriora** el "
        "valor económico. Ese signo opuesto es la razón por la que Basilea exige las dos "
        "perspectivas — un banco puede estar ganando margen hoy mientras destruye valor a "
        "largo plazo, y con una sola métrica no se ve.",
        "",
        f"Lo que sorprende a primera vista es la otra fila: ante **−200 pb el margen también "
        f"mejora, {abajo:+.2%}**. Un choque simétrico subiendo el margen en ambos sentidos "
        "parece un error de signo. No lo es, y la descomposición lo muestra: en el escenario "
        "de bajada el ingreso financiero cae, pero el **costo de fondeo cae más**.",
        "",
        "El mecanismo es la asimetría estimada en el Módulo 2. Cuando las tasas suben, el "
        "banco traslada a sus cuentas transaccionales apenas β⁺ ≈ 0,28 del movimiento y "
        "retiene el resto. Cuando bajan, traslada β⁻ ≈ 0,51 y se queda con la diferencia. "
        "**Eso es una posición larga en volatilidad de tasas**: la franquicia de depósitos "
        "no es sólo fondeo barato, es una opción que el banco tiene contra sus depositantes "
        "y que paga en ambos sentidos.",
        "",
        "## 3. Cuánto vale la asimetría",
        "",
        "| Beta usada | " + " | ".join(sens["asimetria"].columns) + " |",
        "|---|" + "---|" * len(sens["asimetria"].columns),
    ]
    for idx, f in sens["asimetria"].iterrows():
        L.append(f"| {idx} | " + " | ".join(f"{v:+.2%}" for v in f) + " |")
    L += [
        "",
        "La segunda fila es el contrafactual: el mismo balance, la misma proyección, pero "
        "con una beta única en vez de β⁺ y β⁻ por separado. **La ganancia casi desaparece.** "
        "Con beta simétrica este banco parece prácticamente neutral al riesgo de tasa en el "
        "margen, cuando en realidad tiene una posición valiosa en ambos sentidos.",
        "",
        "Ésa es la respuesta a por qué el Módulo 2 se molestó en estimar dos betas: no es "
        "precisión decorativa. Una beta única no reparte mal el efecto — lo **borra**, y le "
        "da al comité una foto en la que no hay nada que gestionar.",
        "",
        "## 3bis. Y por qué esa ganancia tiene fecha de caducidad",
        "",
        "| Nivel de partida (3M) | NII base | " + " | ".join(
            c for c in sens["entorno"].columns if c not in ("nivel_3m", "nii_base_musd", "holgura_piso_pb")
        ) + " | Holgura al piso |",
        "|---|---|" + "---|" * (len(cfg.ESCENARIOS_NII) + 1),
    ]
    for _, f in sens["entorno"].iterrows():
        deltas = " | ".join(
            f"{f[ETIQUETAS[e]]:+.2%}" for e in cfg.ESCENARIOS_NII
        )
        L.append(
            f"| {f['nivel_3m']:.2%} | {f['nii_base_musd']:,.0f} | {deltas} | "
            f"{f['holgura_piso_pb']:,.0f} pb |"
        )
    L += [
        "",
        "El mecanismo que hace ganar al banco cuando las tasas bajan **necesita que quede "
        "tasa que recortar**: la remuneración de un depósito se detiene en cero, y ahí β⁻ "
        "deja de pagar. La última columna mide cuántos puntos básicos le quedan al producto "
        "más barato antes de chocar contra el piso.",
        "",
        "Es la razón por la que una franquicia de depósitos vale mucho menos en un entorno "
        "de tasas cero, y por la que la banca europea y japonesa pasó una década sin poder "
        "monetizar la suya. Un ΔNII reportado sin esta acotación vendería como estructural "
        "una ganancia que depende del nivel de partida.",
        "",
        "## 4. De dónde sale el ΔNII (+200 pb)",
        "",
        "| Lado | Producto | Aporte al margen |",
        "|---|---|---|",
    ]
    for (lado, cat), f in desc.iterrows():
        if abs(f["aporte_al_margen"]) < 0.5:
            continue
        L.append(f"| {lado} | {cat} | {f['aporte_al_margen']:+,.0f} |")
    L += [
        "",
        "El activo aporta en positivo porque repacta al mercado sin fricción; el pasivo resta "
        "menos de lo que su volumen sugeriría, y ahí está el efecto de la beta baja de los "
        "depósitos transaccionales.",
        "",
        "## 5. Sensibilidad",
        "",
    ]
    for titulo, clave in (
        ("Perfil de traspaso", "traspaso"),
        ("Regla de balance", "balance"),
        ("Traspaso en el activo", "beta_activos"),
    ):
        t = sens[clave]
        L += [f"### {titulo}", "",
              "| " + t.index.name.replace("_", " ") + " | " + " | ".join(t.columns) + " |",
              "|---|" + "---|" * len(t.columns)]
        for idx, f in t.iterrows():
            etiqueta = f"{idx:.2f}" if isinstance(idx, float) else str(idx)
            L.append(f"| {etiqueta} | " + " | ".join(f"{v:+.2%}" for v in f) + " |")
        L.append("")

    if delta_nii_modulo0 is not None:
        L += [
            "## 6. Reconciliación con el diagnóstico del Módulo 0",
            "",
            "| Método | ΔNII (+200 pb) |",
            "|---|---|",
            f"| Gap estático a 12 meses (Módulo 0) | {delta_nii_modulo0:+.2%} |",
            f"| Proyección completa (Módulo 3) | {arriba:+.2%} |",
            "",
            "Los dos números miden lo mismo por caminos distintos y no tienen por qué "
            "coincidir. El del Módulo 0 es una aproximación de primer orden: reparte el "
            "choque proporcionalmente al tiempo que resta hasta el fin del horizonte y "
            "aplica la beta entera desde el momento de la repreciación. La proyección "
            "devenga mes a mes y usa el perfil temporal de traspaso.",
            "",
            "Que ambos den el mismo signo y un orden de magnitud parecido es la validación "
            "que interesa. Si difirieran en signo, uno de los dos tendría un error.",
            "",
        ]

    L += [
        "## 7. Qué queda para el Módulo 4",
        "",
        "El margen mejora ante subidas. Falta la otra mitad: el valor económico del "
        "patrimonio bajo los **seis** escenarios, comparado contra el 15% del Tier 1. Los "
        "escenarios ya están definidos en `src/scenarios.py` y el Módulo 4 los consume sin "
        "redefinir nada — que sean los mismos objetos es lo que permite poner NII y EVE en "
        "la misma fila de la tabla del comité sin comparar peras con manzanas.",
    ]
    return "\n".join(L)
