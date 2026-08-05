"""Módulo 2 — Modelo conductual de depósitos sin vencimiento (NMD).

El Módulo 1 dejó un problema concreto: el gap contractual a 12 meses da −20,6% de los
activos mientras el margen financiero mejora ante subidas de tasas. La causa es que
los 22.000 M de vista y ahorro entran en la banda más corta porque contractualmente el
cliente retira mañana — lo que equivale a suponer traspaso del 100%.

El problema tiene dos mitades y este módulo trata las dos:

* la **cuantía** del traspaso — la beta asimétrica;
* el **momento** — el plazo conductual del núcleo.

Y encuentra que sólo una de las dos es estimable.

**La beta se recupera bien.** Separando la parte positiva y negativa de Δr y
regresando con rezagos, β̂⁺ y β̂⁻ caen a menos de 0,05 de sus valores verdaderos.

**El plazo del núcleo no es identificable a partir del saldo agregado.** Series
generadas con vidas verdaderas de 3 y de 10 años tienen correlación 0,99999 y difieren
como mucho un 0,17%. No es identificación débil: es información cero. El saldo
agregado no distingue entre un banco cuyos depósitos rotan rápido con originación alta
y uno cuyos depósitos son pegajosos con originación baja; sólo ve la suma.

Ése es el motivo de fondo por el que IRRBB **acota** el plazo del núcleo en vez de
pedir una estimación mejor: es un parámetro que el banco no puede falsar con los datos
que suele tener, y que además empuja el EVE en la dirección que al banco le conviene.
Un núcleo más largo abarata el descalce en el papel.

**Consecuencia de método.** Cuando un parámetro no está identificado, el entregable
honesto no es un número puntual sino el rango que produce. Por eso el análisis de
sensibilidad de este módulo no es un apéndice: es el resultado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from src.balance import bandas_frame, gap_repreciacion, indicadores_gap

__all__ = [
    "estimar_beta_asimetrica",
    "estimar_proporcion_estable",
    "plazo_replica",
    "diagnostico_identificabilidad",
    "nucleo_irrbb",
    "perfil_bandas_nmd",
    "modelo_nmd",
    "instrumentos_conductuales",
    "sensibilidad_nmd",
    "informe_nmd",
]

PRODUCTOS_NMD = ("vista", "ahorro")


# ---------------------------------------------------------------------------
# 1. Beta asimétrica
# ---------------------------------------------------------------------------

def _ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Mínimos cuadrados con matriz de covarianzas y R²."""
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    gl = max(1, len(y) - x.shape[1])
    s2 = float(resid @ resid) / gl
    cov = s2 * np.linalg.inv(x.T @ x)
    r2 = 1.0 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())
    return beta, cov, r2


def _suma_y_t(beta: np.ndarray, cov: np.ndarray, indices: slice) -> tuple[float, float]:
    """Suma de un subconjunto de coeficientes y su estadístico t."""
    c = np.zeros(len(beta))
    c[indices] = 1.0
    suma = float(c @ beta)
    ee = float(np.sqrt(c @ cov @ c))
    return suma, (suma / ee if ee > 0 else float("nan"))


def estimar_beta_asimetrica(
    ref_rate: pd.Series, tasas_deposito: pd.DataFrame, cfg
) -> pd.DataFrame:
    """Estima β⁺ y β⁻ por separado y los valida contra el ground truth.

    Especificación::

        Δd_t = a₀ + Σ_{k=0..L} a_k·Δr⁺_{t−k} + Σ_{k=0..L} b_k·Δr⁻_{t−k} + ε_t
        β̂⁺ = Σa_k      β̂⁻ = Σb_k

    **Por qué separar Δr en su parte positiva y negativa.** Es lo único que hace
    visible la asimetría. Una regresión sobre Δr a secas devuelve un promedio
    ponderado por la frecuencia de subidas y bajadas *de esta muestra concreta*, que no
    es un parámetro estructural y no sirve para proyectar otro ciclo — el error que el
    control #8 del Módulo 0 señala a propósito.

    **Por qué 12 rezagos.** Con ajuste parcial la respuesta se reparte en el tiempo:
    con λ = 0,30 en vista, sólo un tercio del traspaso ocurre en el mes del
    movimiento. Un estimador contemporáneo recogería ese tercio y llamaría beta a un
    número tres veces menor que el verdadero.

    Args:
        ref_rate: Tasa de política mensual.
        tasas_deposito: Tasas pagadas por producto.
        cfg: Configuración.

    Returns:
        DataFrame por producto con estimados, valores verdaderos, errores, t y R².
        Incluye la beta OLS ingenua sobre niveles como contraste.
    """
    L = cfg.NMD_ESTIMATION["rezagos_beta"]
    r = ref_rate.to_numpy()
    dr = np.diff(r)
    dr_up = np.maximum(dr, 0.0)
    dr_dn = np.minimum(dr, 0.0)

    filas = []
    for prod in tasas_deposito.columns:
        verdad = cfg.DEPOSIT_RATES[prod]
        dd = np.diff(tasas_deposito[prod].to_numpy())
        idx = np.arange(L, len(dd))
        x = np.column_stack(
            [np.ones(len(idx))]
            + [dr_up[idx - k] for k in range(L)]
            + [dr_dn[idx - k] for k in range(L)]
        )
        beta, cov, r2 = _ols(x, dd[idx])
        b_up, t_up = _suma_y_t(beta, cov, slice(1, 1 + L))
        b_dn, t_dn = _suma_y_t(beta, cov, slice(1 + L, 1 + 2 * L))

        # Contraste: la beta ingenua sobre niveles, que es la que suele reportarse.
        x_niv = np.column_stack([np.ones(len(r)), r])
        beta_niv, _, _ = _ols(x_niv, tasas_deposito[prod].to_numpy())

        filas.append(
            {
                "producto": prod,
                "beta_up_est": b_up,
                "beta_up_real": verdad["beta_up"],
                "error_up": b_up - verdad["beta_up"],
                "t_up": t_up,
                "beta_down_est": b_dn,
                "beta_down_real": verdad["beta_down"],
                "error_down": b_dn - verdad["beta_down"],
                "t_down": t_dn,
                "asimetria_est": b_dn - b_up,
                "asimetria_real": verdad["beta_down"] - verdad["beta_up"],
                "beta_ols_niveles": float(beta_niv[1]),
                "r2": r2,
            }
        )
    return pd.DataFrame(filas).set_index("producto")


# ---------------------------------------------------------------------------
# 2. Proporción estable (núcleo de volumen)
# ---------------------------------------------------------------------------

def estimar_proporcion_estable(saldos: pd.DataFrame, cfg) -> pd.DataFrame:
    """Estima la porción del saldo que no se va, destendenciando primero.

    El saldo crece un 4,5% anual, así que medir la estabilidad sobre el nivel bruto
    diría que el depósito es perfectamente estable — nunca baja del nivel inicial. Se
    quita la tendencia log-lineal y se mide la caída relativa contra ella.

    Se reportan dos lecturas:

    * **mínimo**: la mayor caída observada en toda la muestra. Es la práctica
      supervisora, y en esta serie la fija el episodio de estrés del mes 96, que es
      exactamente lo que debe fijarla.
    * **percentil**: menos sensible a un único episodio, útil como contraste de
      robustez cuando se discute si el estrés observado es representativo.

    **Qué mide y qué no.** Esto identifica *lo que no se va en volumen*. El
    `core_share` del generador describe *composición*. Y el núcleo regulatorio de
    IRRBB es un tercer objeto: *lo que no repacta*. Los tres números son distintos y
    confundirlos es el error central que este módulo existe para no cometer — un
    depósito puede ser perfectamente estable en saldo y aun así repactar entero.

    Args:
        saldos: Saldos observables de depósitos.
        cfg: Configuración.

    Returns:
        DataFrame por producto NMD con las dos lecturas, el crecimiento estimado, el
        `core_share` verdadero y el error.
    """
    pct = cfg.NMD_ESTIMATION["percentil_estable"]
    ventana = cfg.NMD_ESTIMATION["ventana_estable_m"]

    filas = []
    for prod in PRODUCTOS_NMD:
        s = saldos[prod].to_numpy()
        t = np.arange(len(s), dtype=float)
        pendiente, intercepto = np.polyfit(t, np.log(s), 1)
        tendencia = np.exp(intercepto + pendiente * t)
        ratio = s / tendencia

        estable_min = float(ratio.min())
        estable_pct = float(np.quantile(ratio, pct))
        caida_media_anual = float(
            pd.Series(ratio).rolling(ventana).min().dropna().mean()
        )
        real = cfg.NMD_PARAMS[prod]["core_share"]
        filas.append(
            {
                "producto": prod,
                "estable_minimo": estable_min,
                "estable_percentil": estable_pct,
                "estable_media_movil": caida_media_anual,
                "crecimiento_anual_est": float(np.exp(pendiente * 12) - 1),
                "crecimiento_anual_real": cfg.NMD_PARAMS[prod]["crecimiento_anual"],
                "core_share_real": real,
                "error_vs_core_share": estable_min - real,
            }
        )
    return pd.DataFrame(filas).set_index("producto")


# ---------------------------------------------------------------------------
# 3. Portafolio de réplica (plazo de repreciación del precio)
# ---------------------------------------------------------------------------

def plazo_replica(ref_rate: pd.Series, tasas_deposito: pd.DataFrame, cfg) -> pd.DataFrame:
    """Portafolio de réplica: qué escalera de tramos sigue mejor la tasa pagada.

    Se busca ``d_t ≈ α + Σ_j w_j·m_j(t)`` con ``m_j`` la media móvil de la tasa de
    referencia sobre *j* meses — el rendimiento de una escalera de tramos a *j* meses
    renovada continuamente — y ``w_j ≥ 0``. La suma de pesos es el traspaso de largo
    plazo; el plazo medio de la escalera es ``Σ w_j·(j/2) / Σ w_j``.

    Es una técnica estándar de mesa, y aquí se publica **como contraste, no como
    insumo**. Responde a "¿con qué plazo debo fondear o cubrir la tasa que pago?", que
    es una pregunta distinta de "¿cuánto tiempo se queda el dinero?". En este banco da
    del orden de cuatro meses frente a una vida de volumen de años, y la diferencia no
    es un error de ninguno de los dos: el precio de un depósito repacta en meses
    aunque el saldo permanezca lustros.

    Usar este número para asignar bandas de EVE mandaría los 22.000 M de NMD a la
    banda corta y dejaría el gap tan mal especificado como en el Módulo 1.

    Args:
        ref_rate: Tasa de política mensual.
        tasas_deposito: Tasas pagadas por producto.
        cfg: Configuración.

    Returns:
        DataFrame por producto con α, suma de pesos, plazo medio en años y los pesos.
    """
    tramos = list(cfg.NMD_ESTIMATION["tramos_replica_m"])
    r = pd.Series(ref_rate.to_numpy())
    burn = max(tramos)
    medias = [r.rolling(j).mean().to_numpy() for j in tramos]

    filas = []
    for prod in tasas_deposito.columns:
        x = np.column_stack([np.ones(len(r))] + medias)[burn - 1:]
        y = tasas_deposito[prod].to_numpy()[burn - 1:]
        sol, _ = nnls(x, y)
        alpha, pesos = float(sol[0]), sol[1:]
        total = float(pesos.sum())
        plazo_m = float((pesos * np.array(tramos) / 2.0).sum() / total) if total > 0 else 0.0
        fila = {
            "producto": prod,
            "alpha_pb": alpha * 1e4,
            "traspaso_implicito": total,
            "plazo_replica_a": plazo_m / 12.0,
            "n_observaciones": len(y),
        }
        fila.update({f"w_{j}m": float(p) for j, p in zip(tramos, pesos)})
        filas.append(fila)
    return pd.DataFrame(filas).set_index("producto")


# ---------------------------------------------------------------------------
# 4. Diagnóstico de identificabilidad — el exhibit central
# ---------------------------------------------------------------------------

def diagnostico_identificabilidad(cfg, producto: str = "vista", vidas=None) -> pd.DataFrame:
    """¿El saldo agregado contiene información sobre la vida del núcleo?

    Genera el banco varias veces cambiando **sólo** la vida promedio verdadera del
    núcleo y compara las series de saldo **observables**. Si el saldo llevara
    información sobre ese parámetro, las series diferirían.

    No difieren. Y eso no es un defecto del generador: es la situación real. El saldo
    agregado es la suma de un stock que se va y otro que entra, y no permite separar
    los dos flujos. Un banco con depósitos que rotan rápido y originación alta produce
    exactamente el mismo saldo que uno con depósitos pegajosos y originación baja.
    Distinguirlos exige datos de antigüedad de cuenta, que el equipo de ALM a menudo
    no tiene a mano aunque el core bancario los guarde.

    Args:
        cfg: Configuración base.
        producto: NMD cuya vida verdadera se varía.
        vidas: Vidas promedio a probar; por defecto las de ``NMD_ESTIMATION``.

    Returns:
        DataFrame con una fila por vida: media, desviación, crecimiento, correlación
        contra el caso base y diferencia relativa máxima.
    """
    # Import local: éste es el único punto del paquete que necesita fabricar
    # configuraciones variantes, y hacerlo global acoplaría src/ a config/.
    from config.params import make_config
    from src.data_gen import (
        generate_deposit_balances,
        generate_deposit_rates,
        generate_reference_rate,
        generate_yield_curves,
    )

    vidas = list(vidas if vidas is not None else cfg.NMD_ESTIMATION["vidas_diagnostico_a"])
    series: dict[float, np.ndarray] = {}
    for vida in vidas:
        cfg_v = make_config(**{f"NMD_PARAMS.{producto}.vida_promedio_a": vida})
        rng = np.random.default_rng(cfg_v.SEED)
        ref = generate_reference_rate(cfg_v, rng)
        # Se consume la curva aunque no se use, para que el flujo de números
        # aleatorios quede alineado con el de generate_dataset y la comparación
        # aísle el efecto de la vida y no el del muestreo.
        generate_yield_curves(ref, cfg_v, rng)
        tasas = generate_deposit_rates(ref, cfg_v, rng)
        saldos, _ = generate_deposit_balances(ref, tasas, cfg_v, rng)
        series[vida] = saldos[producto].to_numpy()

    base = cfg.NMD_PARAMS[producto]["vida_promedio_a"]
    ref_serie = series.get(base, series[vidas[0]])

    filas = []
    for vida, s in series.items():
        filas.append(
            {
                "vida_verdadera_a": vida,
                "saldo_medio": float(s.mean()),
                "desviacion": float(s.std()),
                "crecimiento_anual": float((s[-1] / s[0]) ** (12 / len(s)) - 1),
                "correlacion_vs_base": float(np.corrcoef(s, ref_serie)[0, 1]),
                "dif_relativa_max": float(np.abs(s / ref_serie - 1).max()),
            }
        )
    return pd.DataFrame(filas).set_index("vida_verdadera_a")


# ---------------------------------------------------------------------------
# 5. Núcleo IRRBB y reasignación a bandas
# ---------------------------------------------------------------------------

def nucleo_irrbb(
    producto: str, proporcion_estable: float, beta: float, cfg, vida_supuesta: float | None = None
) -> dict:
    """Aplica el marco estandarizado de IRRBB a un depósito sin vencimiento.

    La cadena es::

        estable  = proporción que no se va en volumen        (estimada)
        núcleo   = estable × (1 − β)                          (lo que no repacta)
        núcleo   = min(núcleo, tope de proporción)
        vida     = min(vida supuesta, tope de plazo medio)

    El paso ``× (1 − β)`` es la traducción de la nota conceptual del proyecto: la
    porción de un depósito que no repacta no es la que no se va. Un saldo puede ser
    perfectamente estable en volumen y aun así seguir a la tasa de mercado; en ese
    caso no aporta nada al descalce y no puede contarse como núcleo.

    Args:
        producto: ``"vista"`` o ``"ahorro"``.
        proporcion_estable: Salida de :func:`estimar_proporcion_estable`.
        beta: Traspaso a usar para el corte (por defecto la de subida).
        cfg: Configuración.
        vida_supuesta: Vida del núcleo en años; por defecto la de ``NMD_ESTIMATION``.

    Returns:
        Diccionario con el desglose y con qué topes mordieron.
    """
    categoria = cfg.NMD_CLIENTE[producto]
    topes = cfg.IRRBB_NMD_CAPS[categoria]
    vida_sup = (
        vida_supuesta if vida_supuesta is not None
        else cfg.NMD_ESTIMATION["vida_supuesta_a"][producto]
    )

    nucleo_bruto = proporcion_estable * (1.0 - beta)
    nucleo = min(nucleo_bruto, topes["cap_core"])
    vida = min(vida_sup, topes["cap_vida_a"])

    return {
        "producto": producto,
        "categoria_cliente": categoria,
        "proporcion_estable": proporcion_estable,
        "beta_corte": beta,
        "nucleo_bruto": nucleo_bruto,
        "nucleo": nucleo,
        "no_nucleo": 1.0 - nucleo,
        "cap_nucleo": topes["cap_core"],
        "tope_nucleo_muerde": nucleo_bruto > topes["cap_core"],
        "vida_supuesta_a": vida_sup,
        "vida_a": vida,
        "cap_vida_a": topes["cap_vida_a"],
        "tope_vida_muerde": vida_sup > topes["cap_vida_a"],
        "vida_real_a": cfg.NMD_PARAMS[producto]["vida_promedio_a"],
        "error_vida_a": vida - cfg.NMD_PARAMS[producto]["vida_promedio_a"],
    }


def perfil_bandas_nmd(saldo: float, nucleo: dict, cfg) -> pd.Series:
    """Reparte un NMD entre las 19 bandas IRRBB según el modelo conductual.

    El **no núcleo** va entero a la banda más corta: es la parte que repacta de
    inmediato. El **núcleo** corre off exponencialmente con ``λ = 1/vida``, de modo
    que la fracción que vence en la banda ``[a, b]`` (en años) es
    ``e^{−λa} − e^{−λb}``. El plazo medio de esa distribución es exactamente ``1/λ``,
    la vida asumida — que es la magnitud que el tope regulatorio acota.

    Se usa decaimiento exponencial y no reparto uniforme porque es la forma funcional
    del propio proceso de runoff y porque su plazo medio es inmediato de leer y de
    comparar contra el tope. Con reparto uniforme sobre ``[0, T]`` el plazo medio es
    ``T/2`` y la mitad del equipo acaba discutiendo si el tope aplica a ``T`` o a
    ``T/2``.

    Args:
        saldo: Saldo del producto en la fecha de corte.
        nucleo: Salida de :func:`nucleo_irrbb`.
        cfg: Configuración.

    Returns:
        Serie indexada por código de banda con el saldo asignado. Suma ``saldo``.
    """
    bandas = bandas_frame(cfg)
    lam = 1.0 / nucleo["vida_a"]
    a = bandas["mes_min"].to_numpy() / 12.0
    b = bandas["mes_max"].to_numpy() / 12.0
    fraccion = np.exp(-lam * a) - np.exp(-lam * b)

    perfil = saldo * nucleo["nucleo"] * fraccion
    perfil[0] += saldo * nucleo["no_nucleo"]
    return pd.Series(perfil, index=bandas.index, name=nucleo["producto"])


def _meses_representativos(cfg) -> pd.Series:
    """Plazo en meses que representa a cada banda, garantizando que cae dentro.

    Se usa el punto medio del marco estandarizado, recortado al interior de la banda.
    Hace falta el recorte porque el punto medio que Basilea da para overnight,
    0,0028 años, es 1/365 redondeado y bajo base 30/360 queda un pelo por encima del
    tope de su propia banda.
    """
    bandas = bandas_frame(cfg)
    pm = bandas["punto_medio_a"].to_numpy() * 12.0
    lo = bandas["mes_min"].to_numpy()
    hi = bandas["mes_max"].to_numpy()
    holgura = np.where(np.isinf(hi), 0.0, np.maximum((hi - lo) * 1e-6, 1e-12))
    tope = np.where(np.isinf(hi), pm, hi - holgura)
    return pd.Series(np.clip(pm, lo + 1e-12, tope), index=bandas.index)


def instrumentos_conductuales(instrumentos: pd.DataFrame, modelo: dict, cfg) -> pd.DataFrame:
    """Sustituye las cohortes NMD por tramos por banda según el modelo conductual.

    El resultado tiene exactamente la forma que consume :mod:`src.balance`, de modo
    que el gap conductual se obtiene con la **misma** función que produjo el
    contractual. Que las dos vistas compartan el código de agregación no es
    comodidad: es lo que garantiza que la diferencia entre ambas venga del supuesto
    de comportamiento y no de dos implementaciones que divergieron.

    Args:
        instrumentos: Inventario del Módulo 0.
        modelo: Salida de :func:`modelo_nmd`.
        cfg: Configuración.

    Returns:
        Inventario con las dos filas NMD reemplazadas por sus tramos.
    """
    meses = _meses_representativos(cfg)
    resto = instrumentos[~instrumentos["es_nmd"]].copy()

    filas = []
    for producto, perfil in modelo["perfiles"].items():
        original = instrumentos[instrumentos["categoria"] == producto].iloc[0]
        for banda, saldo in perfil.items():
            if saldo <= 0:
                continue
            m = float(meses[banda])
            filas.append(
                {
                    "id": f"{producto.upper()}-{banda}",
                    "categoria": producto,
                    "lado": "pasivo",
                    "saldo": float(saldo),
                    "tipo_tasa": "administrada",
                    "tasa": float(original["tasa"]),
                    "factor_devengo": 1.0,
                    "indice_referencia": None,
                    "spread_sobre_referencia": np.nan,
                    "fecha_originacion": original["fecha_originacion"],
                    "fecha_repreciacion": original["fecha_repreciacion"],
                    "fecha_vencimiento": original["fecha_vencimiento"],
                    "meses_a_repreciacion": m,
                    "meses_a_vencimiento": m,
                    "frecuencia_repricing_m": None,
                    "frecuencia_pago_meses": 1,
                    "amortizacion": "bullet",
                    "sensible": True,
                    "es_nmd": True,
                    "moneda": cfg.CONVENTIONS["moneda"],
                }
            )

    nuevos = pd.DataFrame(filas).set_index("id")
    return pd.concat([resto, nuevos.reindex(columns=resto.columns)])


# ---------------------------------------------------------------------------
# 6. Orquestador
# ---------------------------------------------------------------------------

def modelo_nmd(bundle, cfg, vida_supuesta: dict | None = None,
               ajuste_estable: float = 0.0, beta_clave: str | None = None) -> dict:
    """Corre el modelo conductual completo sobre un dataset del Módulo 0.

    Args:
        bundle: :class:`~src.data_gen.DatasetBundle`.
        cfg: Configuración.
        vida_supuesta: Override de la vida del núcleo por producto, para sensibilidad.
        ajuste_estable: Desplazamiento aditivo de la proporción estable, para
            sensibilidad. Se recorta a (0, 1].
        beta_clave: ``"beta_up"`` o ``"beta_largo"``; por defecto el de config.

    Returns:
        Diccionario con ``betas``, ``estables``, ``replica``, ``nucleos``,
        ``perfiles`` y ``saldos``.
    """
    betas = estimar_beta_asimetrica(bundle.ref_rate, bundle.tasas_deposito, cfg)
    estables = estimar_proporcion_estable(bundle.saldos, cfg)
    replica = plazo_replica(bundle.ref_rate, bundle.tasas_deposito, cfg)
    clave = beta_clave or cfg.NMD_ESTIMATION["beta_para_corte"]

    nucleos, perfiles, saldos = {}, {}, {}
    for prod in PRODUCTOS_NMD:
        if clave == "beta_largo":
            beta = 0.5 * (betas.loc[prod, "beta_up_est"] + betas.loc[prod, "beta_down_est"])
        else:
            beta = float(betas.loc[prod, "beta_up_est"])
        estable = float(np.clip(estables.loc[prod, "estable_minimo"] + ajuste_estable, 1e-6, 1.0))
        vida = (vida_supuesta or {}).get(prod)
        n = nucleo_irrbb(prod, estable, beta, cfg, vida_supuesta=vida)
        saldo = float(bundle.instrumentos.loc[bundle.instrumentos["categoria"] == prod, "saldo"].sum())
        nucleos[prod] = n
        saldos[prod] = saldo
        perfiles[prod] = perfil_bandas_nmd(saldo, n, cfg)

    return {
        "betas": betas,
        "estables": estables,
        "replica": replica,
        "nucleos": nucleos,
        "perfiles": perfiles,
        "saldos": saldos,
        "beta_clave": clave,
    }


# ---------------------------------------------------------------------------
# 7. Sensibilidad — el entregable principal
# ---------------------------------------------------------------------------

def _gap_12m(instrumentos: pd.DataFrame, cfg) -> tuple[float, float]:
    tabla = gap_repreciacion(instrumentos, cfg)
    ind = indicadores_gap(tabla, cfg)
    return ind["gap_acumulado_12m_musd"], ind["gap_12m_sobre_activos"]


def sensibilidad_nmd(bundle, cfg) -> dict:
    """Barrido sobre los supuestos **no identificados**.

    Cuando un parámetro no puede estimarse a partir de los datos, reportar un número
    puntual es afirmar más de lo que se sabe. Lo que sí puede reportarse es el rango
    de resultados que el supuesto produce, y qué parte de ese rango la política
    regulatoria ya recorta.

    Args:
        bundle: :class:`~src.data_gen.DatasetBundle`.
        cfg: Configuración.

    Returns:
        Diccionario con tres DataFrames: ``vida``, ``estable`` y ``beta``.
    """
    est = cfg.NMD_ESTIMATION

    filas_vida = []
    for vida in est["sensibilidad_vida_a"]:
        modelo = modelo_nmd(bundle, cfg, vida_supuesta={p: vida for p in PRODUCTOS_NMD})
        inst = instrumentos_conductuales(bundle.instrumentos, modelo, cfg)
        gap, ratio = _gap_12m(inst, cfg)
        filas_vida.append(
            {
                "vida_supuesta_a": vida,
                "vida_vista_aplicada_a": modelo["nucleos"]["vista"]["vida_a"],
                "vida_ahorro_aplicada_a": modelo["nucleos"]["ahorro"]["vida_a"],
                "tope_muerde": any(n["tope_vida_muerde"] for n in modelo["nucleos"].values()),
                "gap_12m_musd": gap,
                "gap_12m_sobre_activos": ratio,
            }
        )

    filas_estable = []
    for delta in est["sensibilidad_estable_delta"]:
        modelo = modelo_nmd(bundle, cfg, ajuste_estable=delta)
        inst = instrumentos_conductuales(bundle.instrumentos, modelo, cfg)
        gap, ratio = _gap_12m(inst, cfg)
        filas_estable.append(
            {
                "ajuste_estable": delta,
                "estable_vista": modelo["nucleos"]["vista"]["proporcion_estable"],
                "nucleo_vista": modelo["nucleos"]["vista"]["nucleo"],
                "gap_12m_musd": gap,
                "gap_12m_sobre_activos": ratio,
            }
        )

    filas_beta = []
    for clave in ("beta_up", "beta_largo"):
        modelo = modelo_nmd(bundle, cfg, beta_clave=clave)
        inst = instrumentos_conductuales(bundle.instrumentos, modelo, cfg)
        gap, ratio = _gap_12m(inst, cfg)
        filas_beta.append(
            {
                "beta_usada": clave,
                "beta_vista": modelo["nucleos"]["vista"]["beta_corte"],
                "nucleo_vista": modelo["nucleos"]["vista"]["nucleo"],
                "gap_12m_musd": gap,
                "gap_12m_sobre_activos": ratio,
            }
        )

    return {
        "vida": pd.DataFrame(filas_vida).set_index("vida_supuesta_a"),
        "estable": pd.DataFrame(filas_estable).set_index("ajuste_estable"),
        "beta": pd.DataFrame(filas_beta).set_index("beta_usada"),
    }


# ---------------------------------------------------------------------------
# 8. Informe
# ---------------------------------------------------------------------------

def informe_nmd(bundle, cfg, delta_nii: float | None = None) -> str:
    """Informe del modelo conductual de NMD, en formato de comité.

    Args:
        bundle: :class:`~src.data_gen.DatasetBundle`.
        cfg: Configuración.
        delta_nii: ΔNII a 12 meses ante +200 pb, del Módulo 0.

    Returns:
        Texto en Markdown.
    """
    modelo = modelo_nmd(bundle, cfg)
    ident = diagnostico_identificabilidad(cfg)
    sens = sensibilidad_nmd(bundle, cfg)

    inst_cond = instrumentos_conductuales(bundle.instrumentos, modelo, cfg)
    gap_c, ratio_c = _gap_12m(bundle.instrumentos, cfg)
    gap_b, ratio_b = _gap_12m(inst_cond, cfg)

    L = ["# Módulo 2 — Modelo conductual de depósitos sin vencimiento", ""]

    L += [
        "## 1. Beta asimétrica — estimada y validada",
        "",
        "| Producto | β̂⁺ | real | error | β̂⁻ | real | error | R² |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for prod, f in modelo["betas"].iterrows():
        L.append(
            f"| {prod} | {f['beta_up_est']:.3f} | {f['beta_up_real']:.2f} | "
            f"{f['error_up']:+.3f} | {f['beta_down_est']:.3f} | {f['beta_down_real']:.2f} | "
            f"{f['error_down']:+.3f} | {f['r2']:.2f} |"
        )
    L += [
        "",
        "La asimetría se recupera en los tres productos: el banco traslada más de una "
        "bajada que de una subida, y el estimador lo ve sin que se le haya dicho.",
        "",
        "Contraste con la beta ingenua sobre niveles, que es la que suele reportarse:",
        "",
        "| Producto | OLS niveles | β⁺ real | β⁻ real |",
        "|---|---|---|---|",
    ]
    for prod, f in modelo["betas"].iterrows():
        L.append(
            f"| {prod} | {f['beta_ols_niveles']:.3f} | {f['beta_up_real']:.2f} | "
            f"{f['beta_down_real']:.2f} |"
        )
    L += [
        "",
        "Queda entre las dos betas verdaderas sin ser ninguna. No es imprecisión: es un "
        "promedio ponderado por la frecuencia de subidas y bajadas **de esta muestra**, "
        "que no sirve para proyectar otro ciclo.",
        "",
    ]

    L += [
        "## 2. Proporción estable",
        "",
        "| Producto | Mínimo | Percentil | `core_share` real | Error |",
        "|---|---|---|---|---|",
    ]
    for prod, f in modelo["estables"].iterrows():
        L.append(
            f"| {prod} | {f['estable_minimo']:.3f} | {f['estable_percentil']:.3f} | "
            f"{f['core_share_real']:.2f} | {f['error_vs_core_share']:+.3f} |"
        )
    L += [
        "",
        "La sobreestimación no es ruido. El estimador mide *lo que no se va en volumen*; "
        "el parámetro del generador describe *composición*. Son objetos distintos, y el "
        "núcleo regulatorio es un tercero: *lo que no repacta*.",
        "",
    ]

    L += [
        "## 3. El plazo del núcleo no es estimable",
        "",
        "Cuatro bancos idénticos salvo la vida promedio **verdadera** del núcleo de vista:",
        "",
        "| Vida verdadera | Saldo medio | Crec. anual | Correlación vs. base | Dif. máx. |",
        "|---|---|---|---|---|",
    ]
    for vida, f in ident.iterrows():
        L.append(
            f"| {vida:.1f} a | {f['saldo_medio']:,.0f} | {f['crecimiento_anual']:.2%} | "
            f"{f['correlacion_vs_base']:.6f} | {f['dif_relativa_max']:.2%} |"
        )
    L += [
        "",
        "Una vida de 3 años y una de 10 producen la misma serie observable. No es "
        "identificación débil: es información cero. El saldo agregado es la suma de un "
        "stock que se va y otro que entra, y no permite separar los dos flujos.",
        "",
        "**Por eso IRRBB acota el plazo del núcleo en vez de pedir una estimación mejor.** "
        "Es un parámetro que el banco no puede falsar con los datos que suele tener, y que "
        "además empuja el EVE en la dirección que al banco le conviene: un núcleo más largo "
        "abarata el descalce en el papel. Los topes no son conservadurismo arbitrario, son "
        "la respuesta correcta a un problema de identificación.",
        "",
    ]

    L += [
        "## 4. Tres plazos distintos, tres preguntas distintas",
        "",
        "| Concepto | Pregunta que responde | Valor |",
        "|---|---|---|",
    ]
    for prod in PRODUCTOS_NMD:
        n = modelo["nucleos"][prod]
        rep = modelo["replica"].loc[prod, "plazo_replica_a"]
        L += [
            f"| Vida de volumen — {prod} | ¿cuánto tarda en irse el dinero? | "
            f"{n['vida_supuesta_a']:.1f} a (supuesto; real {n['vida_real_a']:.1f} a) |",
            f"| Plazo de réplica — {prod} | ¿cuánto tarda en repactar el precio? | "
            f"{rep:.2f} a (estimado) |",
            f"| Plazo IRRBB del núcleo — {prod} | ¿qué asigno a bandas para EVE? | "
            f"{n['vida_a']:.1f} a (tope {n['cap_vida_a']:.1f} a) |",
        ]
    L += [
        "",
        "El de réplica se calcula y se publica **como contraste, no como insumo**. El precio "
        "de un depósito repacta en meses aunque el saldo permanezca años; usar ese número "
        "para asignar bandas de EVE mandaría los 22.000 M a la banda corta y dejaría el gap "
        "tan mal especificado como en el Módulo 1.",
        "",
    ]

    L += [
        "## 5. Núcleo bajo el marco estandarizado IRRBB",
        "",
        "| Producto | Categoría | Estable | β corte | Núcleo bruto | Tope | Núcleo | ¿Tope muerde? |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for prod in PRODUCTOS_NMD:
        n = modelo["nucleos"][prod]
        L.append(
            f"| {prod} | {n['categoria_cliente'].replace('_', ' ')} | "
            f"{n['proporcion_estable']:.3f} | {n['beta_corte']:.3f} | "
            f"{n['nucleo_bruto']:.3f} | {n['cap_nucleo']:.2f} | {n['nucleo']:.3f} | "
            f"{'sí' if n['tope_nucleo_muerde'] else 'no'} |"
        )
    L += [
        "",
        "El paso `× (1 − β)` es la traducción operativa de la nota conceptual del proyecto: "
        "la porción que no repacta no es la que no se va. Un saldo puede ser perfectamente "
        "estable en volumen y aun así seguir a la tasa de mercado, y en ese caso no aporta "
        "nada al descalce.",
        "",
    ]

    L += [
        "## 6. El gap corregido",
        "",
        "| Tratamiento de los NMD | Gap acum. 12m | Sobre activos |",
        "|---|---|---|",
        f"| Contractual (Módulo 1) | {gap_c:,.0f} | {ratio_c:+.1%} |",
        f"| Conductual (Módulo 2) | {gap_b:,.0f} | {ratio_b:+.1%} |",
        "",
    ]
    if delta_nii is not None:
        L += [
            f"El gap contractual decía **{ratio_c:+.1%}** —fuertemente pasivo-sensible, en "
            f"estado ALERTA, con el margen debiendo caer— mientras el ΔNII del Módulo 0 mide "
            f"**{delta_nii:+.2%}**. Con el tratamiento conductual el gap pasa a "
            f"**{ratio_b:+.1%}**: de un descalce que exigiría plan de acción a un banco "
            "esencialmente calzado, dentro de política.",
            "",
            "Lo que cambió no fue el balance. Fue reconocer que de una subida de 100 pb el "
            "banco traslada del orden de 27 pb a las cuentas transaccionales, no 100.",
            "",
            "**Pero conviene no sobreafirmar el resultado.** El gap conductual sigue siendo "
            f"levemente negativo ({ratio_b:+.1%}) mientras el margen mejora. Un gap cercano a "
            "cero es compatible con un ΔNII pequeño y positivo, pero no lo predice, y la "
            "razón es estructural: **cualquier** análisis de brechas trata cada peso que "
            "repacta dentro de su banda como si trasladara el 100% del choque. El modelo "
            "conductual arregla el *momento* del repricing y la *proporción* que repacta; no "
            "puede arreglar que, una vez dentro de la banda, el traspaso se suponga completo.",
            "",
            "Es decir: el gap pasó de estar mal especificado a estar bien especificado y aun "
            "así seguir siendo una aproximación de primer orden. Determinar el signo y la "
            "magnitud del margen requiere proyectarlo — Módulo 3.",
            "",
        ]

    L += ["## 7. Sensibilidad — lo que no está identificado", ""]
    L += ["### Vida supuesta del núcleo", "",
          "| Vida supuesta | Vista aplicada | Ahorro aplicada | ¿Tope muerde? | Gap 12m | Sobre activos |",
          "|---|---|---|---|---|---|"]
    for vida, f in sens["vida"].iterrows():
        L.append(
            f"| {vida:.1f} a | {f['vida_vista_aplicada_a']:.1f} a | "
            f"{f['vida_ahorro_aplicada_a']:.1f} a | {'sí' if f['tope_muerde'] else 'no'} | "
            f"{f['gap_12m_musd']:,.0f} | {f['gap_12m_sobre_activos']:+.1%} |"
        )

    L += ["", "### Proporción estable", "",
          "| Ajuste | Estable vista | Núcleo vista | Gap 12m | Sobre activos |",
          "|---|---|---|---|---|"]
    for delta, f in sens["estable"].iterrows():
        L.append(
            f"| {delta:+.0%} | {f['estable_vista']:.3f} | {f['nucleo_vista']:.3f} | "
            f"{f['gap_12m_musd']:,.0f} | {f['gap_12m_sobre_activos']:+.1%} |"
        )

    L += ["", "### Beta usada para el corte", "",
          "| Beta | Valor vista | Núcleo vista | Gap 12m | Sobre activos |",
          "|---|---|---|---|---|"]
    for clave, f in sens["beta"].iterrows():
        L.append(
            f"| {clave} | {f['beta_vista']:.3f} | {f['nucleo_vista']:.3f} | "
            f"{f['gap_12m_musd']:,.0f} | {f['gap_12m_sobre_activos']:+.1%} |"
        )

    rango = sens["vida"]["gap_12m_sobre_activos"]
    L += [
        "",
        f"El supuesto de vida del núcleo mueve el gap a 12 meses entre "
        f"**{rango.min():+.1%}** y **{rango.max():+.1%}** de los activos. Ése es el rango "
        "honesto: un número puntual afirmaría más de lo que los datos permiten saber.",
        "",
        "## 8. Qué hereda el resto del proyecto",
        "",
        "- El **Módulo 3** usa β̂⁺ para proyectar el costo de fondeo, en lugar de inferir el "
        "signo del margen de una brecha.",
        "- El **Módulo 4** usa el perfil de bandas del núcleo para descontar los NMD, y debe "
        "reportar el ΔEVE **con su rango**, no con un número.",
        "- El tope de plazo del núcleo es la palanca supervisora sobre este banco: sin él, el "
        "supuesto de vida podría estirarse hasta hacer desaparecer el descalce en el papel.",
    ]
    return "\n".join(L)
