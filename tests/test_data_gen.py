"""Tests del generador del Módulo 0.

Cubren las tres cosas que un validador independiente miraría primero: que el
generador produce lo que dice producir, que los parámetros del ground truth están
efectivamente **en** los datos (y no sólo en el JSON), y que todo es reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from config.params import make_config
from src.curves import discount_factors
from src.data_gen import (
    _fraccion_principal_vivo,
    build_ground_truth,
    curve_tenors,
    generate_dataset,
)


# --------------------------------------------------------------------------
# Forma y completitud
# --------------------------------------------------------------------------

def test_series_tienen_120_meses_consecutivos(bundle, cfg):
    esperado = pd.date_range(cfg.DATES["inicio"], periods=120, freq="ME")
    for t in (bundle.ref_rate, bundle.curvas, bundle.tasas_deposito, bundle.saldos):
        assert len(t) == 120
        assert t.index.equals(esperado)


def test_sin_nan_en_ninguna_tabla(bundle):
    assert not bundle.ref_rate.isna().any()
    for t in (bundle.curvas, bundle.tasas_deposito, bundle.saldos, bundle.latentes):
        assert not t.isna().to_numpy().any()


def test_la_fecha_de_corte_es_el_ultimo_mes(bundle):
    assert bundle.fecha_corte == bundle.curvas.index[-1]


# --------------------------------------------------------------------------
# Tasa de referencia: la capa de decisión de comité
# --------------------------------------------------------------------------

def test_la_tasa_se_mueve_en_escalones_de_25_pb(bundle, cfg):
    cambios = np.diff(bundle.ref_rate.to_numpy())
    cambios = cambios[np.abs(cambios) > 1e-12]
    escalon = cfg.POLICY_RATE["escalon_pb"] / 1e4
    assert np.allclose(cambios / escalon, np.round(cambios / escalon), atol=1e-9)


def test_la_tasa_es_persistente(bundle):
    """Un Vasicek puro se movería casi todos los meses. La histéresis debe dejar
    la mayoría de los meses quietos, que es como se comporta una tasa de política."""
    cambios = np.diff(bundle.ref_rate.to_numpy())
    quietos = np.mean(np.abs(cambios) < 1e-12)
    assert quietos > 0.5


def test_ningun_movimiento_excede_el_tope_por_reunion(bundle, cfg):
    cambios = np.abs(np.diff(bundle.ref_rate.to_numpy()))
    assert cambios.max() <= cfg.POLICY_RATE["movimiento_max_pb"] / 1e4 + 1e-12


def test_la_tasa_recorre_los_regimenes(bundle):
    r = bundle.ref_rate.to_numpy()
    assert r[:30].mean() < 0.025            # tramo de tasas bajas
    assert r[54:84].mean() > 0.045          # meseta alta
    assert r[-1] < r[54:84].mean()          # normalización a la baja


# --------------------------------------------------------------------------
# Curva
# --------------------------------------------------------------------------

def test_la_curva_a_3m_esta_anclada_a_la_referencia(bundle, cfg):
    dif_pb = (bundle.curvas["3M"] - bundle.ref_rate).abs().max() * 1e4
    assert dif_pb <= cfg.VALIDATION_THRESHOLDS["curva_3m_vs_ref_pb"]


def test_la_curva_se_invierte_en_el_pico_del_ciclo(bundle):
    """Sin tramo invertido, el escenario de empinamiento del Módulo 4 no tendría
    nada que revertir. Debe estar invertida en el grueso de los meses 60–80."""
    pendiente = bundle.curvas["10A"] - bundle.curvas["3M"]
    ventana = pendiente.iloc[59:80]
    assert (ventana < 0).mean() > 0.8
    assert pendiente.iloc[:30].mean() > 0   # y empinada en tasas bajas


def test_factores_de_descuento_monotonos_en_las_120_fechas(bundle, cfg):
    tenores = curve_tenors(cfg)
    taus = np.linspace(0.05, 30.0, 200)
    for _, fila in bundle.curvas.iterrows():
        dfs = discount_factors(tenores, fila.to_numpy(), taus)
        assert dfs.min() > 0
        assert np.all(np.diff(dfs) < 0)


# --------------------------------------------------------------------------
# Tasas de depósito: el ground truth debe estar EN los datos
# --------------------------------------------------------------------------

def test_las_tasas_estan_ordenadas_por_producto(bundle):
    d = bundle.tasas_deposito
    assert (d["vista"] <= d["ahorro"] + 1e-9).all()
    assert (d["ahorro"] <= d["plazo"] + 1e-9).all()


def test_el_traspaso_es_asimetrico_en_la_direccion_correcta(bundle, cfg):
    """El test que da sentido a todo el proyecto: de una subida se traslada menos
    que de una bajada. Se mide sobre los datos generados, no sobre el config."""
    r = bundle.ref_rate.to_numpy()
    dr = np.diff(r)
    for prod in ("vista", "ahorro"):
        dd = np.diff(bundle.tasas_deposito[prod].to_numpy())
        sube, baja = dr > 0, dr < 0
        traspaso_sube = dd[sube].sum() / dr[sube].sum()
        traspaso_baja = dd[baja].sum() / dr[baja].sum()
        assert traspaso_baja > traspaso_sube, f"{prod}: {traspaso_baja:.2f} vs {traspaso_sube:.2f}"


def test_el_traspaso_acumulado_recupera_las_betas_del_ground_truth(bundle, cfg):
    """§10 — el modelo de depósitos debe recuperar la beta verdadera dentro de un
    margen. Con rezagos suficientes, Σβ̂ debe caer entre β⁺ y β⁻, y cerca del
    promedio ponderado por la frecuencia de subidas y bajadas de la muestra."""
    r = bundle.ref_rate.to_numpy()
    dr = np.diff(r)
    n_sube, n_baja = int((dr > 0).sum()), int((dr < 0).sum())
    w_up = n_sube / (n_sube + n_baja)

    for prod in ("vista", "ahorro", "plazo"):
        p = cfg.DEPOSIT_RATES[prod]
        dd = np.diff(bundle.tasas_deposito[prod].to_numpy())
        L = 12
        idx = np.arange(L, len(dd))
        x = np.column_stack([np.ones(len(idx))] + [dr[idx - k] for k in range(L)])
        beta, *_ = np.linalg.lstsq(x, dd[idx], rcond=None)
        suma = beta[1:].sum()
        esperado = w_up * p["beta_up"] + (1 - w_up) * p["beta_down"]
        assert p["beta_up"] - 0.10 <= suma <= p["beta_down"] + 0.10, f"{prod}: Σβ̂={suma:.2f}"
        assert abs(suma - esperado) < 0.20, f"{prod}: Σβ̂={suma:.2f} vs esperado {esperado:.2f}"


def test_una_ols_sobre_niveles_esta_mal_especificada(bundle, cfg):
    """El punto pedagógico del control #8: la beta ingenua no coincide con ninguna
    de las dos betas verdaderas. Debe quedar estrictamente entre ellas."""
    for prod in ("vista", "ahorro"):
        p = cfg.DEPOSIT_RATES[prod]
        beta_ols = stats.linregress(
            bundle.ref_rate.to_numpy(), bundle.tasas_deposito[prod].to_numpy()
        ).slope
        assert p["beta_up"] < beta_ols < p["beta_down"]


# --------------------------------------------------------------------------
# Saldos
# --------------------------------------------------------------------------

def test_los_saldos_son_positivos_y_crecen(bundle):
    assert (bundle.saldos > 0).to_numpy().all()
    for col in ("vista", "ahorro"):
        assert bundle.saldos[col].iloc[-12:].mean() > bundle.saldos[col].iloc[:12].mean()


def test_el_episodio_de_estres_esta_donde_dice_estar(bundle, cfg):
    """Caída de ~9% en dos meses desde el mes 96, y recuperación parcial después."""
    est = cfg.NMD_PARAMS["estres"]
    m = est["mes"] - 1
    v = bundle.saldos["vista"]
    caida = v.iloc[m + 1] / v.iloc[m - 1] - 1
    assert -0.13 < caida < -0.05
    assert v.iloc[m + 12] > v.iloc[m + 1]  # recupera


def test_el_estres_deja_una_erosion_permanente_del_core(bundle, cfg):
    """La erosión permanente está definida sobre C_t, no sobre el saldo observado.
    Se mide ahí para no confundirla con la estacionalidad: comparar noviembre contra
    diciembre mete 3 puntos de aguinaldos y esconde el efecto por completo.

    Clientes que se fueron a otro banco y no volvieron. Importa para el Módulo 2
    porque es un salto permanente de nivel en medio de la serie, justo el tipo de
    contaminación que rompe una estimación ingenua de persistencia."""
    est = cfg.NMD_PARAMS["estres"]
    m = est["mes"] - 1
    core = bundle.latentes["vista_core"]
    g = cfg.NMD_PARAMS["vista"]["crecimiento_anual"]

    meses = 24
    crecimiento_sin_estres = np.exp(g * (meses + 1) / 12.0)
    observado = core.iloc[m + meses] / core.iloc[m - 1]
    erosion = 1.0 - observado / crecimiento_sin_estres
    assert erosion == pytest.approx(est["erosion_permanente_core"], abs=0.01)


def test_el_saldo_a_plazo_absorbe_la_migracion(bundle):
    """El fondeo se traslada, no se evapora: en el pico del ciclo, cuando r − d se
    abre, plazo debe ganar peso relativo frente a vista."""
    mix_inicio = bundle.saldos["plazo"].iloc[:12].sum() / bundle.saldos.iloc[:12].sum().sum()
    mix_pico = bundle.saldos["plazo"].iloc[60:80].sum() / bundle.saldos.iloc[60:80].sum().sum()
    assert mix_pico > mix_inicio


def test_el_core_latente_no_viaja_en_las_tablas_observables(bundle):
    """Disciplina de diseño: el core verdadero es la respuesta del examen y no debe
    estar en lo que el Módulo 2 consume como observable."""
    assert set(bundle.saldos.columns) == {"vista", "ahorro", "plazo"}
    assert any("core" in c for c in bundle.latentes.columns)


# --------------------------------------------------------------------------
# Instrumentos
# --------------------------------------------------------------------------

def test_el_balance_cuadra(bundle):
    """§10 — el balance cuadra."""
    inst = bundle.instrumentos
    a = inst.loc[inst["lado"] == "activo", "saldo"].sum()
    pk = inst.loc[inst["lado"] != "activo", "saldo"].sum()
    assert abs(a - pk) / a < 1e-12


def test_el_mix_replica_el_perfil_del_banco(bundle, cfg):
    total = cfg.BANK_PROFILE["activos_totales_musd"]
    por_cat = bundle.instrumentos.groupby("categoria")["saldo"].sum()
    for categoria, spec in cfg.INSTRUMENT_SPECS.items():
        assert por_cat[categoria] == pytest.approx(total * spec["share"], rel=1e-10)


def test_fija_implica_repricing_igual_a_vencimiento(bundle):
    fijas = bundle.instrumentos.query("tipo_tasa == 'fija'")
    assert len(fijas) > 0
    assert (fijas["fecha_repreciacion"] == fijas["fecha_vencimiento"]).all()


def test_repreciacion_nunca_posterior_al_vencimiento(bundle):
    inst = bundle.instrumentos
    assert (inst["fecha_repreciacion"] <= inst["fecha_vencimiento"]).all()


def test_las_originaciones_estan_escalonadas(bundle):
    """Si todo se originara el mismo día, el balance entero repactaría en una banda
    y el gap del Módulo 1 sería un artefacto del generador."""
    meses = bundle.instrumentos["fecha_originacion"].dt.to_period("M").nunique()
    assert meses > 100


def test_las_cohortes_hipotecarias_heredan_la_tasa_de_su_añada(bundle):
    """El mecanismo económico central: las hipotecas originadas en el régimen de
    tasas bajas siguen vivas hoy con un cupón bajo. Si no fuera así, la pérdida de
    EVE ante +200 pb sería un artefacto y no un resultado."""
    hip = bundle.instrumentos.query("categoria == 'hipotecario' and tipo_tasa == 'fija'")
    viejas = hip[hip["fecha_originacion"] < hip["fecha_originacion"].min() + pd.DateOffset(months=30)]
    recientes = hip[hip["fecha_originacion"] > hip["fecha_originacion"].max() - pd.DateOffset(months=30)]
    assert len(viejas) > 20 and len(recientes) > 20
    assert viejas["tasa"].mean() < recientes["tasa"].mean() - 0.005


def test_las_tasas_activas_superan_el_costo_de_fondeo(bundle):
    inst = bundle.instrumentos
    pas = inst[(inst["lado"] == "pasivo") & inst["sensible"]]
    costo = (pas["saldo"] * pas["tasa"]).sum() / pas["saldo"].sum()
    act = inst[(inst["lado"] == "activo") & inst["sensible"]]
    for cat, d in act.groupby("categoria"):
        tasa = (d["saldo"] * d["tasa"] * d["factor_devengo"]).sum() / d["saldo"].sum()
        assert tasa > costo, f"{cat}: {tasa:.2%} <= {costo:.2%}"


def test_el_nmd_aparece_como_una_sola_cohorte_por_producto(bundle):
    inst = bundle.instrumentos
    assert int(inst["es_nmd"].sum()) == 2
    assert set(inst.loc[inst["es_nmd"], "categoria"]) == {"vista", "ahorro"}


# --------------------------------------------------------------------------
# Amortización de cohortes
# --------------------------------------------------------------------------

def test_una_cohorte_recien_originada_tiene_todo_el_principal_vivo():
    assert _fraccion_principal_vivo("frances", 0.05, 240, 0, 1) == pytest.approx(1.0)


def test_una_cohorte_vencida_no_tiene_principal_vivo():
    assert _fraccion_principal_vivo("frances", 0.05, 240, 240, 1) == pytest.approx(0.0)


def test_el_principal_vivo_decrece_con_la_edad():
    fracs = [_fraccion_principal_vivo("frances", 0.045, 240, e, 1) for e in (0, 24, 60, 120, 200)]
    assert all(np.diff(fracs) < 0)


def test_un_bullet_no_amortiza_antes_del_vencimiento():
    assert _fraccion_principal_vivo("bullet", 0.05, 60, 40, 6) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Ground truth y reproducibilidad
# --------------------------------------------------------------------------

def test_ground_truth_distingue_vida_promedio_de_half_life(cfg):
    gt = build_ground_truth(cfg)
    for prod in ("vista", "ahorro"):
        v = gt["nmd"][prod]
        assert v["half_life_a"] == pytest.approx(v["vida_promedio_a"] * np.log(2))
        assert v["lambda_decaimiento_anual"] == pytest.approx(1 / v["vida_promedio_a"])


def test_la_vida_promedio_del_core_respeta_el_tope_irrbb(cfg):
    """Con 4,2 años de vida promedio el banco queda bajo el tope de 5 años para
    minorista transaccional. Si esos 4,2 fueran half-life, la vida promedio sería
    6,06 y el core verdadero excedería el límite regulatorio."""
    gt = build_ground_truth(cfg)
    for prod in ("vista", "ahorro"):
        assert gt["nmd"][prod]["vida_promedio_a"] <= gt["nmd"][prod]["tope_irrbb_a"]


def test_misma_semilla_mismo_hash(cfg):
    """§9.7 — control #11."""
    assert generate_dataset(cfg).hash() == generate_dataset(make_config()).hash()


def test_semilla_distinta_hash_distinto(cfg):
    assert generate_dataset(cfg, seed=1).hash() != generate_dataset(cfg, seed=2).hash()


def test_los_overrides_no_contaminan_la_configuracion_base():
    """Un barrido de sensibilidad que mutara el módulo global rompería la
    reproducibilidad de la peor manera: en silencio y según el orden de ejecución."""
    base = make_config()
    tocado = make_config(**{"NMD_PARAMS.vista.core_share": 0.55})
    assert tocado.NMD_PARAMS["vista"]["core_share"] == 0.55
    assert make_config().NMD_PARAMS["vista"]["core_share"] == base.NMD_PARAMS["vista"]["core_share"]


def test_un_override_inexistente_falla_ruidosamente():
    with pytest.raises(KeyError):
        make_config(**{"NMD_PARAMS.vista.parametro_que_no_existe": 1.0})
