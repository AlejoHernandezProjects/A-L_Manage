"""Tests de los controles del Módulo 0.

Que un control pase sobre datos buenos no prueba nada: un control que siempre
devuelve "PASA" también pasaría. Por eso la mitad de estos tests **corrompen el
dataset a propósito** y verifican que el control correspondiente lo detecta. Es la
misma lógica que una prueba de reversión en validación de modelos: se comprueba que
el detector detecta, no sólo que no molesta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.curves import macaulay_duration, schedule_cashflows
from src.data_gen import curve_tenors
from src.validation import (
    assert_no_errors,
    calibration_metrics,
    report,
    run_all_checks,
)


def _checks(bundle, cfg):
    return run_all_checks(bundle, cfg, hash_repetido=bundle.hash())


def _control(checks, numero):
    return [c for c in checks if c.numero == numero]


# --------------------------------------------------------------------------
# El dataset limpio pasa el gate
# --------------------------------------------------------------------------

def test_el_dataset_base_no_tiene_ningun_error(bundle, cfg):
    checks = _checks(bundle, cfg)
    errores = [c for c in checks if c.severidad == "ERROR" and not c.passed]
    assert not errores, "\n".join(f"#{c.numero} {c.nombre}: {c.detalle}" for c in errores)


def test_assert_no_errors_no_levanta_sobre_datos_buenos(bundle, cfg):
    assert_no_errors(_checks(bundle, cfg))


def test_se_ejecutan_los_once_controles(bundle, cfg):
    numeros = {c.numero for c in _checks(bundle, cfg)}
    assert numeros == set(range(1, 12))


def test_el_informe_es_markdown_con_veredicto(bundle, cfg):
    texto = report(_checks(bundle, cfg), calibration_metrics(bundle, cfg))
    assert "# Módulo 0 — Informe de validación" in texto
    assert "## Veredicto" in texto
    assert "ERROR fallidos" in texto


# --------------------------------------------------------------------------
# Mutaciones: cada control debe detectar su propio tipo de corrupción
# --------------------------------------------------------------------------

def test_control_01_detecta_un_balance_descuadrado(bundle_mutable, cfg):
    bundle_mutable.instrumentos.iloc[0, bundle_mutable.instrumentos.columns.get_loc("saldo")] += 1.0
    c = _control(_checks(bundle_mutable, cfg), 1)[0]
    assert not c.passed and c.severidad == "ERROR"


def test_control_02_detecta_un_nan(bundle_mutable, cfg):
    bundle_mutable.tasas_deposito.iloc[10, 0] = np.nan
    c = _control(_checks(bundle_mutable, cfg), 2)[0]
    assert not c.passed


def test_control_02_detecta_una_serie_truncada(bundle_mutable, cfg):
    bundle_mutable.saldos = bundle_mutable.saldos.iloc[:-3]
    c = _control(_checks(bundle_mutable, cfg), 2)[0]
    assert not c.passed


def test_control_03_detecta_repricing_posterior_al_vencimiento(bundle_mutable, cfg):
    inst = bundle_mutable.instrumentos
    col = inst.columns.get_loc("fecha_repreciacion")
    inst.iloc[0, col] = inst["fecha_vencimiento"].iloc[0] + pd.DateOffset(years=5)
    c = _control(_checks(bundle_mutable, cfg), 3)[0]
    assert not c.passed and c.severidad == "ERROR"


def test_control_03_detecta_una_fija_que_repacta_antes_de_vencer(bundle_mutable, cfg):
    inst = bundle_mutable.instrumentos
    fija = inst.index[inst["tipo_tasa"] == "fija"][0]
    inst.loc[fija, "fecha_repreciacion"] = inst.loc[fija, "fecha_vencimiento"] - pd.DateOffset(months=6)
    c = _control(_checks(bundle_mutable, cfg), 3)[0]
    assert not c.passed


def test_control_04_detecta_un_producto_que_pierde_plata(bundle_mutable, cfg):
    inst = bundle_mutable.instrumentos
    inst.loc[inst["categoria"] == "hipotecario", "tasa"] = 0.001
    c = _control(_checks(bundle_mutable, cfg), 4)[0]
    assert not c.passed and c.severidad == "ERROR"


def test_control_05_detecta_una_curva_desanclada(bundle_mutable, cfg):
    bundle_mutable.curvas.iloc[40, bundle_mutable.curvas.columns.get_loc("3M")] += 0.01
    c = _control(_checks(bundle_mutable, cfg), 5)[0]
    assert not c.passed and c.severidad == "ERROR"


def test_control_06_detecta_descuento_no_monotono(bundle_mutable, cfg):
    """Una tasa cero negativa en el tramo largo invierte el orden de los factores de
    descuento: valdría más un dólar dentro de 30 años que dentro de 20."""
    bundle_mutable.curvas.iloc[5, bundle_mutable.curvas.columns.get_loc("30A")] = -0.05
    c = _control(_checks(bundle_mutable, cfg), 6)[0]
    assert not c.passed and c.severidad == "ERROR"


def test_control_07_detecta_tasas_de_deposito_sin_relacion_con_el_mercado(bundle_mutable, cfg):
    rng = np.random.default_rng(0)
    for col in bundle_mutable.tasas_deposito.columns:
        bundle_mutable.tasas_deposito[col] = 0.02 + 0.001 * rng.standard_normal(120)
    c = _control(_checks(bundle_mutable, cfg), 7)[0]
    assert not c.passed and c.severidad == "ERROR"


def test_control_09_detecta_un_salto_de_saldo_fuera_del_estres(bundle_mutable, cfg):
    bundle_mutable.saldos.iloc[40, 0] *= 1.5
    c = _control(_checks(bundle_mutable, cfg), 9)[0]
    assert not c.passed and c.severidad == "WARNING"


def test_control_09_exime_la_ventana_del_estres(bundle, cfg):
    """El episodio del mes 96 es una caída de 9% por diseño. Que el control la
    ignorara por casualidad sería suerte; está exenta explícitamente."""
    c = _control(_checks(bundle, cfg), 9)[0]
    assert c.passed


def test_control_11_falla_si_no_se_compara_contra_una_segunda_corrida(bundle, cfg):
    """Un control de reproducibilidad que se saltea en silencio no es un control."""
    c = _control(run_all_checks(bundle, cfg), 11)[0]
    assert not c.passed and c.severidad == "ERROR"


def test_control_11_falla_ante_un_hash_distinto(bundle, cfg):
    c = _control(run_all_checks(bundle, cfg, hash_repetido="otro"), 11)[0]
    assert not c.passed


def test_assert_no_errors_levanta_cuando_algo_falla(bundle_mutable, cfg):
    bundle_mutable.curvas.iloc[3, bundle_mutable.curvas.columns.get_loc("3M")] += 0.05
    with pytest.raises(AssertionError, match="ERROR"):
        assert_no_errors(_checks(bundle_mutable, cfg))


# --------------------------------------------------------------------------
# Control #8: se ESPERA que roce el umbral
# --------------------------------------------------------------------------

def test_control_08_es_warning_y_no_bloquea(bundle, cfg):
    """Es WARNING a propósito: una regresión simétrica sobre un proceso asimétrico
    está mal especificada, así que su desviación es información, no defecto."""
    c = _control(_checks(bundle, cfg), 8)[0]
    assert c.severidad == "WARNING"


def test_la_beta_ols_no_coincide_con_ninguna_beta_verdadera(bundle, cfg):
    """El resultado que el Módulo 2 tiene que explicar: la beta ingenua queda entre
    β⁺ y β⁻ sin ser ninguna de las dos."""
    from scipy import stats as sp

    for prod in ("vista", "ahorro"):
        p = cfg.DEPOSIT_RATES[prod]
        beta = sp.linregress(
            bundle.ref_rate.to_numpy(), bundle.tasas_deposito[prod].to_numpy()
        ).slope
        assert abs(beta - p["beta_up"]) > 0.05
        assert abs(beta - p["beta_down"]) > 0.05


# --------------------------------------------------------------------------
# Métricas de calibración
# --------------------------------------------------------------------------

def test_la_tension_pedagogica_se_sostiene(bundle, cfg):
    """El resultado central del proyecto: ante +200 pb el NII **mejora** y el EVE se
    **deteriora**. Ese signo opuesto es la razón por la que Basilea exige ambas
    perspectivas, y es lo que este banco debe demostrar con números propios."""
    m = calibration_metrics(bundle, cfg)
    assert m["delta_nii_12m_up200"] > 0
    assert m["delta_eve_peor_sobre_tier1"] < 0


def test_el_gap_de_duracion_es_positivo(bundle, cfg):
    """Activos más largos que pasivos: es lo que hace que suba la tasa duela en EVE."""
    m = calibration_metrics(bundle, cfg)
    assert m["gap_duracion_a"] > 0


def test_la_duracion_efectiva_de_pasivos_es_menor_que_la_de_runoff(bundle, cfg):
    """Las dos convenciones no son intercambiables. La efectiva descuenta por beta y
    sale bastante más corta; mezclarlas es lo que hace que dos áreas del mismo banco
    reporten duraciones de pasivo distintas por un factor de dos."""
    m = calibration_metrics(bundle, cfg)
    assert m["duracion_pasivos_efectiva_a"] < m["duracion_pasivos_sensibles_a"]


def test_el_banco_supera_el_umbral_de_alerta_supervisora(bundle, cfg):
    """El *outlier test* de IRRBB: peor ΔEVE contra el 15% del Tier 1. **Este banco lo
    supera**, y ése es el resultado headline del proyecto.

    Este test afirmaba lo contrario hasta la revisión final, cuando el diagnóstico usaba
    la duración de pasivo sin ajustar por beta y daba −11,5%. No estaba comprobando una
    propiedad del modelo: estaba fijando una creencia sobre el resultado. Corregida la
    definición de núcleo, el diagnóstico da −21,9% y el Módulo 4 confirma −17,0% por
    revaluación completa.

    La lección va al README: un test que codifica el resultado esperado en vez de una
    invariante deja de avisar justo cuando el resultado cambia."""
    m = calibration_metrics(bundle, cfg)
    assert m["delta_eve_peor_sobre_tier1"] <= -0.15


def test_el_diagnostico_usa_la_duracion_de_pasivo_ajustada_por_beta(bundle, cfg):
    """La duración de runoff trata como núcleo todo el saldo que no se va; IRRBB define
    el núcleo como el que no **repacta**. Usar la primera hace parecer al banco más
    cubierto de lo que está — y un control de validación que halaga al balance es peor
    que no tenerlo."""
    m = calibration_metrics(bundle, cfg)
    tier1 = cfg.BANK_PROFILE["tier1_musd"]
    a, p = m["activos_sensibles_musd"], m["pasivos_sensibles_musd"]

    con_efectiva = -0.02 * (m["duracion_activos_sensibles_a"] * a - m["duracion_pasivos_efectiva_a"] * p)
    con_runoff = -0.02 * (m["duracion_activos_sensibles_a"] * a - m["duracion_pasivos_sensibles_a"] * p)

    assert m["delta_eve_up200_musd"] == pytest.approx(con_efectiva, rel=1e-9)
    assert con_runoff > con_efectiva  # la de runoff es la que halaga
    assert con_runoff / tier1 > -0.15 > con_efectiva / tier1  # y cambiaba el veredicto


def test_un_balance_calzado_da_delta_eve_cercano_a_cero(bundle, cfg):
    """§10 — un balance perfectamente calzado no debe tener riesgo de tasa.

    Se construye una pata activa y una pata pasiva con flujos idénticos y se
    comprueba que la sensibilidad de primer orden se cancela. Si esto fallara,
    cualquier ΔEVE del Módulo 4 sería ruido de la maquinaria de valoración."""
    tenores = curve_tenors(cfg)
    ceros = bundle.curvas.iloc[-1].to_numpy()
    nominal = 1_000.0
    for plazo, amort, freq in ((60, "bullet", 6), (240, "frances", 1), (36, "bullet", 1)):
        activo = schedule_cashflows(nominal, 0.05, plazo, amort, freq)
        pasivo = schedule_cashflows(nominal, 0.05, plazo, amort, freq)
        d_a = macaulay_duration(*activo, tenores, ceros)
        d_p = macaulay_duration(*pasivo, tenores, ceros)
        delta_eve = -0.02 * (d_a * nominal - d_p * nominal)
        assert delta_eve == pytest.approx(0.0, abs=1e-9)


def test_las_metricas_no_dependen_del_orden_de_las_filas(bundle_mutable, cfg):
    """Una métrica que cambiara al barajar el inventario indicaría que en algún
    punto se está usando el índice como si fuera información."""
    base = calibration_metrics(bundle_mutable, cfg)
    bundle_mutable.instrumentos = bundle_mutable.instrumentos.sample(frac=1.0, random_state=0)
    barajado = calibration_metrics(bundle_mutable, cfg)
    for k in base:
        assert barajado[k] == pytest.approx(base[k], rel=1e-10)
