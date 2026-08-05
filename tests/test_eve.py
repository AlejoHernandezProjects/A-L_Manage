"""Tests del Módulo 4 — valor económico del patrimonio.

El test que da sentido a todo el proyecto está aquí: ante el mismo choque de +200 pb
el margen mejora y el valor económico se deteriora. Si alguna vez los dos apuntaran en
la misma dirección, o bien el banco cambió, o bien uno de los dos módulos se rompió.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.params import make_config
from src.deposits import instrumentos_conductuales, modelo_nmd
from src.eve import (
    cartera_valorable,
    duracion_convexidad,
    eve,
    informe_eve,
    sensibilidad_eve,
    tabla_aproximaciones,
    tabla_delta_eve,
    valor_presente,
)
from src.data_gen import curve_tenors
from src.nii import tabla_delta_nii
from src.scenarios import ESCENARIOS, aplicar_escenario


@pytest.fixture(scope="module")
def modelo(bundle, cfg):
    return modelo_nmd(bundle, cfg)


@pytest.fixture(scope="module")
def ceros(bundle):
    return bundle.curvas.iloc[-1].to_numpy()


@pytest.fixture(scope="module")
def cartera(bundle, cfg, modelo, ceros):
    inst = instrumentos_conductuales(bundle.instrumentos, modelo, cfg)
    return cartera_valorable(inst, ceros, cfg)


@pytest.fixture(scope="module")
def tabla(cartera, ceros, cfg):
    return tabla_delta_eve(cartera, ceros, cfg)


# --------------------------------------------------------------------------
# Calibración a la par
# --------------------------------------------------------------------------

def test_cada_instrumento_vale_su_saldo_en_la_curva_base(cartera, ceros, cfg):
    """La calibración a la par es exacta por construcción: si no lo fuera, el EVE de
    partida traería plusvalías latentes inventadas por la convención de descuento."""
    vp = valor_presente(cartera, ceros, cfg)
    saldos = pd.Series({it["id"]: it["saldo"] for it in cartera})
    assert np.allclose(vp.to_numpy(), saldos.reindex(vp.index).to_numpy(), rtol=1e-8)


def test_el_eve_base_es_el_patrimonio_contable(cartera, ceros, cfg):
    r = eve(cartera, ceros, cfg)
    patrimonio = cfg.BANK_PROFILE["activos_totales_musd"] * cfg.INSTRUMENT_SPECS["patrimonio"]["share"]
    assert r["eve"] == pytest.approx(patrimonio, rel=1e-8)


def test_el_patrimonio_no_entra_en_la_valoracion(cartera):
    """El EVE **es** el valor del patrimonio económico; incluirlo entre los pasivos
    sería contarlo dos veces."""
    assert not any(it["lado"] == "patrimonio" for it in cartera)


def test_los_spreads_calibrados_son_razonables(cartera):
    spreads = np.array([it["spread"] for it in cartera if it["sensible"]])
    assert np.all(np.abs(spreads) < 0.30)


# --------------------------------------------------------------------------
# Flujos de repreciación, no contractuales
# --------------------------------------------------------------------------

def test_un_variable_no_tiene_la_duracion_de_su_vencimiento(bundle, cfg, ceros):
    """Un crédito comercial a cinco años que repacta cada trimestre vuelve a valer su
    nominal en cada reset. Tratarlo como bono a cinco años infla la duración del activo
    casi un año y triplica el ΔEVE — el error más caro posible en este módulo."""
    inst = bundle.instrumentos
    comercial = inst[(inst["categoria"] == "comercial") & (inst["tipo_tasa"] == "variable")]
    cart = cartera_valorable(comercial, ceros, cfg)
    dc = duracion_convexidad(cart, ceros, cfg)
    assert dc["duracion_activos"] < 0.5
    assert comercial["meses_a_vencimiento"].median() > 24


def test_una_hipoteca_fija_si_arrastra_su_plazo(bundle, cfg, ceros):
    inst = bundle.instrumentos
    hip = inst[(inst["categoria"] == "hipotecario") & (inst["tipo_tasa"] == "fija")]
    dc = duracion_convexidad(cartera_valorable(hip, ceros, cfg), ceros, cfg)
    assert dc["duracion_activos"] > 4.0


# --------------------------------------------------------------------------
# El resultado central del proyecto
# --------------------------------------------------------------------------

def test_el_eve_se_deteriora_ante_subidas(tabla):
    assert tabla.loc["paralelo_arriba", "delta_musd"] < 0


def test_la_tension_central_del_proyecto(bundle, cfg, modelo, tabla):
    """**El resultado que el proyecto existe para demostrar.** Ante +200 pb el margen a
    doce meses mejora y el valor económico se deteriora. Son dos horizontes: el NII mira
    quién repacta antes, el EVE mira quién tiene más duración. Gestionar sólo por NII
    llevaría a celebrar una subida que está destruyendo patrimonio económico."""
    nii = tabla_delta_nii(bundle, cfg, modelo, escenarios=["paralelo_arriba"])
    assert nii.loc["paralelo_arriba", "delta_pct"] > 0
    assert tabla.loc["paralelo_arriba", "delta_sobre_tier1"] < 0


def test_el_peor_escenario_es_el_que_se_compara_con_el_umbral(tabla):
    peor = tabla["delta_sobre_tier1"].idxmin()
    assert tabla.loc[peor, "supera_umbral"] == (tabla.loc[peor, "delta_sobre_tier1"] <= -0.15)
    assert tabla["supera_umbral"].sum() <= len(tabla)


def test_los_seis_escenarios_estan_evaluados(tabla):
    assert list(tabla.index) == list(ESCENARIOS)


def test_los_paralelos_apuntan_en_direcciones_opuestas(tabla):
    assert tabla.loc["paralelo_arriba", "delta_musd"] < 0 < tabla.loc["paralelo_abajo", "delta_musd"]


def test_el_delta_se_mide_contra_el_tier_1(tabla, cfg):
    tier1 = cfg.BANK_PROFILE["tier1_musd"]
    assert np.allclose(tabla["delta_sobre_tier1"], tabla["delta_musd"] / tier1)


# --------------------------------------------------------------------------
# Duración y convexidad
# --------------------------------------------------------------------------

def test_el_gap_de_duracion_es_positivo(cartera, ceros, cfg):
    """Activo más largo que pasivo: es lo que hace que subir la tasa duela en valor."""
    dc = duracion_convexidad(cartera, ceros, cfg)
    assert dc["gap_duracion"] > 0
    assert dc["duracion_activos"] > dc["duracion_pasivos"]


def test_la_aproximacion_lineal_converge_para_choques_pequenos(cartera, ceros, cfg):
    aprox = tabla_aproximaciones(cartera, ceros, cfg, choques_pb=(1, 10))
    assert abs(aprox.loc[1, "error_relativo_primer_orden"]) < 0.01
    assert abs(aprox.loc[10, "error_relativo_primer_orden"]) < 0.02


def test_el_error_lineal_crece_con_el_choque(cartera, ceros, cfg):
    """Crece con el **cuadrado** del choque. Es el argumento de por qué un límite de ALM
    expresado sólo en duración subestima precisamente el riesgo que motiva tenerlo."""
    aprox = tabla_aproximaciones(cartera, ceros, cfg, choques_pb=(50, 200, 800))
    errores = [abs(aprox.loc[pb, "error_relativo_primer_orden"]) for pb in (50, 200, 800)]
    assert errores[0] < errores[1] < errores[2]
    assert errores[2] > 4 * errores[0]


def test_la_convexidad_recupera_casi_todo(cartera, ceros, cfg):
    aprox = tabla_aproximaciones(cartera, ceros, cfg, choques_pb=(200, 400))
    for pb in (200, 400):
        assert abs(aprox.loc[pb, "error_segundo_orden"]) < abs(aprox.loc[pb, "error_primer_orden"]) / 3


def test_la_duracion_efectiva_no_depende_del_tamano_del_bump(cartera, ceros, cfg):
    """Si dependiera, la derivada numérica estaría mal condicionada."""
    dc = duracion_convexidad(cartera, ceros, cfg)
    base = eve(cartera, ceros, cfg)["eve"]
    dy = 1e-5
    completa = eve(cartera, np.asarray(ceros) + dy, cfg)["eve"] - base
    lineal = -(dc["duracion_activos"] * dc["vp_activos"] - dc["duracion_pasivos"] * dc["vp_pasivos"]) * dy
    assert completa == pytest.approx(lineal, rel=0.01)


# --------------------------------------------------------------------------
# Balance calzado (§10)
# --------------------------------------------------------------------------

def _cartera_calzada() -> pd.DataFrame:
    base = {
        "categoria": "comercial", "tipo_tasa": "fija", "amortizacion": "bullet",
        "frecuencia_pago_meses": 6, "meses_a_repreciacion": 60,
        "meses_a_vencimiento": 60, "sensible": True, "es_nmd": False, "tasa": 0.05,
    }
    df = pd.DataFrame([
        {**base, "lado": "activo", "saldo": 1000.0},
        {**base, "lado": "pasivo", "saldo": 1000.0},
    ])
    df.index = ["A-1", "P-1"]
    return df


def test_un_balance_calzado_da_delta_eve_cero(cfg, ceros):
    """§10 — flujos idénticos en ambos lados: no hay riesgo de tasa que valorar."""
    cart = cartera_valorable(_cartera_calzada(), ceros, cfg)
    base = eve(cart, ceros, cfg)["eve"]
    assert base == pytest.approx(0.0, abs=1e-9)
    for nombre in ESCENARIOS:
        ceros_e = aplicar_escenario(ceros, curve_tenors(cfg), nombre, cfg)
        assert eve(cart, ceros_e, cfg)["eve"] == pytest.approx(0.0, abs=1e-9)


def test_un_balance_calzado_tiene_gap_de_duracion_cero(cfg, ceros):
    dc = duracion_convexidad(cartera_valorable(_cartera_calzada(), ceros, cfg), ceros, cfg)
    assert dc["gap_duracion"] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# El tratamiento de los NMD y la sensibilidad al supuesto no identificado
# --------------------------------------------------------------------------

def test_el_tratamiento_conductual_amortigua_el_deterioro(bundle, cfg, modelo, ceros):
    """Depósitos exigibles mañana valen su nominal pase lo que pase y no compensan nada.
    Con su plazo conductual absorben parte del choque. Aquí se cobra el Módulo 2."""
    contractual = cartera_valorable(bundle.instrumentos, ceros, cfg)
    conductual = cartera_valorable(
        instrumentos_conductuales(bundle.instrumentos, modelo, cfg), ceros, cfg
    )
    d_con = tabla_delta_eve(contractual, ceros, cfg).loc["paralelo_arriba", "delta_musd"]
    d_beh = tabla_delta_eve(conductual, ceros, cfg).loc["paralelo_arriba", "delta_musd"]
    assert d_beh > d_con


def test_un_nucleo_mas_largo_mejora_el_eve(bundle, cfg, modelo):
    """Monótono: más plazo conductual en el pasivo cierra el gap de duración. Si no lo
    fuera, el perfil de bandas estaría mal construido."""
    sens = sensibilidad_eve(bundle, cfg, modelo)["vida"]
    assert sens["peor_sobre_tier1"].is_monotonic_increasing


def test_el_veredicto_regulatorio_depende_del_supuesto_no_identificado(bundle, cfg, modelo):
    """El hallazgo que cierra el proyecto: el Módulo 2 demostró que el plazo del núcleo
    no es estimable desde el saldo agregado, y aquí ese mismo supuesto decide si el
    banco es o no *outlier*. La conclusión regulatoria no está determinada por los
    datos. Por eso el tope de 5 años del estándar no es conservadurismo: es lo único
    que impide al banco suponerse fuera del problema."""
    sens = sensibilidad_eve(bundle, cfg, modelo)["vida"]
    assert len(set(sens["supera_umbral"])) > 1


def test_el_tope_regulatorio_acota_el_mejor_caso(bundle, cfg, modelo):
    """Con vida supuesta de 7 años el tope la recorta a 5, así que el resultado se
    congela: el banco no puede seguir estirando el supuesto a su favor."""
    sens = sensibilidad_eve(bundle, cfg, modelo)["vida"]
    assert sens.loc[7.0, "peor_sobre_tier1"] == pytest.approx(sens.loc[5.0, "peor_sobre_tier1"], abs=0.005)
    assert bool(sens.loc[7.0, "tope_muerde"])


def test_una_proporcion_estable_mayor_mejora_el_eve(bundle, cfg, modelo):
    sens = sensibilidad_eve(bundle, cfg, modelo)["estable"]
    assert sens["peor_sobre_tier1"].is_monotonic_increasing


# --------------------------------------------------------------------------
# Informe
# --------------------------------------------------------------------------

def test_el_informe_lleva_las_secciones(bundle, cfg, modelo):
    nii = tabla_delta_nii(bundle, cfg, modelo)
    texto = informe_eve(bundle, cfg, modelo, tabla_nii=nii)
    for seccion in (
        "ΔEVE bajo los seis escenarios",
        "NII y EVE juntos",
        "dónde falla lo lineal",
        "Reconciliación con el diagnóstico del Módulo 0",
        "Sensibilidad al supuesto no identificado",
    ):
        assert seccion in texto


def test_el_informe_explica_la_tension(bundle, cfg, modelo):
    nii = tabla_delta_nii(bundle, cfg, modelo)
    texto = informe_eve(bundle, cfg, modelo, tabla_nii=nii)
    assert "tensión que el proyecto existe para demostrar" in texto
    assert "outlier" in texto.lower()
