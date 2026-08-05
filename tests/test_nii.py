"""Tests del Módulo 3 — margen financiero proyectado.

Fijan bajo test el resultado central y su acotación. El resultado: la asimetría de la
beta hace que el banco gane margen en **ambas** direcciones del choque. La acotación:
esa ganancia depende de que quede tasa que recortar, y desaparece contra el piso de
cero. Un test que sólo comprobara el primero vendería como estructural algo que no lo
es.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.params import make_config
from src.deposits import modelo_nmd
from src.nii import (
    descomposicion_nii,
    informe_nii,
    perfil_traspaso,
    proyectar_nii,
    sensibilidad_entorno,
    sensibilidad_nii,
    tabla_delta_nii,
)
from src.validation import calibration_metrics


@pytest.fixture(scope="module")
def modelo(bundle, cfg):
    return modelo_nmd(bundle, cfg)


@pytest.fixture(scope="module")
def tabla(bundle, cfg, modelo):
    return tabla_delta_nii(bundle, cfg, modelo)


@pytest.fixture(scope="module")
def sens(bundle, cfg, modelo):
    return sensibilidad_nii(bundle, cfg, modelo)


def _cartera(*filas) -> pd.DataFrame:
    base = {
        "categoria": "comercial", "lado": "activo", "saldo": 100.0, "tasa": 0.05,
        "tipo_tasa": "variable", "factor_devengo": 1.0, "amortizacion": "bullet",
        "frecuencia_pago_meses": 1, "frecuencia_repricing_m": 3.0,
        "meses_a_repreciacion": 3, "meses_a_vencimiento": 60,
        "sensible": True, "es_nmd": False,
    }
    df = pd.DataFrame([{**base, **f} for f in filas])
    df.index = [f"I-{i:03d}" for i in range(len(df))]
    return df


# --------------------------------------------------------------------------
# Perfil de traspaso
# --------------------------------------------------------------------------

def test_el_perfil_de_traspaso_es_monotono_y_acotado(modelo):
    for prod in ("vista", "ahorro"):
        for sube in (True, False):
            p = perfil_traspaso(modelo, prod, sube, 12)
            assert len(p) == 12
            assert np.all((p >= 0) & (p <= 1))
            assert np.all(np.diff(p) >= 0)


def test_el_traspaso_no_llega_entero_el_primer_mes(modelo):
    """Con λ = 0,30 en vista sólo un tercio del traspaso ocurre en el mes del
    movimiento. Suponerlo inmediato sobrestima el costo de fondeo del primer
    trimestre."""
    p = perfil_traspaso(modelo, "vista", True, 12)
    assert p[0] < 0.7
    assert p[-1] == pytest.approx(1.0, abs=0.05)


# --------------------------------------------------------------------------
# Coherencia del caso base
# --------------------------------------------------------------------------

def test_el_nii_base_coincide_con_el_del_modulo_0(bundle, cfg, modelo):
    """El Módulo 0 calculó el NII devengado a la fecha de corte; la proyección a 12
    meses sin choque debe reproducirlo. Si no, una de las dos cuentas está mal."""
    base = proyectar_nii(bundle.instrumentos, bundle.curvas.iloc[-1].to_numpy(), cfg, modelo)
    assert base["nii"] == pytest.approx(
        calibration_metrics(bundle, cfg)["nii_base_musd"], rel=0.02
    )


def test_el_ingreso_supera_al_costo(bundle, cfg, modelo):
    base = proyectar_nii(bundle.instrumentos, bundle.curvas.iloc[-1].to_numpy(), cfg, modelo)
    assert base["ingreso"] > base["costo"] > 0


def test_la_senda_mensual_suma_el_nii(bundle, cfg, modelo):
    base = proyectar_nii(bundle.instrumentos, bundle.curvas.iloc[-1].to_numpy(), cfg, modelo)
    assert base["senda_nii"].sum() == pytest.approx(base["nii"])
    assert len(base["senda_nii"]) == cfg.NII_PARAMS["horizonte_meses"]


# --------------------------------------------------------------------------
# El resultado central
# --------------------------------------------------------------------------

def test_el_margen_mejora_ante_subidas(tabla):
    """La mitad de la tensión pedagógica: +200 pb mejora el NII. El Módulo 4 mostrará
    que el mismo choque deteriora el EVE."""
    assert tabla.loc["paralelo_arriba", "delta_pct"] > 0


def test_el_margen_tambien_mejora_ante_bajadas(tabla):
    """No es un error de signo. Con β⁻ > β⁺ el banco traslada más de una bajada que de
    una subida, así que gana en ambas direcciones: la franquicia de depósitos es una
    posición larga en volatilidad de tasas."""
    assert tabla.loc["paralelo_abajo", "delta_pct"] > 0


def test_en_la_bajada_el_costo_cae_mas_que_el_ingreso(bundle, cfg, modelo):
    """El mecanismo detrás del resultado anterior, aislado."""
    ceros = bundle.curvas.iloc[-1].to_numpy()
    base = proyectar_nii(bundle.instrumentos, ceros, cfg, modelo)
    abajo = proyectar_nii(bundle.instrumentos, ceros, cfg, modelo, escenario="paralelo_abajo")
    assert abajo["ingreso"] < base["ingreso"]
    assert abajo["costo"] < base["costo"]
    assert (base["costo"] - abajo["costo"]) > (base["ingreso"] - abajo["ingreso"])


def test_la_beta_simetrica_borra_la_ganancia(sens):
    """El contrafactual que justifica haber estimado dos betas en el Módulo 2: con una
    beta única el banco parece neutral al riesgo de tasa en el margen."""
    a = sens["asimetria"]
    for col in a.columns:
        assert a.loc["asimétrica (β̂⁺ / β̂⁻)", col] > a.loc["simétrica (promedio)", col]
    assert a.loc["simétrica (promedio)", "Paralelo arriba"] < 0.01


# --------------------------------------------------------------------------
# La acotación: el piso de la tasa de depósito
# --------------------------------------------------------------------------

def test_la_ganancia_por_bajadas_se_voltea_en_tasas_bajas(bundle, cfg, modelo):
    """β⁻ sólo paga mientras quede tasa que recortar. Contra el piso de cero, el mismo
    banco pierde margen cuando las tasas bajan. Es la razón por la que una franquicia
    de depósitos vale mucho menos en un entorno de tasas cero."""
    ent = sensibilidad_entorno(bundle, cfg, modelo, desplazamientos_pb=(0, -200))
    assert ent.loc[0, "Paralelo abajo"] > 0
    assert ent.loc[-200, "Paralelo abajo"] < 0


def test_la_holgura_al_piso_es_estrecha_hoy(bundle, cfg, modelo):
    """Este banco está al borde: al producto más barato le quedan pocos pb antes de
    chocar contra cero en el escenario de bajada."""
    ent = sensibilidad_entorno(bundle, cfg, modelo, desplazamientos_pb=(0,))
    assert 0 <= ent.loc[0, "holgura_piso_pb"] < 50


def test_la_ganancia_ante_subidas_aguanta_en_tasas_bajas(bundle, cfg, modelo):
    """El piso corta la ganancia por bajadas, no la de subidas: ahí no hay restricción."""
    ent = sensibilidad_entorno(bundle, cfg, modelo, desplazamientos_pb=(0, -200))
    assert ent.loc[-200, "Paralelo arriba"] > 0


def test_ninguna_tasa_de_pasivo_queda_negativa(bundle, cfg, modelo):
    ceros = bundle.curvas.iloc[-1].to_numpy() - 0.03
    r = proyectar_nii(bundle.instrumentos, ceros, cfg, modelo, escenario="paralelo_abajo")
    assert r["holgura_piso_pasivo"] >= -1e-12


# --------------------------------------------------------------------------
# Mecánica de la proyección
# --------------------------------------------------------------------------

def test_un_balance_calzado_da_delta_nii_cero(cfg, modelo, bundle):
    """§10 — si activo y pasivo repactan igual y con la misma beta, no hay riesgo de
    tasa en el margen."""
    cartera = _cartera(
        dict(lado="activo", tasa=0.06, meses_a_repreciacion=3, saldo=1000.0),
        dict(lado="pasivo", tasa=0.03, meses_a_repreciacion=3, saldo=1000.0,
             categoria="interbancario"),
    )
    ceros = bundle.curvas.iloc[-1].to_numpy()
    base = proyectar_nii(cartera, ceros, cfg, modelo)
    arriba = proyectar_nii(cartera, ceros, cfg, modelo, escenario="paralelo_arriba")
    assert arriba["nii"] == pytest.approx(base["nii"], abs=1e-9)


def test_un_instrumento_devenga_la_tasa_nueva_tras_repactar(cfg, modelo, bundle):
    """Repacta en el mes 3: los tres primeros meses van a la tasa vieja y el cuarto ya
    a la nueva."""
    cartera = _cartera(dict(meses_a_repreciacion=3, saldo=1200.0, tasa=0.05))
    ceros = bundle.curvas.iloc[-1].to_numpy()
    r = proyectar_nii(cartera, ceros, cfg, modelo, escenario="paralelo_arriba")
    senda = r["senda_nii"]
    assert np.allclose(senda[:3], senda[0])
    assert senda[3] > senda[2]
    assert np.allclose(senda[3:], senda[3])


def test_un_instrumento_que_no_repacta_no_reacciona(cfg, modelo, bundle):
    cartera = _cartera(dict(meses_a_repreciacion=36, meses_a_vencimiento=36, tipo_tasa="fija"))
    ceros = bundle.curvas.iloc[-1].to_numpy()
    base = proyectar_nii(cartera, ceros, cfg, modelo)
    arriba = proyectar_nii(cartera, ceros, cfg, modelo, escenario="paralelo_arriba")
    assert arriba["nii"] == pytest.approx(base["nii"])


def test_el_traspaso_inmediato_sobrestima_el_costo_de_fondeo(sens):
    """Aplicar la beta entera desde el primer mes encarece el fondeo antes de tiempo y
    sesga el ΔNII de subida a la baja."""
    t = sens["traspaso"]
    assert t.loc["inmediato", "Paralelo arriba"] < t.loc["con rezagos", "Paralelo arriba"]


def test_la_regla_de_crecimiento_no_altera_el_delta_relativo(sens):
    """Un balance que crece escala ingreso y costo a la vez; el ΔNII en porcentaje
    apenas se mueve. Si cambiara mucho, la regla estaría haciendo algo indebido."""
    b = sens["balance"]
    assert b.loc["crecimiento 4,5%", "Paralelo arriba"] == pytest.approx(
        b.loc["constante", "Paralelo arriba"], abs=0.005
    )


def test_menos_traspaso_en_el_activo_reduce_la_ganancia(sens):
    ba = sens["beta_activos"]
    assert ba.loc[0.80, "Paralelo arriba"] < ba.loc[1.00, "Paralelo arriba"]


# --------------------------------------------------------------------------
# Tablas, descomposición e informe
# --------------------------------------------------------------------------

def test_la_tabla_cubre_los_seis_escenarios(tabla):
    from src.scenarios import ESCENARIOS

    assert list(tabla.index) == list(ESCENARIOS)


def test_el_delta_es_consistente_con_los_niveles(tabla):
    assert np.allclose(
        tabla["delta_musd"], tabla["nii_escenario_musd"] - tabla["nii_base_musd"]
    )


def test_la_descomposicion_suma_el_delta(bundle, cfg, modelo, tabla):
    desc = descomposicion_nii(bundle, cfg, modelo, "paralelo_arriba")
    assert desc["aporte_al_margen"].sum() == pytest.approx(
        tabla.loc["paralelo_arriba", "delta_musd"], rel=1e-9
    )


def test_el_activo_aporta_en_positivo_ante_subidas(bundle, cfg, modelo):
    desc = descomposicion_nii(bundle, cfg, modelo, "paralelo_arriba")
    activos = desc.xs("activo", level="lado")["aporte_al_margen"]
    assert (activos >= -1e-9).all()


def test_la_magnitud_del_choque_mueve_el_delta_nii(bundle, cfg, modelo):
    """Los escenarios se definen una sola vez: recalibrar el choque en config debe
    propagarse al NII sin tocar src/nii.py."""
    fuerte = make_config(**{"IRRBB_SHOCKS_PB.paralelo": 400.0})
    t_base = tabla_delta_nii(bundle, cfg, modelo, escenarios=["paralelo_arriba"])
    t_fuerte = tabla_delta_nii(bundle, fuerte, modelo, escenarios=["paralelo_arriba"])
    assert t_fuerte.loc["paralelo_arriba", "delta_pct"] > t_base.loc["paralelo_arriba", "delta_pct"]


def test_el_informe_lleva_las_secciones(bundle, cfg, modelo):
    texto = informe_nii(bundle, cfg, modelo, delta_nii_modulo0=0.0172)
    for seccion in (
        "ΔNII bajo los seis escenarios",
        "gana en las dos direcciones",
        "Cuánto vale la asimetría",
        "fecha de caducidad",
        "Reconciliación con el diagnóstico del Módulo 0",
    ):
        assert seccion in texto


def test_el_informe_reconcilia_con_el_modulo_0(bundle, cfg, modelo, tabla):
    """Los dos métodos miden lo mismo por caminos distintos. Coincidir en signo y orden
    de magnitud es la validación; diferir en signo indicaría un error en uno de los dos."""
    m0 = calibration_metrics(bundle, cfg)["delta_nii_12m_up200"]
    m3 = tabla.loc["paralelo_arriba", "delta_pct"]
    assert np.sign(m0) == np.sign(m3)
    assert abs(m3 - m0) < 0.03
