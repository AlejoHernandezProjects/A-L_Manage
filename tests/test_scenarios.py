"""Tests de los seis escenarios IRRBB.

Este módulo se define una sola vez y lo consumen los Módulos 3 y 4 (§9.5), así que un
error aquí se propaga a las dos métricas del informe final sin que nada más falle. De
ahí que los tests comprueben la **forma** de cada choque y no sólo que devuelva algo.
"""

from __future__ import annotations

import numpy as np
import pytest

from config.params import make_config
from src.data_gen import curve_tenors
from src.scenarios import (
    ESCENARIOS,
    aplicar_escenario,
    choque,
    curvas_escenarios,
    escalares,
    piso_post_choque,
)


@pytest.fixture
def tenores(cfg):
    return curve_tenors(cfg)


# --------------------------------------------------------------------------
# Escalares de tramo
# --------------------------------------------------------------------------

def test_el_escalar_corto_vale_uno_en_el_plazo_inmediato(cfg):
    s_corto, s_largo = escalares([0.0], cfg)
    assert s_corto[0] == pytest.approx(1.0)
    assert s_largo[0] == pytest.approx(0.0)


def test_los_escalares_suman_uno(cfg, tenores):
    s_corto, s_largo = escalares(tenores, cfg)
    assert np.allclose(s_corto + s_largo, 1.0)


def test_el_escalar_corto_decae(cfg, tenores):
    s_corto, _ = escalares(tenores, cfg)
    assert np.all(np.diff(s_corto) < 0)
    assert s_corto[-1] < 0.01  # a 30 años ya no queda nada de "corto"


# --------------------------------------------------------------------------
# Forma de cada escenario
# --------------------------------------------------------------------------

def test_los_paralelos_son_constantes_en_el_plazo(cfg, tenores):
    arriba = choque("paralelo_arriba", tenores, cfg)
    abajo = choque("paralelo_abajo", tenores, cfg)
    r = cfg.IRRBB_SHOCKS_PB["paralelo"] / 1e4
    assert np.allclose(arriba, r)
    assert np.allclose(abajo, -r)
    assert np.allclose(arriba, -abajo)


def test_las_cortas_se_apagan_en_el_tramo_largo(cfg, tenores):
    up = choque("cortas_arriba", tenores, cfg)
    r = cfg.IRRBB_SHOCKS_PB["cortas"] / 1e4
    # A un mes el escalar ya vale 0,979, no 1: el choque de cortas empieza a apagarse
    # desde el primer día. A 30 años no queda prácticamente nada.
    assert 0.97 * r < up[0] < r
    assert abs(up[-1]) < r * 0.02
    assert np.all(np.diff(up) < 0)


def test_las_cortas_son_simetricas_entre_si(cfg, tenores):
    assert np.allclose(
        choque("cortas_arriba", tenores, cfg), -choque("cortas_abajo", tenores, cfg)
    )


def test_el_empinamiento_baja_cortas_y_sube_largas(cfg, tenores):
    """Si no cambiara de signo entre tramos no estaría empinando nada."""
    d = choque("empinamiento", tenores, cfg)
    assert d[0] < 0
    assert d[-1] > 0
    assert np.all(np.diff(d) > 0)


def test_el_aplanamiento_hace_lo_contrario(cfg, tenores):
    d = choque("aplanamiento", tenores, cfg)
    assert d[0] > 0
    assert d[-1] < 0
    assert np.all(np.diff(d) < 0)


def test_empinamiento_y_aplanamiento_no_son_opuestos(cfg, tenores):
    """El estándar usa factores distintos (0,65/0,90 contra 0,80/0,60) a propósito:
    una curva no se empina y se aplana con la misma intensidad en cada tramo."""
    emp = choque("empinamiento", tenores, cfg)
    apl = choque("aplanamiento", tenores, cfg)
    assert not np.allclose(emp, -apl)


def test_un_escenario_inventado_falla(cfg, tenores):
    with pytest.raises(ValueError, match="desconocido"):
        choque("paralelo_de_lado", tenores, cfg)


# --------------------------------------------------------------------------
# Suelo post-choque
# --------------------------------------------------------------------------

def test_el_suelo_empieza_negativo_y_llega_a_cero(cfg):
    piso = piso_post_choque([0.0, 10.0, 20.0, 30.0], cfg)
    assert piso[0] == pytest.approx(-0.01)
    assert piso[1] == pytest.approx(-0.005)
    assert piso[2] == pytest.approx(0.0)
    assert piso[3] == pytest.approx(0.0)


def test_el_suelo_es_no_decreciente(cfg, tenores):
    piso = piso_post_choque(tenores, cfg)
    assert np.all(np.diff(piso) >= 0)


def test_el_suelo_muerde_sobre_una_curva_baja(cfg, tenores):
    """Sin suelo, un choque de −200 pb sobre una curva al 0,5% daría tasas
    profundamente negativas y un EVE que sería un artefacto aritmético."""
    curva_baja = np.full(len(tenores), 0.005)
    resultado = aplicar_escenario(curva_baja, tenores, "paralelo_abajo", cfg)
    piso = piso_post_choque(tenores, cfg)
    assert np.allclose(resultado, piso)
    assert np.all(resultado >= piso - 1e-12)


def test_el_suelo_no_estorba_a_una_curva_normal(cfg, tenores):
    curva = np.linspace(0.03, 0.045, len(tenores))
    resultado = aplicar_escenario(curva, tenores, "paralelo_abajo", cfg)
    assert np.allclose(resultado, curva - 0.02)


# --------------------------------------------------------------------------
# Aplicación e integración
# --------------------------------------------------------------------------

def test_aplicar_escenario_desplaza_la_curva(bundle, cfg, tenores):
    ceros = bundle.curvas.iloc[-1].to_numpy()
    arriba = aplicar_escenario(ceros, tenores, "paralelo_arriba", cfg)
    assert np.allclose(arriba - ceros, cfg.IRRBB_SHOCKS_PB["paralelo"] / 1e4)


def test_curvas_escenarios_devuelve_los_seis(bundle, cfg, tenores):
    ceros = bundle.curvas.iloc[-1].to_numpy()
    tabla = curvas_escenarios(ceros, tenores, cfg)
    assert list(tabla.index) == list(ESCENARIOS)
    assert tabla.shape == (6, len(tenores))


def test_la_magnitud_del_choque_sale_del_config(cfg, tenores):
    """Regla §9.6: los 200 pb no están escritos en el código. Si alguien calibra otra
    moneda, los seis escenarios deben moverse solos."""
    otro = make_config(**{"IRRBB_SHOCKS_PB.paralelo": 400.0})
    assert np.allclose(choque("paralelo_arriba", tenores, otro), 0.04)
    assert np.allclose(choque("paralelo_arriba", tenores, cfg), 0.02)


def test_un_choque_de_cero_no_mueve_la_curva(bundle, tenores):
    nulo = make_config(
        **{"IRRBB_SHOCKS_PB.paralelo": 0.0, "IRRBB_SHOCKS_PB.cortas": 0.0,
           "IRRBB_SHOCKS_PB.largas": 0.0}
    )
    ceros = bundle.curvas.iloc[-1].to_numpy()
    for nombre in ESCENARIOS:
        assert np.allclose(aplicar_escenario(ceros, tenores, nombre, nulo), ceros)
