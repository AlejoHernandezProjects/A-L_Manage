"""Tests de las utilidades de curva.

El primero de §10 ("un bono a tasa par vale su nominal") es el test de humo de
cualquier librería de valoración: si falla, todo el EVE del Módulo 4 está mal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.curves import (
    amortization_schedule,
    discount_factors,
    interpolate_zero,
    macaulay_duration,
    nelson_siegel,
    par_rate,
    present_value,
    price_fixed_bond,
    schedule_cashflows,
    solve_nelson_siegel_anchored,
    year_fraction_30_360,
)

TENORES = np.array([1 / 12, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0])


def curva_normal() -> np.ndarray:
    """Curva empinada típica: 2% en el tramo corto, 4% en el largo."""
    b0, b1 = solve_nelson_siegel_anchored(0.02, 0.04, 0.25, 10.0, 0.005, 2.5)
    return nelson_siegel(TENORES, b0, b1, 0.005, 2.5)


def curva_invertida() -> np.ndarray:
    """Curva invertida: 5,25% a 3M, 4,45% a 10A (el régimen de meses 55–81)."""
    b0, b1 = solve_nelson_siegel_anchored(0.0525, 0.0445, 0.25, 10.0, 0.0, 2.5)
    return nelson_siegel(TENORES, b0, b1, 0.0, 2.5)


# --------------------------------------------------------------------------
# Convención de días
# --------------------------------------------------------------------------

def test_30_360_un_anio_exacto():
    assert year_fraction_30_360(pd.Timestamp("2020-01-15"), pd.Timestamp("2021-01-15")) == pytest.approx(1.0)


def test_30_360_un_mes_vale_un_doceavo():
    assert year_fraction_30_360(pd.Timestamp("2020-03-15"), pd.Timestamp("2020-04-15")) == pytest.approx(1 / 12)


# --------------------------------------------------------------------------
# Nelson-Siegel y anclaje
# --------------------------------------------------------------------------

@pytest.mark.parametrize("y_corto,y_largo", [(0.02, 0.04), (0.0525, 0.0445), (0.001, 0.001)])
def test_anclaje_es_exacto(y_corto, y_largo):
    """El sistema 2x2 debe reproducir las dos anclas al centésimo de pb."""
    b2, lam = 0.004, 2.5
    b0, b1 = solve_nelson_siegel_anchored(y_corto, y_largo, 0.25, 10.0, b2, lam)
    assert nelson_siegel(0.25, b0, b1, b2, lam) == pytest.approx(y_corto, abs=1e-10)
    assert nelson_siegel(10.0, b0, b1, b2, lam) == pytest.approx(y_largo, abs=1e-10)


def test_anclaje_independiente_de_la_curvatura():
    """β₂ da forma al tramo intermedio pero no puede mover las anclas."""
    for b2 in (-0.01, 0.0, 0.01):
        b0, b1 = solve_nelson_siegel_anchored(0.03, 0.045, 0.25, 10.0, b2, 2.5)
        assert nelson_siegel(0.25, b0, b1, b2, 2.5) == pytest.approx(0.03, abs=1e-10)


# --------------------------------------------------------------------------
# Interpolación y descuento
# --------------------------------------------------------------------------

def test_interpolacion_reproduce_los_nodos():
    ceros = curva_normal()
    assert interpolate_zero(TENORES, ceros, TENORES) == pytest.approx(ceros)


def test_extrapolacion_plana_fuera_de_rango():
    ceros = curva_normal()
    assert interpolate_zero(TENORES, ceros, 0.01) == pytest.approx(ceros[0])
    assert interpolate_zero(TENORES, ceros, 40.0) == pytest.approx(ceros[-1])


@pytest.mark.parametrize("curva", [curva_normal, curva_invertida])
def test_factores_de_descuento_positivos_y_decrecientes(curva):
    """Control #6. Se prueba también con la curva invertida: es ahí donde una
    interpolación lineal en y(τ) — en vez de en y(τ)·τ — produciría forwards
    negativos y rompería la monotonía."""
    taus = np.linspace(0.01, 30.0, 500)
    dfs = discount_factors(TENORES, curva(), taus)
    assert np.all(dfs > 0)
    assert np.all(np.diff(dfs) < 0)


# --------------------------------------------------------------------------
# §10 — un bono a tasa par vale su nominal
# --------------------------------------------------------------------------

@pytest.mark.parametrize("curva", [curva_normal, curva_invertida])
@pytest.mark.parametrize("plazo_a", [1.0, 2.0, 5.0, 10.0, 30.0])
def test_bono_a_la_par_vale_su_nominal(curva, plazo_a):
    ceros = curva()
    c = par_rate(TENORES, ceros, plazo_a, frecuencia_pago_meses=6)
    precio = price_fixed_bond(100.0, c, plazo_a, TENORES, ceros, frecuencia_pago_meses=6)
    assert precio == pytest.approx(100.0, abs=1e-8)


def test_cupon_alto_cotiza_sobre_la_par():
    ceros = curva_normal()
    c = par_rate(TENORES, ceros, 5.0)
    assert price_fixed_bond(100.0, c + 0.01, 5.0, TENORES, ceros) > 100.0
    assert price_fixed_bond(100.0, c - 0.01, 5.0, TENORES, ceros) < 100.0


# --------------------------------------------------------------------------
# Flujos y duración
# --------------------------------------------------------------------------

def test_frances_cuota_constante_y_amortiza_el_nominal():
    tiempos, montos = schedule_cashflows(100.0, 0.06, 120, "frances", 1)
    assert len(tiempos) == 120
    assert np.allclose(montos, montos[0])
    # La suma de cuotas excede el nominal exactamente en los intereses pagados.
    assert montos.sum() > 100.0
    # Descontado a la propia tasa del préstamo, el VP es el nominal.
    ceros_planos = np.full_like(TENORES, np.log(1 + 0.06 / 12) * 12)
    assert present_value(tiempos, montos, TENORES, ceros_planos) == pytest.approx(100.0, rel=1e-9)


def test_bullet_devuelve_principal_al_final():
    tiempos, montos = schedule_cashflows(100.0, 0.05, 24, "bullet", 6)
    assert len(tiempos) == 4
    assert montos[0] == pytest.approx(100.0 * 0.05 * 0.5)
    assert montos[-1] == pytest.approx(100.0 * 0.05 * 0.5 + 100.0)


def test_duracion_bullet_cupon_cero_es_el_plazo():
    ceros = curva_normal()
    tiempos, montos = schedule_cashflows(100.0, 0.0, 60, "bullet", 60)
    assert macaulay_duration(tiempos, montos, TENORES, ceros) == pytest.approx(5.0)


def test_frances_dura_mucho_menos_que_bullet():
    """El punto que justifica modelar hipotecas como amortizables: a 20 años, la
    duración de un préstamo francés es una fracción de la de un bullet."""
    ceros = curva_normal()
    d_fra = macaulay_duration(*schedule_cashflows(100.0, 0.045, 240, "frances", 1), TENORES, ceros)
    d_bul = macaulay_duration(*schedule_cashflows(100.0, 0.045, 240, "bullet", 1), TENORES, ceros)
    assert d_fra < 0.7 * d_bul  # ≈ 8,6 a contra ≈ 13,2 a
    assert 6.0 < d_fra < 9.5


def test_el_principal_amortizado_suma_el_nominal():
    for amort, freq in (("frances", 1), ("bullet", 6), ("frances", 3)):
        _, principal = amortization_schedule(100.0, 0.06, 120, amort, freq)
        assert principal.sum() == pytest.approx(100.0)


def test_un_bullet_devuelve_todo_el_principal_al_final():
    meses, principal = amortization_schedule(100.0, 0.05, 24, "bullet", 6)
    assert np.allclose(principal[:-1], 0.0)
    assert principal[-1] == pytest.approx(100.0)
    assert meses[-1] == 24


def test_el_principal_frances_crece_con_cada_cuota():
    """En cuota nivelada la parte de interés cae y la de principal sube. Es lo que
    hace que la exposición de una hipoteca se concentre más tarde de lo que sugiere
    un reparto lineal, pero mucho antes de lo que sugiere tratarla como bullet."""
    _, principal = amortization_schedule(100.0, 0.06, 120, "frances", 1)
    assert np.all(np.diff(principal) > 0)


def test_el_principal_frances_llega_pronto_en_parte():
    """A los 12 meses de una hipoteca a 20 años ya se devolvió algo de principal —
    poco, pero no cero. Tratarla como bullet dice que es exactamente cero."""
    meses, principal = amortization_schedule(100.0, 0.045, 240, "frances", 1)
    devuelto_1a = principal[meses <= 12].sum()
    assert 1.0 < devuelto_1a < 5.0


def test_sin_plazo_no_hay_calendario():
    meses, principal = amortization_schedule(100.0, 0.05, 0, "frances", 1)
    assert len(meses) == 0 and len(principal) == 0


def test_duracion_crece_con_el_plazo():
    ceros = curva_normal()
    duraciones = [
        macaulay_duration(*schedule_cashflows(100.0, 0.04, m, "bullet", 6), TENORES, ceros)
        for m in (12, 36, 60, 120, 240)
    ]
    assert all(np.diff(duraciones) > 0)
