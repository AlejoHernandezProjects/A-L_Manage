"""Módulo 5 — el documento de alcance, bajo test.

Un README que se desincroniza del código es peor que no tenerlo: da confianza falsa.
Estos tests no comprueban prosa, comprueban que el documento **siga cubriendo** lo que
§4 y §7 del contrato del proyecto exigen, y que las cifras que titula no se hayan
quedado atrás respecto de lo que el código produce hoy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.deposits import instrumentos_conductuales, modelo_nmd
from src.eve import cartera_valorable, tabla_delta_eve
from src.nii import tabla_delta_nii
from src.scenarios import ESCENARIOS

RAIZ = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def readme() -> str:
    ruta = RAIZ / "README.md"
    assert ruta.exists(), "§4 exige un README con problema, supuestos, resultados y alcance"
    return ruta.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Cobertura exigida por el contrato del proyecto
# --------------------------------------------------------------------------

def test_cubre_las_cuatro_secciones_de_la_seccion_4(readme):
    for tema in ("El problema", "Resultado principal", "El banco sintético", "Alcance"):
        assert tema in readme, f"falta la sección «{tema}»"


def test_documenta_todo_lo_que_la_seccion_7_pide_explicar(readme):
    """§7 nombra cinco temas fuera de alcance que hay que poder defender en entrevista."""
    for tema in ("prepago", "Retiro anticipado", "Riesgo de base", "Multi-moneda", "SR 11-7"):
        assert tema in readme, f"§7 exige explicar «{tema}» y no aparece"


def test_documenta_la_capa_de_gobernanza(readme):
    for pieza in ("límites", "Validación independiente", "Backtesting", "auditable"):
        assert pieza.lower() in readme.lower(), f"falta la pieza de gobernanza «{pieza}»"


def test_nombra_los_seis_escenarios(readme):
    for etiqueta in ("Paralelo arriba", "Paralelo abajo", "Empinamiento",
                     "Aplanamiento", "Cortas arriba", "Cortas abajo"):
        assert etiqueta in readme
    assert len(ESCENARIOS) == 6


def test_reconoce_los_objetivos_incumplidos(readme):
    """La pre-registración solo vale si los incumplimientos se reportan. Un README que
    solo enseñara los objetivos alcanzados sería propaganda, no documentación."""
    assert "no se cumplen" in readme or "no se movió la meta" in readme
    assert "fuera" in readme


def test_explica_por_que_el_banco_es_sintetico(readme):
    assert "ground_truth" in readme
    assert "error" in readme


# --------------------------------------------------------------------------
# Las cifras que el README titula no pueden haberse quedado atrás
# --------------------------------------------------------------------------

def _normalizar(texto: str) -> str:
    """Iguala el menos tipográfico del texto con el guion ASCII que produce el código."""
    return texto.replace("−", "-").replace(".", ",")


def test_el_delta_nii_titulado_coincide_con_el_codigo(readme, bundle, cfg):
    modelo = modelo_nmd(bundle, cfg)
    tabla = tabla_delta_nii(bundle, cfg, modelo, escenarios=["paralelo_arriba"])
    valor = tabla.loc["paralelo_arriba", "delta_pct"]
    texto = _normalizar(f"{valor:+.2%}")
    assert texto in _normalizar(readme), f"el README no refleja el ΔNII actual ({texto})"


def test_el_delta_eve_titulado_coincide_con_el_codigo(readme, bundle, cfg):
    modelo = modelo_nmd(bundle, cfg)
    ceros = bundle.curvas.iloc[-1].to_numpy()
    inst = instrumentos_conductuales(bundle.instrumentos, modelo, cfg)
    tabla = tabla_delta_eve(cartera_valorable(inst, ceros, cfg), ceros, cfg)
    valor = tabla.loc["paralelo_arriba", "delta_sobre_tier1"]
    texto = _normalizar(f"{valor:+.1%}")
    assert texto in _normalizar(readme), f"el README no refleja el ΔEVE actual ({texto})"


def test_el_umbral_del_outlier_test_es_el_del_config(readme, cfg):
    umbral = 0.15 * cfg.BANK_PROFILE["tier1_musd"]
    assert f"{umbral:,.0f}".replace(",", ".") in readme
