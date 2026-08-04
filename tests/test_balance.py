"""Tests del Módulo 1 — brecha de repreciación.

Dos familias. La primera comprueba que las bandas y el reparto son aritméticamente
correctos: si el saldo repartido no suma el saldo original, todo el informe es
ficción. La segunda fija bajo test los **hallazgos** del módulo — que el gap por
repreciación no es el gap por vencimiento, y que el gap contractual de este banco
contradice a su propio margen financiero. Un hallazgo que no está bajo test es un
hallazgo que se pierde en el próximo refactor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.balance import (
    asignar_banda,
    bandas_frame,
    gap_repreciacion,
    gap_vencimiento,
    indicadores_gap,
    informe_alco,
    perfil_repreciacion,
    reconciliacion,
)


def _instrumento(**kwargs) -> dict:
    """Cohorte mínima con los campos que el Módulo 1 consume."""
    base = {
        "categoria": "comercial",
        "lado": "activo",
        "saldo": 100.0,
        "tasa": 0.05,
        "amortizacion": "bullet",
        "frecuencia_pago_meses": 1,
        "meses_a_repreciacion": 3,
        "meses_a_vencimiento": 60,
        "sensible": True,
    }
    base.update(kwargs)
    return base


def _cartera(*filas) -> pd.DataFrame:
    df = pd.DataFrame([_instrumento(**f) for f in filas])
    df.index = [f"I-{i:03d}" for i in range(len(df))]
    return df


# --------------------------------------------------------------------------
# Bandas
# --------------------------------------------------------------------------

def test_las_19_bandas_estan_completas(cfg):
    b = bandas_frame(cfg)
    assert len(b) == 19
    assert b["orden"].is_monotonic_increasing
    assert b["punto_medio_a"].is_monotonic_increasing


def test_las_bandas_son_contiguas_y_sin_huecos(cfg):
    b = bandas_frame(cfg)
    assert b["mes_min"].iloc[0] == 0.0
    assert np.isinf(b["mes_max"].iloc[-1])
    assert np.allclose(b["mes_min"].to_numpy()[1:], b["mes_max"].to_numpy()[:-1])


def test_cada_punto_medio_cae_dentro_de_su_banda(cfg):
    """Tolerancia de un día: el punto medio que Basilea prescribe para la banda
    overnight, 0,0028 años, es 1/365 redondeado hacia arriba, y bajo base 30/360 (un
    día = 1/30 de mes) queda tres diezmilésimas de mes por fuera del tope. Se
    conserva el número regulatorio en vez de "arreglarlo": el punto medio es el plazo
    al que el marco estandarizado descuenta, y aquí manda el texto, no la aritmética
    de nuestra convención de días."""
    b = bandas_frame(cfg)
    tolerancia = 1.0 / 30.0
    for codigo, fila in b.iloc[:-1].iterrows():
        pm_meses = fila["punto_medio_a"] * 12
        assert fila["mes_min"] - tolerancia <= pm_meses <= fila["mes_max"] + tolerancia, codigo


def test_todo_plazo_cae_en_exactamente_una_banda(cfg):
    meses = np.arange(0, 601, dtype=float)
    codigos = asignar_banda(meses, cfg)
    assert len(codigos) == len(meses)
    assert set(codigos) <= set(bandas_frame(cfg).index)


def test_los_intervalos_son_cerrados_por_arriba(cfg):
    """Un plazo de 3,0 meses repacta dentro del trimestre: va a 1M–3M, no a 3M–6M."""
    assert asignar_banda(3.0, cfg) == "1m_3m"
    assert asignar_banda(3.01, cfg) == "3m_6m"
    assert asignar_banda(12.0, cfg) == "9m_1a"
    assert asignar_banda(12.5, cfg) == "1a_1_5a"


def test_el_plazo_cero_cae_en_overnight(cfg):
    assert asignar_banda(0.0, cfg) == "overnight"


def test_un_plazo_enorme_cae_en_la_ultima_banda(cfg):
    assert asignar_banda(5000.0, cfg) == "mas_20a"


def test_las_bandas_alco_cubren_las_19(cfg):
    b = bandas_frame(cfg)
    assert not b["banda_alco"].isna().any()
    cubiertas = {c for cs in cfg.BANDAS_ALCO.values() for c in cs}
    assert cubiertas == set(b.index)


def test_un_agrupamiento_incompleto_falla_ruidosamente(cfg):
    from config.params import make_config

    roto = make_config()
    roto.BANDAS_ALCO = {"≤ 1M": ["overnight"]}
    with pytest.raises(ValueError, match="no cubre"):
        bandas_frame(roto)


# --------------------------------------------------------------------------
# Reparto del saldo
# --------------------------------------------------------------------------

def test_el_metodo_bullet_pone_cada_instrumento_en_una_sola_banda(bundle, cfg):
    perfil = perfil_repreciacion(bundle.instrumentos, cfg, metodo="bullet")
    assert perfil.groupby("id").size().max() == 1


def test_el_metodo_amortizado_conserva_el_saldo_de_cada_instrumento(bundle, cfg):
    perfil = perfil_repreciacion(bundle.instrumentos, cfg, metodo="amortizado")
    repartido = perfil.groupby("id")["saldo_en_banda"].sum()
    original = bundle.instrumentos.loc[repartido.index, "saldo"]
    assert np.allclose(repartido.to_numpy(), original.to_numpy(), atol=1e-9)


def test_el_reparto_no_crea_ni_destruye_saldo(bundle, cfg):
    inst = bundle.instrumentos
    for metodo in ("bullet", "amortizado"):
        perfil = perfil_repreciacion(inst, cfg, metodo=metodo)
        por_lado = perfil.groupby("lado")["saldo_en_banda"].sum()
        esperado = inst[inst["sensible"]].groupby("lado")["saldo"].sum()
        for lado in esperado.index:
            assert por_lado[lado] == pytest.approx(esperado[lado], rel=1e-12)


def test_los_no_sensibles_quedan_fuera_del_gap(bundle, cfg):
    """Otros activos, otros pasivos y patrimonio no repactan nunca. Asignarles una
    banda sería inventar exposición que el banco no tiene."""
    perfil = perfil_repreciacion(bundle.instrumentos, cfg)
    no_sensibles = set(bundle.instrumentos.index[~bundle.instrumentos["sensible"]])
    assert not (set(perfil["id"]) & no_sensibles)


def test_una_hipoteca_amortizable_se_reparte_en_varias_bandas(cfg):
    cartera = _cartera(
        dict(categoria="hipotecario", amortizacion="frances", tipo_tasa="fija",
             meses_a_repreciacion=240, meses_a_vencimiento=240, saldo=1000.0, tasa=0.045)
    )
    perfil = perfil_repreciacion(cartera, cfg, metodo="amortizado")
    assert perfil["banda"].nunique() > 5
    assert perfil["saldo_en_banda"].sum() == pytest.approx(1000.0)


def test_un_deposito_a_plazo_cae_en_una_sola_banda(cfg):
    cartera = _cartera(
        dict(categoria="plazo", lado="pasivo", amortizacion="bullet",
             meses_a_repreciacion=9, meses_a_vencimiento=9, saldo=500.0)
    )
    perfil = perfil_repreciacion(cartera, cfg, metodo="amortizado")
    assert len(perfil) == 1
    assert perfil["banda"].iloc[0] == "6m_9m"


def test_un_variable_amortizable_deja_el_remanente_en_su_reset(cfg):
    """Antes del reset sólo está expuesto el principal que se va devolviendo; en el
    reset se expone todo lo que quede."""
    cartera = _cartera(
        dict(categoria="hipotecario", amortizacion="frances", meses_a_repreciacion=6,
             meses_a_vencimiento=240, saldo=1000.0, tasa=0.045)
    )
    perfil = perfil_repreciacion(cartera, cfg, metodo="amortizado")
    en_reset = perfil.loc[perfil["banda"] == "3m_6m", "saldo_en_banda"].sum()
    assert en_reset > 990.0                      # casi todo sigue vivo a los 6 meses
    assert perfil["saldo_en_banda"].sum() == pytest.approx(1000.0)
    assert perfil["banda"].nunique() >= 2         # pero algo ya se devolvió


def test_metodo_y_eje_invalidos_fallan(bundle, cfg):
    with pytest.raises(ValueError, match="método"):
        perfil_repreciacion(bundle.instrumentos, cfg, metodo="inventado")
    with pytest.raises(ValueError, match="eje"):
        perfil_repreciacion(bundle.instrumentos, cfg, eje="inventado")


# --------------------------------------------------------------------------
# Tablas de gap y reconciliación
# --------------------------------------------------------------------------

def test_la_reconciliacion_cuadra(bundle, cfg):
    rec = reconciliacion(bundle.instrumentos, cfg)
    assert rec.loc["Descuadre", "musd"] == pytest.approx(0.0, abs=1e-9)
    total = cfg.BANK_PROFILE["activos_totales_musd"]
    assert rec.loc["TOTAL ACTIVOS", "musd"] == pytest.approx(total)
    assert rec.loc["TOTAL PASIVO + PATRIMONIO", "musd"] == pytest.approx(total)


def test_el_gap_acumulado_final_es_rsa_menos_rsl(bundle, cfg):
    tabla = gap_repreciacion(bundle.instrumentos, cfg)
    rec = reconciliacion(bundle.instrumentos, cfg)
    esperado = rec.loc["Activos sensibles a tasa (RSA)", "musd"] - rec.loc[
        "Pasivos sensibles a tasa (RSL)", "musd"
    ]
    assert tabla["gap_acumulado"].iloc[-1] == pytest.approx(esperado, rel=1e-12)


def test_el_gap_es_activos_menos_pasivos_en_cada_banda(bundle, cfg):
    tabla = gap_repreciacion(bundle.instrumentos, cfg, agrupar=True)
    assert np.allclose(tabla["gap"], tabla["activos"] - tabla["pasivos"])
    assert np.allclose(tabla["gap_acumulado"], tabla["gap"].cumsum())


def test_agrupar_no_cambia_los_totales(bundle, cfg):
    fino = gap_repreciacion(bundle.instrumentos, cfg, agrupar=False)
    grueso = gap_repreciacion(bundle.instrumentos, cfg, agrupar=True)
    for col in ("activos", "pasivos", "gap"):
        assert grueso[col].sum() == pytest.approx(fino[col].sum(), rel=1e-12)


def test_un_balance_calzado_da_gap_cero_en_todas_las_bandas(cfg):
    """§10 — si activo y pasivo repactan igual, no hay riesgo de tasa que medir."""
    cartera = _cartera(
        dict(lado="activo", meses_a_repreciacion=3, meses_a_vencimiento=3, saldo=100.0),
        dict(lado="pasivo", meses_a_repreciacion=3, meses_a_vencimiento=3, saldo=100.0),
        dict(lado="activo", meses_a_repreciacion=36, meses_a_vencimiento=36, saldo=250.0),
        dict(lado="pasivo", meses_a_repreciacion=36, meses_a_vencimiento=36, saldo=250.0),
    )
    tabla = gap_repreciacion(cartera, cfg)
    assert np.allclose(tabla["gap"], 0.0, atol=1e-12)
    assert np.allclose(tabla["gap_acumulado"], 0.0, atol=1e-12)


# --------------------------------------------------------------------------
# Los hallazgos del módulo, bajo test
# --------------------------------------------------------------------------

def test_repreciacion_y_vencimiento_no_son_la_misma_tabla(bundle, cfg):
    """Si coincidieran, el módulo no tendría razón de existir."""
    rep = gap_repreciacion(bundle.instrumentos, cfg, agrupar=True)
    ven = gap_vencimiento(bundle.instrumentos, cfg, agrupar=True)
    assert not np.allclose(rep["gap"].to_numpy(), ven["gap"].to_numpy())


def test_la_cartera_comercial_repacta_mucho_antes_de_vencer(bundle, cfg):
    """El caso de manual: 16.200 M que vencen en años pero repactan cada trimestre.
    Clasificar por vencimiento haría ver largo un activo que es corto."""
    inst = bundle.instrumentos
    comercial = inst[inst["categoria"] == "comercial"]
    assert comercial["meses_a_repreciacion"].median() <= 3
    assert comercial["meses_a_vencimiento"].median() >= 24


def test_el_gap_contractual_a_12m_es_muy_negativo(bundle, cfg):
    """El hallazgo del módulo. Los NMD entran en la banda más corta porque el cliente
    puede retirar mañana, lo que equivale a suponer traspaso del 100%."""
    tabla = gap_repreciacion(bundle.instrumentos, cfg)
    ind = indicadores_gap(tabla, cfg)
    assert ind["gap_12m_sobre_activos"] < -0.15
    assert ind["estado_politica"] == "ALERTA"
    assert ind["rsa_rsl_12m"] < 1.0


def test_el_gap_contradice_al_margen_financiero(bundle, cfg):
    """Los dos números son correctos y apuntan en direcciones opuestas. El mal
    especificado es el gap: mide un banco que traslada el 100% de cada subida a sus
    depósitos a la vista, y este banco traslada el 25%."""
    from src.validation import calibration_metrics

    ind = indicadores_gap(gap_repreciacion(bundle.instrumentos, cfg), cfg)
    delta_nii = calibration_metrics(bundle, cfg)["delta_nii_12m_up200"]
    assert ind["gap_12m_sobre_activos"] < 0 < delta_nii


def test_los_nmd_dominan_la_banda_mas_corta(bundle, cfg):
    """La causa del hallazgo, aislada: 22.000 M de NMD son el grueso del pasivo que
    aparece repactando de inmediato."""
    perfil = perfil_repreciacion(bundle.instrumentos, cfg)
    cortas = perfil[perfil["banda"].isin(["overnight", "on_1m"])]
    pasivo_corto = cortas[cortas["lado"] == "pasivo"]
    nmd = pasivo_corto[pasivo_corto["categoria"].isin(["vista", "ahorro"])]
    assert nmd["saldo_en_banda"].sum() / pasivo_corto["saldo_en_banda"].sum() > 0.8


def test_el_metodo_amortizado_acorta_el_activo_frente_al_bullet(bundle, cfg):
    """16.200 M de cartera amortizable devuelven principal todos los meses. Tratarla
    como bullet la manda entera a las bandas largas y sobrestima el plazo del activo."""
    inst = bundle.instrumentos
    amort = gap_repreciacion(inst, cfg, metodo="amortizado")
    bullet = gap_repreciacion(inst, cfg, metodo="bullet")
    corto = bandas_frame(cfg)["mes_max"] <= 12.0
    assert amort.loc[corto, "activos"].sum() > bullet.loc[corto, "activos"].sum()
    largo = bandas_frame(cfg)["mes_min"] >= 120.0
    assert amort.loc[largo, "activos"].sum() < bullet.loc[largo, "activos"].sum()


def test_el_metodo_no_cambia_los_pasivos(bundle, cfg):
    """Todo el pasivo de este banco es bullet, así que el método de reparto no debe
    tocarlo. Si lo tocara, habría un error en la regla de amortización."""
    inst = bundle.instrumentos
    a = gap_repreciacion(inst, cfg, metodo="amortizado")["pasivos"]
    b = gap_repreciacion(inst, cfg, metodo="bullet")["pasivos"]
    assert np.allclose(a.to_numpy(), b.to_numpy())


# --------------------------------------------------------------------------
# Informe
# --------------------------------------------------------------------------

def test_el_informe_lleva_las_secciones_del_comite(bundle, cfg):
    texto = informe_alco(bundle.instrumentos, cfg, delta_nii_modulo0=0.0172)
    for seccion in (
        "Reconciliación con el balance",
        "clasificado por cuándo repacta",
        "clasificada por vencimiento",
        "Indicadores contra política de ALM",
        "Sensibilidad al método de reparto",
        "Lectura del resultado",
    ):
        assert seccion in texto


def test_el_informe_declara_la_contradiccion(bundle, cfg):
    texto = informe_alco(bundle.instrumentos, cfg, delta_nii_modulo0=0.0172)
    assert "El que está mal especificado es el gap" in texto
    assert "Módulo 2" in texto and "Módulo 3" in texto


def test_el_informe_funciona_sin_el_delta_nii(bundle, cfg):
    texto = informe_alco(bundle.instrumentos, cfg)
    assert "Gap acumulado a 12 meses" in texto
