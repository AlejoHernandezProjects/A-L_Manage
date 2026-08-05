"""Tests del Módulo 2 — modelo conductual de NMD.

Tres familias:

1. **Recuperación**: los estimadores encuentran el ground truth dentro de un margen.
   Es lo único que distingue un modelo validado de uno que sólo produce números.
2. **No identificabilidad**: la vida del núcleo NO se recupera, y eso queda fijado
   bajo test. Es un hallazgo, no un fallo, y perderlo en un refactor sería perder el
   argumento de por qué existen los topes regulatorios.
3. **Aritmética**: el reparto conserva saldos y el balance sigue cuadrando. Si esto
   falla, el gap corregido es ficción.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.balance import bandas_frame, gap_repreciacion, indicadores_gap
from src.deposits import (
    PRODUCTOS_NMD,
    diagnostico_identificabilidad,
    estimar_beta_asimetrica,
    estimar_proporcion_estable,
    informe_nmd,
    instrumentos_conductuales,
    modelo_nmd,
    nucleo_irrbb,
    perfil_bandas_nmd,
    plazo_replica,
    sensibilidad_nmd,
)


@pytest.fixture(scope="module")
def modelo(bundle, cfg):
    return modelo_nmd(bundle, cfg)


@pytest.fixture(scope="module")
def sens(bundle, cfg):
    """El barrido completo es caro (doce modelos con su gap). Se corre una vez."""
    return sensibilidad_nmd(bundle, cfg)


# --------------------------------------------------------------------------
# 1. Recuperación de la beta asimétrica
# --------------------------------------------------------------------------

def test_las_betas_recuperan_el_ground_truth(bundle, cfg):
    est = estimar_beta_asimetrica(bundle.ref_rate, bundle.tasas_deposito, cfg)
    for prod, f in est.iterrows():
        assert abs(f["error_up"]) < 0.06, f"{prod} β⁺ error {f['error_up']:+.3f}"
        assert abs(f["error_down"]) < 0.06, f"{prod} β⁻ error {f['error_down']:+.3f}"


def test_la_asimetria_se_recupera_en_la_direccion_correcta(bundle, cfg):
    """β̂⁻ > β̂⁺: el banco traslada más de una bajada que de una subida. El estimador
    lo ve sin que se le haya dicho que la asimetría existe."""
    est = estimar_beta_asimetrica(bundle.ref_rate, bundle.tasas_deposito, cfg)
    for prod in ("vista", "ahorro"):
        assert est.loc[prod, "beta_down_est"] > est.loc[prod, "beta_up_est"]


def test_las_betas_estimadas_son_significativas(bundle, cfg):
    est = estimar_beta_asimetrica(bundle.ref_rate, bundle.tasas_deposito, cfg)
    assert (est["t_up"] > 3).all()
    assert (est["t_down"].abs() > 3).all()


def test_el_orden_entre_productos_se_preserva(bundle, cfg):
    est = estimar_beta_asimetrica(bundle.ref_rate, bundle.tasas_deposito, cfg)
    assert est.loc["vista", "beta_up_est"] < est.loc["ahorro", "beta_up_est"]
    assert est.loc["ahorro", "beta_up_est"] < est.loc["plazo", "beta_up_est"]


def test_la_beta_ingenua_no_coincide_con_ninguna_verdadera(bundle, cfg):
    """El contraste que justifica el estimador asimétrico: la OLS sobre niveles cae
    entre β⁺ y β⁻ sin ser ninguna, porque es un promedio ponderado por la frecuencia
    de subidas y bajadas de esta muestra concreta."""
    est = estimar_beta_asimetrica(bundle.ref_rate, bundle.tasas_deposito, cfg)
    for prod in ("vista", "ahorro"):
        f = est.loc[prod]
        assert f["beta_up_real"] < f["beta_ols_niveles"] < f["beta_down_real"]
        assert abs(f["beta_ols_niveles"] - f["beta_up_real"]) > 0.05


# --------------------------------------------------------------------------
# 2. Proporción estable
# --------------------------------------------------------------------------

def test_la_proporcion_estable_es_razonable(bundle, cfg):
    est = estimar_proporcion_estable(bundle.saldos, cfg)
    for prod, f in est.iterrows():
        assert 0.7 < f["estable_minimo"] < 1.0
        assert abs(f["error_vs_core_share"]) < 0.10


def test_el_crecimiento_de_la_tendencia_se_recupera(bundle, cfg):
    """Si el destendenciado no recuperara el 4,5% anual, la proporción estable
    estaría midiendo tendencia en vez de volatilidad."""
    est = estimar_proporcion_estable(bundle.saldos, cfg)
    for prod, f in est.iterrows():
        assert abs(f["crecimiento_anual_est"] - f["crecimiento_anual_real"]) < 0.02


def test_el_percentil_es_menos_severo_que_el_minimo(bundle, cfg):
    est = estimar_proporcion_estable(bundle.saldos, cfg)
    assert (est["estable_percentil"] >= est["estable_minimo"]).all()


# --------------------------------------------------------------------------
# 3. No identificabilidad — el hallazgo, bajo test
# --------------------------------------------------------------------------

def test_la_vida_del_nucleo_no_es_identificable(cfg):
    """Vidas verdaderas de 3 y de 10 años producen la misma serie observable. No es
    identificación débil: es información cero. Éste es el argumento de por qué IRRBB
    acota el plazo del núcleo en vez de pedir una estimación mejor."""
    diag = diagnostico_identificabilidad(cfg, vidas=[3.0, 10.0])
    assert (diag["correlacion_vs_base"] > 0.999).all()
    assert (diag["dif_relativa_max"] < 0.01).all()


def test_el_saldo_medio_no_depende_de_la_vida(cfg):
    diag = diagnostico_identificabilidad(cfg, vidas=[3.0, 10.0])
    medias = diag["saldo_medio"].to_numpy()
    assert abs(medias[0] / medias[1] - 1) < 0.005


def test_el_plazo_de_replica_mide_otra_cosa(bundle, cfg):
    """El portafolio de réplica da un plazo de meses frente a una vida de volumen de
    años. Los dos números son correctos y responden preguntas distintas: el precio de
    un depósito repacta rápido aunque el saldo permanezca. Usar el de réplica para
    asignar bandas de EVE sería el error que este test documenta."""
    rep = plazo_replica(bundle.ref_rate, bundle.tasas_deposito, cfg)
    for prod in PRODUCTOS_NMD:
        plazo = rep.loc[prod, "plazo_replica_a"]
        vida = cfg.NMD_PARAMS[prod]["vida_promedio_a"]
        assert plazo < 1.5
        assert plazo < vida / 2


def test_el_traspaso_implicito_de_la_replica_es_creciente(bundle, cfg):
    rep = plazo_replica(bundle.ref_rate, bundle.tasas_deposito, cfg)
    assert rep.loc["vista", "traspaso_implicito"] < rep.loc["ahorro", "traspaso_implicito"]
    assert rep.loc["ahorro", "traspaso_implicito"] < rep.loc["plazo", "traspaso_implicito"]


# --------------------------------------------------------------------------
# 4. Núcleo IRRBB y topes
# --------------------------------------------------------------------------

def test_el_nucleo_es_estable_por_uno_menos_beta(cfg):
    n = nucleo_irrbb("vista", 0.90, 0.25, cfg, vida_supuesta=4.0)
    assert n["nucleo_bruto"] == pytest.approx(0.90 * 0.75)
    assert n["nucleo"] + n["no_nucleo"] == pytest.approx(1.0)


def test_el_tope_de_proporcion_muerde_cuando_toca(cfg):
    """Un depósito muy estable con beta muy baja daría un núcleo del 95%; el marco lo
    recorta al 90% para minorista transaccional."""
    n = nucleo_irrbb("vista", 0.99, 0.04, cfg)
    assert n["nucleo_bruto"] > n["cap_nucleo"]
    assert n["nucleo"] == pytest.approx(n["cap_nucleo"])
    assert n["tope_nucleo_muerde"]


def test_el_tope_de_plazo_muerde_cuando_toca(cfg):
    n = nucleo_irrbb("vista", 0.90, 0.25, cfg, vida_supuesta=7.0)
    assert n["vida_a"] == pytest.approx(5.0)
    assert n["tope_vida_muerde"]


def test_ahorro_tiene_topes_mas_estrictos_que_vista(cfg):
    """Minorista no transaccional: 70% y 4,5 años, contra 90% y 5 años. El saldo que
    compara precio recibe menos crédito conductual."""
    v = nucleo_irrbb("vista", 0.99, 0.04, cfg, vida_supuesta=9.0)
    a = nucleo_irrbb("ahorro", 0.99, 0.04, cfg, vida_supuesta=9.0)
    assert a["nucleo"] < v["nucleo"]
    assert a["vida_a"] < v["vida_a"]


def test_los_topes_del_caso_base_no_muerden(modelo):
    """Este banco está calibrado por dentro de los topes. Que no muerdan es
    información: el supervisor no está siendo la restricción activa aquí."""
    for n in modelo["nucleos"].values():
        assert not n["tope_nucleo_muerde"]
        assert not n["tope_vida_muerde"]


# --------------------------------------------------------------------------
# 5. Perfil de bandas
# --------------------------------------------------------------------------

def test_el_perfil_suma_el_saldo(cfg):
    n = nucleo_irrbb("vista", 0.90, 0.25, cfg, vida_supuesta=4.0)
    perfil = perfil_bandas_nmd(11_000.0, n, cfg)
    assert perfil.sum() == pytest.approx(11_000.0)
    assert (perfil >= 0).all()


def test_el_no_nucleo_va_entero_a_la_banda_mas_corta(cfg):
    n = nucleo_irrbb("vista", 0.90, 0.25, cfg, vida_supuesta=4.0)
    perfil = perfil_bandas_nmd(11_000.0, n, cfg)
    assert perfil.iloc[0] >= 11_000.0 * n["no_nucleo"]


def test_el_plazo_medio_del_nucleo_es_la_vida_asumida(cfg):
    """El decaimiento exponencial con λ = 1/vida tiene plazo medio exactamente igual a
    la vida — que es la magnitud sobre la que actúa el tope regulatorio. Se comprueba
    con los puntos medios de banda, así que la tolerancia absorbe la discretización."""
    bandas = bandas_frame(cfg)
    pm = bandas["punto_medio_a"].to_numpy()
    for vida in (2.0, 4.0, 5.0):
        n = nucleo_irrbb("vista", 1.0, 0.0, cfg, vida_supuesta=vida)
        perfil = perfil_bandas_nmd(1.0, n, cfg).to_numpy(copy=True)
        perfil[0] -= n["no_nucleo"]
        plazo_medio = float((perfil * pm).sum() / perfil.sum())
        assert plazo_medio == pytest.approx(vida, rel=0.20)


def test_una_vida_mas_larga_alarga_el_perfil(cfg):
    bandas = bandas_frame(cfg)
    pm = bandas["punto_medio_a"].to_numpy()
    plazos = []
    for vida in (2.0, 3.0, 5.0):
        n = nucleo_irrbb("vista", 0.90, 0.25, cfg, vida_supuesta=vida)
        p = perfil_bandas_nmd(1.0, n, cfg).to_numpy()
        plazos.append(float((p * pm).sum()))
    assert all(np.diff(plazos) > 0)


# --------------------------------------------------------------------------
# 6. Instrumentos conductuales y gap corregido
# --------------------------------------------------------------------------

def test_los_instrumentos_conductuales_conservan_el_balance(bundle, cfg, modelo):
    inst = instrumentos_conductuales(bundle.instrumentos, modelo, cfg)
    total = cfg.BANK_PROFILE["activos_totales_musd"]
    a = inst.loc[inst["lado"] == "activo", "saldo"].sum()
    pk = inst.loc[inst["lado"] != "activo", "saldo"].sum()
    assert a == pytest.approx(total, rel=1e-10)
    assert pk == pytest.approx(total, rel=1e-10)


def test_los_nmd_se_reparten_en_varias_bandas(bundle, cfg, modelo):
    """Antes eran dos cohortes en la banda overnight; ahora son tramos repartidos."""
    inst = instrumentos_conductuales(bundle.instrumentos, modelo, cfg)
    nmd = inst[inst["es_nmd"]]
    assert len(nmd) > 20
    assert nmd["saldo"].sum() == pytest.approx(22_000.0, rel=1e-9)


def test_cada_tramo_cae_en_la_banda_que_le_toca(bundle, cfg, modelo):
    """El plazo representativo de cada banda debe reasignarse a esa misma banda. Si
    se colara a la vecina, el perfil conductual se deformaría en silencio."""
    from src.balance import asignar_banda

    inst = instrumentos_conductuales(bundle.instrumentos, modelo, cfg)
    nmd = inst[inst["es_nmd"]]
    for id_, fila in nmd.iterrows():
        banda_destino = id_.split("-", 1)[1]
        assert asignar_banda(fila["meses_a_repreciacion"], cfg) == banda_destino


def test_el_gap_conductual_sale_de_alerta(bundle, cfg, modelo):
    """El cierre del arco que abrió el Módulo 1: el descalce a 12 meses pasa de
    −20,6% de los activos (ALERTA) a esencialmente calzado."""
    inst = instrumentos_conductuales(bundle.instrumentos, modelo, cfg)
    contractual = indicadores_gap(gap_repreciacion(bundle.instrumentos, cfg), cfg)
    conductual = indicadores_gap(gap_repreciacion(inst, cfg), cfg)

    assert contractual["estado_politica"] == "ALERTA"
    assert conductual["estado_politica"] == "DENTRO DE POLÍTICA"
    assert abs(conductual["gap_12m_sobre_activos"]) < abs(contractual["gap_12m_sobre_activos"]) / 5


def test_el_gap_conductual_sigue_sin_predecir_el_signo_del_margen(bundle, cfg, modelo):
    """Honestidad sobre el alcance: el modelo conductual arregla el *momento* del
    repricing y la *proporción* que repacta, pero cualquier análisis de brechas sigue
    tratando cada peso dentro de su banda como traspaso del 100%. El signo del margen
    sólo sale de proyectarlo — Módulo 3."""
    from src.validation import calibration_metrics

    inst = instrumentos_conductuales(bundle.instrumentos, modelo, cfg)
    gap = indicadores_gap(gap_repreciacion(inst, cfg), cfg)["gap_12m_sobre_activos"]
    delta_nii = calibration_metrics(bundle, cfg)["delta_nii_12m_up200"]
    assert abs(gap) < 0.05          # esencialmente calzado
    assert delta_nii > 0            # y aun así el margen mejora


# --------------------------------------------------------------------------
# 7. Sensibilidad
# --------------------------------------------------------------------------

def test_la_sensibilidad_es_monotona_en_la_vida_supuesta(sens):
    """Un núcleo más largo saca más pasivo de las bandas cortas y mejora el gap. Si no
    fuera monótono, el reparto estaría mal."""
    assert sens["vida"]["gap_12m_musd"].is_monotonic_increasing


def test_el_tope_aparece_en_la_sensibilidad(sens):
    """Con vida supuesta de 7 años el tope de 5 debe activarse; con 4, no."""
    v = sens["vida"]
    assert not bool(v.loc[4.0, "tope_muerde"])
    assert bool(v.loc[7.0, "tope_muerde"])
    assert v.loc[7.0, "vida_vista_aplicada_a"] == pytest.approx(5.0)


def test_el_supuesto_no_identificado_mueve_el_resultado(sens):
    """La razón por la que la sensibilidad es el entregable y no un apéndice: el
    parámetro que no se puede estimar mueve el resultado de forma material."""
    rango = sens["vida"]["gap_12m_sobre_activos"]
    assert rango.max() - rango.min() > 0.05


def test_una_beta_de_largo_plazo_da_un_nucleo_menor(sens):
    """β̄ > β⁺, así que recorta más núcleo y deja al banco más descalzado en el papel.
    Elegir la beta del corte no es un detalle técnico: mueve el resultado."""
    b = sens["beta"]
    assert b.loc["beta_largo", "nucleo_vista"] < b.loc["beta_up", "nucleo_vista"]
    assert b.loc["beta_largo", "gap_12m_musd"] < b.loc["beta_up", "gap_12m_musd"]


# --------------------------------------------------------------------------
# 8. Informe
# --------------------------------------------------------------------------

def test_el_informe_lleva_las_secciones(bundle, cfg):
    texto = informe_nmd(bundle, cfg, delta_nii=0.0172)
    for seccion in (
        "Beta asimétrica",
        "Proporción estable",
        "El plazo del núcleo no es estimable",
        "Tres plazos distintos",
        "marco estandarizado IRRBB",
        "El gap corregido",
        "Sensibilidad",
    ):
        assert seccion in texto


def test_el_informe_no_sobreafirma_el_cierre(bundle, cfg):
    """El gap conductual queda cerca de cero, no positivo. El informe debe decir que
    un gap ya bien especificado sigue sin predecir el signo del margen."""
    texto = informe_nmd(bundle, cfg, delta_nii=0.0172)
    assert "no sobreafirmar" in texto
    assert "Módulo 3" in texto
