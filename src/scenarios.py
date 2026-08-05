"""Los seis escenarios de choque de IRRBB — definidos **una sola vez** (§9.5).

Este módulo es la única fuente de verdad sobre qué significa "paralelo arriba" en
este proyecto. Los Módulos 3 y 4 lo consumen por igual, y esa unicidad no es estética:
si el NII y el EVE se calcularan bajo definiciones distintas del mismo escenario, la
tabla final del comité —que los pone en la misma fila— compararía peras con manzanas
sin que nadie lo notara.

Los seis (Comité de Basilea, 2016):

    1. paralelo_arriba      4. aplanamiento
    2. paralelo_abajo       5. cortas_arriba
    3. empinamiento         6. cortas_abajo

Los cuatro últimos se construyen con un escalar que reparte el choque entre tramos,
``S(t) = e^{−t/x}``, que vale 1 en el plazo inmediato y se apaga hacia el largo.

Todos se aplican como **choques instantáneos** sobre la curva cero, seguidos de un
suelo post-choque.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "ESCENARIOS",
    "escalares",
    "choque",
    "piso_post_choque",
    "aplicar_escenario",
    "curvas_escenarios",
]

ESCENARIOS = (
    "paralelo_arriba",
    "paralelo_abajo",
    "empinamiento",
    "aplanamiento",
    "cortas_arriba",
    "cortas_abajo",
)

ETIQUETAS = {
    "paralelo_arriba": "Paralelo arriba",
    "paralelo_abajo": "Paralelo abajo",
    "empinamiento": "Empinamiento",
    "aplanamiento": "Aplanamiento",
    "cortas_arriba": "Cortas arriba",
    "cortas_abajo": "Cortas abajo",
}


def escalares(tenores_a, cfg) -> tuple[np.ndarray, np.ndarray]:
    """Escalares de tramo corto y largo del estándar.

        S_corto(t) = e^{−t/x}        S_largo(t) = 1 − S_corto(t)

    Con ``x = 4`` años, el escalar corto vale 1 en el plazo inmediato, ~0,78 al año,
    ~0,29 a los cinco años y ~0,08 a los diez. Es lo que hace que un choque "de
    cortas" sea de cortas y no un paralelo disfrazado.

    Args:
        tenores_a: Plazos en años.
        cfg: Configuración.

    Returns:
        Tupla ``(S_corto, S_largo)``.
    """
    t = np.asarray(tenores_a, dtype=float)
    s_corto = np.exp(-t / cfg.IRRBB_SHOCK_PARAMS["x_decaimiento_a"])
    return s_corto, 1.0 - s_corto


def choque(nombre: str, tenores_a, cfg) -> np.ndarray:
    """Desplazamiento en tasa que un escenario aplica a cada plazo.

    Args:
        nombre: Uno de :data:`ESCENARIOS`.
        tenores_a: Plazos en años.
        cfg: Configuración.

    Returns:
        Arreglo de desplazamientos en tasa (no en pb), del mismo tamaño que
        ``tenores_a``.

    Raises:
        ValueError: Si el escenario no existe.
    """
    if nombre not in ESCENARIOS:
        raise ValueError(f"escenario desconocido: {nombre!r}; los válidos son {ESCENARIOS}")

    s_corto, s_largo = escalares(tenores_a, cfg)
    r_par = cfg.IRRBB_SHOCKS_PB["paralelo"] / 1e4
    r_cor = cfg.IRRBB_SHOCKS_PB["cortas"] / 1e4
    r_lar = cfg.IRRBB_SHOCKS_PB["largas"] / 1e4
    p = cfg.IRRBB_SHOCK_PARAMS

    if nombre == "paralelo_arriba":
        return np.full(len(s_corto), r_par)
    if nombre == "paralelo_abajo":
        return np.full(len(s_corto), -r_par)
    if nombre == "cortas_arriba":
        return r_cor * s_corto
    if nombre == "cortas_abajo":
        return -r_cor * s_corto
    if nombre == "empinamiento":
        return -p["empinamiento_corto"] * r_cor * s_corto + p["empinamiento_largo"] * r_lar * s_largo
    return p["aplanamiento_corto"] * r_cor * s_corto - p["aplanamiento_largo"] * r_lar * s_largo


def piso_post_choque(tenores_a, cfg) -> np.ndarray:
    """Suelo que la curva no puede perforar tras el choque.

    Empieza en −100 pb en el plazo inmediato y sube 5 pb por año hasta 0% a los 20
    años. Que el suelo sea **negativo** y no cero es reconocimiento explícito de que
    las tasas nominales negativas ocurrieron de verdad; el estándar no las declara
    imposibles, sólo acota cuánto.

    Sin este suelo, el escenario de bajada sobre una curva ya baja produciría tasas
    profundamente negativas y un EVE que sería un artefacto aritmético.

    Args:
        tenores_a: Plazos en años.
        cfg: Configuración.

    Returns:
        Nivel mínimo admisible por plazo.
    """
    f = cfg.POST_SHOCK_FLOOR
    t = np.asarray(tenores_a, dtype=float)
    nivel = (f["nivel_inicial_pb"] + f["pendiente_pb_por_ano"] * t) / 1e4
    return np.minimum(nivel, f["tope"])


def aplicar_escenario(ceros, tenores_a, nombre: str, cfg) -> np.ndarray:
    """Curva cero tras aplicar el choque y el suelo post-choque.

    Args:
        ceros: Tasas cero de la curva base, en composición continua.
        tenores_a: Plazos en años, en el mismo orden.
        nombre: Uno de :data:`ESCENARIOS`.
        cfg: Configuración.

    Returns:
        Curva cero desplazada.
    """
    ceros = np.asarray(ceros, dtype=float)
    return np.maximum(ceros + choque(nombre, tenores_a, cfg), piso_post_choque(tenores_a, cfg))


def curvas_escenarios(ceros, tenores_a, cfg, escenarios=None) -> pd.DataFrame:
    """Todas las curvas desplazadas, una fila por escenario.

    Args:
        ceros: Curva cero base.
        tenores_a: Plazos en años.
        cfg: Configuración.
        escenarios: Subconjunto a calcular; por defecto los seis.

    Returns:
        DataFrame indexado por escenario, con una columna por plazo.
    """
    nombres = list(escenarios if escenarios is not None else ESCENARIOS)
    datos = {n: aplicar_escenario(ceros, tenores_a, n, cfg) for n in nombres}
    return pd.DataFrame(datos, index=[f"{t:g}" for t in tenores_a]).T
