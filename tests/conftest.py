"""Fixtures compartidas.

El dataset se genera una sola vez por sesión: son ~3.000 cohortes y 120 meses de
series, y regenerarlo en cada test multiplicaría el tiempo de la suite sin añadir
información. Los tests que necesitan corromper los datos piden ``bundle_mutable``,
que entrega una copia.
"""

from __future__ import annotations

import copy

import pytest

from config.params import make_config
from src.data_gen import generate_dataset


@pytest.fixture(scope="session")
def cfg():
    """Configuración del caso base."""
    return make_config()


@pytest.fixture(scope="session")
def bundle(cfg):
    """Dataset del caso base, generado una vez para toda la sesión."""
    return generate_dataset(cfg)


@pytest.fixture
def bundle_mutable(bundle):
    """Copia profunda del dataset, para tests que corrompen los datos a propósito."""
    return copy.deepcopy(bundle)
