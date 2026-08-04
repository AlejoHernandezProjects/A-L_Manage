"""Módulo 1: construye la brecha de repreciación y escribe el informe al comité.

    python run_module1.py

Regenera el dataset desde la semilla en vez de leer los CSV de `data/`: la fuente de
verdad del proyecto es `config/params.py` más `SEED`, y depender de un archivo
intermedio abriría la puerta a informar sobre un balance que ya no corresponde a la
configuración vigente.

Escribe `data/gap_report.md`.
"""

from __future__ import annotations

import sys

from config.params import make_config
from src.balance import (
    gap_repreciacion,
    indicadores_gap,
    informe_alco,
    reconciliacion,
)
from src.data_gen import generate_dataset
from src.validation import calibration_metrics


def main() -> int:
    """Genera el balance, construye las tablas de gap y publica el informe."""
    for flujo in (sys.stdout, sys.stderr):
        try:
            flujo.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):  # pragma: no cover
            pass

    cfg = make_config()
    print(f"Generando balance (semilla {cfg.SEED})…")
    bundle = generate_dataset(cfg)
    inst = bundle.instrumentos

    delta_nii = calibration_metrics(bundle, cfg)["delta_nii_12m_up200"]

    tabla_alco = gap_repreciacion(inst, cfg, agrupar=True)
    tabla_19 = gap_repreciacion(inst, cfg, agrupar=False)
    ind = indicadores_gap(tabla_19, cfg)
    rec = reconciliacion(inst, cfg)

    print("\nReconciliación con el balance (USD M):")
    for concepto, fila in rec.iterrows():
        print(f"  {concepto:<34} {fila['musd']:>12,.0f}")

    print("\nBrecha de repreciación por banda (USD M):")
    print(f"  {'Banda':<8}{'Activos':>11}{'Pasivos':>11}{'Gap':>11}{'Gap acum.':>12}{'% activos':>11}")
    for banda, fila in tabla_alco.iterrows():
        print(
            f"  {banda:<8}{fila['activos']:>11,.0f}{fila['pasivos']:>11,.0f}"
            f"{fila['gap']:>11,.0f}{fila['gap_acumulado']:>12,.0f}"
            f"{fila['gap_acum_pct_activos']:>10.1%}"
        )

    print("\nIndicadores contra política de ALM:")
    print(f"  Gap acumulado a 12 meses           {ind['gap_acumulado_12m_musd']:>12,.0f} M")
    print(f"  Gap 12m / activos totales          {ind['gap_12m_sobre_activos']:>12.1%}")
    print(f"  RSA/RSL a 12 meses                 {ind['rsa_rsl_12m']:>12.2f}")
    print(f"  Estado                             {ind['estado_politica']:>12}")

    texto = informe_alco(inst, cfg, delta_nii_modulo0=delta_nii)
    destino = cfg.DATA_DIR / "gap_report.md"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(texto, encoding="utf-8")

    print(f"\nΔNII 12m (+200 pb) del Módulo 0:      {delta_nii:>12.2%}")
    print(
        "  El gap dice pasivo-sensible y el margen mejora. Los dos números son correctos;\n"
        "  el mal especificado es el gap — ver sección 6 del informe."
    )
    print(f"\nInforme escrito en {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
