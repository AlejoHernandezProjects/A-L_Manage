"""Módulo 3: proyecta el margen financiero y publica el informe.

    python run_module3.py

Escribe `data/nii_report.md`.
"""

from __future__ import annotations

import sys

from config.params import make_config
from src.data_gen import generate_dataset
from src.deposits import modelo_nmd
from src.nii import informe_nii, sensibilidad_nii, tabla_delta_nii
from src.scenarios import ETIQUETAS
from src.validation import calibration_metrics


def main() -> int:
    """Proyecta el NII bajo los seis escenarios y reporta."""
    for flujo in (sys.stdout, sys.stderr):
        try:
            flujo.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):  # pragma: no cover
            pass

    cfg = make_config()
    print(f"Generando balance (semilla {cfg.SEED})…")
    bundle = generate_dataset(cfg)
    modelo = modelo_nmd(bundle, cfg)
    delta_m0 = calibration_metrics(bundle, cfg)["delta_nii_12m_up200"]

    tabla = tabla_delta_nii(bundle, cfg, modelo)
    base = float(tabla["nii_base_musd"].iloc[0])

    print(f"\nNII base a {cfg.NII_PARAMS['horizonte_meses']} meses: {base:,.0f} M")
    print(f"\n  {'Escenario':<18}{'NII':>12}{'Δ':>12}{'Δ%':>10}")
    for _, f in tabla.iterrows():
        titular = "*" if f.name in cfg.ESCENARIOS_NII else " "
        print(
            f" {titular}{f['etiqueta']:<18}{f['nii_escenario_musd']:>12,.0f}"
            f"{f['delta_musd']:>+12,.0f}{f['delta_pct']:>+10.2%}"
        )
    print("  (*) los dos que el NII titula por convención; los seis se calculan igual.")

    sens = sensibilidad_nii(bundle, cfg, modelo)
    print("\nLa asimetría de la beta paga:")
    for idx, f in sens["asimetria"].iterrows():
        print(f"  {idx:<24}" + "   ".join(f"{c} {v:+.2%}" for c, v in f.items()))

    print("\nReconciliación con el Módulo 0:")
    print(f"  Gap estático a 12 meses      {delta_m0:>+8.2%}")
    print(f"  Proyección completa          {tabla.loc['paralelo_arriba', 'delta_pct']:>+8.2%}")

    texto = informe_nii(bundle, cfg, modelo, delta_nii_modulo0=delta_m0)
    destino = cfg.DATA_DIR / "nii_report.md"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(texto, encoding="utf-8")
    print(f"\nInforme escrito en {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
