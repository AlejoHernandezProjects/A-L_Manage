"""Módulo 4: valora el balance bajo los seis escenarios y publica el informe.

    python run_module4.py

Escribe `data/eve_report.md`.
"""

from __future__ import annotations

import sys

from config.params import make_config
from src.data_gen import generate_dataset
from src.deposits import instrumentos_conductuales, modelo_nmd
from src.eve import (
    cartera_valorable,
    duracion_convexidad,
    informe_eve,
    sensibilidad_eve,
    tabla_aproximaciones,
    tabla_delta_eve,
)
from src.nii import tabla_delta_nii


def main() -> int:
    """Calcula el EVE bajo los seis escenarios y reporta contra el umbral del 15%."""
    for flujo in (sys.stdout, sys.stderr):
        try:
            flujo.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):  # pragma: no cover
            pass

    cfg = make_config()
    print(f"Generando balance (semilla {cfg.SEED})…")
    bundle = generate_dataset(cfg)
    modelo = modelo_nmd(bundle, cfg)
    ceros = bundle.curvas.iloc[-1].to_numpy()

    inst = instrumentos_conductuales(bundle.instrumentos, modelo, cfg)
    print("Calibrando spreads a la par…")
    cartera = cartera_valorable(inst, ceros, cfg)

    tabla = tabla_delta_eve(cartera, ceros, cfg)
    tier1 = cfg.BANK_PROFILE["tier1_musd"]
    peor = tabla["delta_sobre_tier1"].idxmin()

    print(f"\nEVE base {tabla.attrs['eve_base']:,.0f} M   ·   Tier 1 {tier1:,.0f} M   ·   "
          f"umbral −{0.15 * tier1:,.0f} M")
    print(f"\n  {'Escenario':<18}{'EVE':>11}{'ΔEVE':>11}{'/Tier 1':>10}  outlier")
    for nombre, f in tabla.iterrows():
        marca = " <== PEOR" if nombre == peor else ""
        print(
            f"  {f['etiqueta']:<18}{f['eve_musd']:>11,.0f}{f['delta_musd']:>+11,.0f}"
            f"{f['delta_sobre_tier1']:>+10.1%}  {'SÍ' if f['supera_umbral'] else 'no'}{marca}"
        )

    dc = duracion_convexidad(cartera, ceros, cfg)
    print(
        f"\nDuración efectiva: activo {dc['duracion_activos']:.2f} a, "
        f"pasivo {dc['duracion_pasivos']:.2f} a, gap {dc['gap_duracion']:+.2f} a"
    )

    aprox = tabla_aproximaciones(cartera, ceros, cfg)
    print("\nDónde falla la aproximación lineal:")
    print(f"  {'choque':>8}{'completa':>12}{'1er orden':>12}{'2do orden':>12}{'error lineal':>14}")
    for pb, f in aprox.iterrows():
        if pb < 0:
            continue
        print(
            f"  {pb:>+7,}p{f['revaluacion_completa']:>12,.0f}{f['primer_orden']:>12,.0f}"
            f"{f['segundo_orden']:>12,.0f}{f['error_relativo_primer_orden']:>+14.1%}"
        )

    print("\nLa tensión central del proyecto:")
    t_nii = tabla_delta_nii(bundle, cfg, modelo)
    print(f"  ΔNII 12m (+200 pb)        {t_nii.loc['paralelo_arriba', 'delta_pct']:>+8.2%}")
    print(f"  ΔEVE / Tier 1 (+200 pb)   {tabla.loc['paralelo_arriba', 'delta_sobre_tier1']:>+8.1%}")

    sens = sensibilidad_eve(bundle, cfg, modelo)
    print("\nSensibilidad al plazo del núcleo (el supuesto no identificado):")
    for vida, f in sens["vida"].iterrows():
        print(
            f"  vida {vida:>4.1f} a -> peor {f['peor_escenario']:<18}"
            f"{f['peor_sobre_tier1']:>+8.1%}   outlier {'SÍ' if f['supera_umbral'] else 'no'}"
        )

    texto = informe_eve(bundle, cfg, modelo, tabla_nii=t_nii)
    destino = cfg.DATA_DIR / "eve_report.md"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(texto, encoding="utf-8")
    print(f"\nInforme escrito en {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
