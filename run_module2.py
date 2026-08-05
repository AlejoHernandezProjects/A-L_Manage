"""Módulo 2: estima el modelo conductual de NMD y publica el informe.

    python run_module2.py

Escribe `data/nmd_report.md`.
"""

from __future__ import annotations

import sys

from config.params import make_config
from src.balance import gap_repreciacion, indicadores_gap
from src.data_gen import generate_dataset
from src.deposits import (
    diagnostico_identificabilidad,
    informe_nmd,
    instrumentos_conductuales,
    modelo_nmd,
)
from src.validation import calibration_metrics


def main() -> int:
    """Estima, reasigna bandas, compara contra el gap contractual y reporta."""
    for flujo in (sys.stdout, sys.stderr):
        try:
            flujo.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):  # pragma: no cover
            pass

    cfg = make_config()
    print(f"Generando balance (semilla {cfg.SEED})…")
    bundle = generate_dataset(cfg)
    delta_nii = calibration_metrics(bundle, cfg)["delta_nii_12m_up200"]

    modelo = modelo_nmd(bundle, cfg)

    print("\nBeta asimétrica — estimada contra el ground truth:")
    print(f"  {'prod':<8}{'β̂+':>8}{'real':>7}{'error':>8}{'β̂-':>9}{'real':>7}{'error':>8}{'R²':>7}")
    for prod, f in modelo["betas"].iterrows():
        print(
            f"  {prod:<8}{f['beta_up_est']:>8.3f}{f['beta_up_real']:>7.2f}{f['error_up']:>+8.3f}"
            f"{f['beta_down_est']:>9.3f}{f['beta_down_real']:>7.2f}{f['error_down']:>+8.3f}"
            f"{f['r2']:>7.2f}"
        )

    print("\nProporción estable (destendenciada):")
    for prod, f in modelo["estables"].iterrows():
        print(
            f"  {prod:<8} mínimo {f['estable_minimo']:.3f}   core_share real "
            f"{f['core_share_real']:.2f}   error {f['error_vs_core_share']:+.3f}"
        )

    print("\n¿Es estimable la vida del núcleo desde el saldo agregado?")
    ident = diagnostico_identificabilidad(cfg)
    for vida, f in ident.iterrows():
        print(
            f"  vida verdadera {vida:>4.1f} a  ->  correlación vs. base "
            f"{f['correlacion_vs_base']:.6f}   dif. máx {f['dif_relativa_max']:.2%}"
        )
    print("  No. Una vida de 3 años y una de 10 dan la misma serie observable.")
    print("  Por eso IRRBB acota el plazo del núcleo en vez de pedir mejor estimación.")

    print("\nNúcleo bajo el marco estandarizado IRRBB:")
    for prod, n in modelo["nucleos"].items():
        print(
            f"  {prod:<8} estable {n['proporcion_estable']:.3f} × (1−β {n['beta_corte']:.3f}) "
            f"= {n['nucleo_bruto']:.3f}  ->  núcleo {n['nucleo']:.3f} "
            f"(tope {n['cap_nucleo']:.2f}{', MUERDE' if n['tope_nucleo_muerde'] else ''})"
        )
        print(
            f"           vida supuesta {n['vida_supuesta_a']:.1f} a  ->  aplicada "
            f"{n['vida_a']:.1f} a (tope {n['cap_vida_a']:.1f} a"
            f"{', MUERDE' if n['tope_vida_muerde'] else ''}); real {n['vida_real_a']:.1f} a, "
            f"error {n['error_vida_a']:+.1f} a"
        )

    inst_cond = instrumentos_conductuales(bundle.instrumentos, modelo, cfg)
    ind_c = indicadores_gap(gap_repreciacion(bundle.instrumentos, cfg), cfg)
    ind_b = indicadores_gap(gap_repreciacion(inst_cond, cfg), cfg)

    print("\nGap acumulado a 12 meses:")
    print(
        f"  Contractual (Módulo 1)  {ind_c['gap_acumulado_12m_musd']:>12,.0f} M  "
        f"{ind_c['gap_12m_sobre_activos']:>+8.1%}   {ind_c['estado_politica']}"
    )
    print(
        f"  Conductual  (Módulo 2)  {ind_b['gap_acumulado_12m_musd']:>12,.0f} M  "
        f"{ind_b['gap_12m_sobre_activos']:>+8.1%}   {ind_b['estado_politica']}"
    )
    print(f"  ΔNII 12m (+200 pb)      {delta_nii:>12.2%}")

    texto = informe_nmd(bundle, cfg, delta_nii=delta_nii)
    destino = cfg.DATA_DIR / "nmd_report.md"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(texto, encoding="utf-8")
    print(f"\nInforme escrito en {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
