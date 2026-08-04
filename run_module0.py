"""Gate del Módulo 0: genera el banco sintético, lo valida y decide si se avanza.

    python run_module0.py

Escribe las tablas y el `ground_truth.json` en `data/`, imprime la tabla de los 11
controles y deja el informe en `data/validation_report.md`. **Sale con código 1 si
algún control de severidad ERROR falla**, que es lo que impide avanzar al Módulo 1.

El código de salida importa: convierte el protocolo de fases en algo que una tubería
de CI puede hacer cumplir, en lugar de una promesa de que alguien miró el informe.
"""

from __future__ import annotations

import sys

from config.params import make_config
from src.data_gen import generate_dataset
from src.validation import assert_no_errors, calibration_metrics, report, run_all_checks


def main() -> int:
    """Genera, valida y reporta. Devuelve el código de salida del proceso."""
    # La consola de Windows arranca en cp1252 y no sabe imprimir «≤» ni «β».
    for flujo in (sys.stdout, sys.stderr):
        try:
            flujo.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):  # pragma: no cover
            pass

    cfg = make_config()

    print(f"Generando banco sintético (semilla {cfg.SEED})…")
    bundle = generate_dataset(cfg)

    # Control #11: segunda corrida con la misma semilla, hash independiente.
    hash_repetido = generate_dataset(make_config()).hash()

    metricas = calibration_metrics(bundle, cfg)
    checks = run_all_checks(bundle, cfg, hash_repetido=hash_repetido)

    bundle.write(cfg.DATA_DIR)
    texto = report(checks, metricas)
    (cfg.DATA_DIR / "validation_report.md").write_text(texto, encoding="utf-8")

    print()
    print(f"{'#':<3} {'Sev':<8} {'Estado':<8} Control")
    print("-" * 78)
    for c in checks:
        estado = "PASA" if c.passed else ("FALLA" if c.severidad == "ERROR" else "avisa")
        print(f"{c.numero:<3} {c.severidad:<8} {estado:<8} {c.nombre}")

    print()
    print("Métricas de calibración:")
    for k, v in metricas.items():
        print(f"  {k:<38} {v:>12,.4f}")

    print()
    print(f"Datos y informe escritos en {cfg.DATA_DIR}")

    try:
        assert_no_errors(checks)
    except AssertionError as exc:
        print()
        print(exc)
        print("\nEl Módulo 0 NO pasa el gate. No se avanza al Módulo 1.")
        return 1

    avisos = [c for c in checks if not c.passed]
    print(f"\nGate superado: 0 ERROR, {len(avisos)} WARNING activo(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
