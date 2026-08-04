"""Parámetros del Simulador ALM / IRRBB.

Regla de consistencia §9.6: **todos** los supuestos viven aquí. Ningún número
mágico dentro de la lógica de `src/`.

Organización del archivo:

    1.  Semilla y convenciones de mercado
    2.  BANK_PROFILE      — mix de balance (§5)
    3.  MARKET            — régimen de tasas y curva (§6.2, §6.3)
    4.  DEPOSIT_RATES     — ajuste parcial asimétrico (§6.4)
    5.  NMD_PARAMS        — saldos core/volátil (§6.5)
    6.  INSTRUMENT_SPECS  — cohortes por producto (§6.6)
    7.  CALIBRATION_TARGETS (§6.1) y VALIDATION_THRESHOLDS (§6.8)
    8.  SENSITIVITY_GRID  — barridos declarados de antemano

Distinción de vocabulario que se usa en todo el archivo: los **objetivos de
calibración** (§7) son criterios de aceptación fijados ANTES de generar y no se
tocan; los **parámetros de calibración** (spreads, plazos, duraciones objetivo)
son las perillas que se mueven para alcanzarlos.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. Semilla y convenciones de mercado
# ---------------------------------------------------------------------------

SEED = 20251231
"""Semilla maestra. Regla §9.7: todo el proyecto es reproducible desde aquí."""

DATES = {
    "inicio": "2016-01-31",
    "n_meses": 120,
    "freq": "ME",  # fin de mes
}
"""120 observaciones mensuales. La última fecha es la **fecha de corte**: el
momento en que se fotografía el balance y desde el que se proyectan NII y EVE."""

CONVENTIONS = {
    "base_dias": "30/360",
    "capitalizacion": "continua",
    "moneda": "USD",
    "dias_habiles": False,  # calendario ideal, sin ajuste de días hábiles
}
"""Convenciones de §6.6.

La capitalización continua es una decisión de ingeniería, no de negocio: hace que
`DF(τ) = exp(−y(τ)·τ)` sea positivo y monótono por construcción mientras `y > 0`,
que es justo el control #6. Una mesa real cotiza en composición simple o semestral
según el mercado; la traducción es mecánica y no cambia ninguna conclusión.
"""

# ---------------------------------------------------------------------------
# 2. Perfil del banco (§5) — decisiones cerradas
# ---------------------------------------------------------------------------

BANK_PROFILE = {
    "activos_totales_musd": 50_000.0,
    "tier1_musd": 4_000.0,
    "nombre": "Banco universal genérico, corporativo + minorista",
}
"""Tier 1 ≈ 4.000 M fija el umbral de alerta supervisora: 15% de 4.000 = **−600 M
de ΔEVE**. Es el número contra el que se compara el peor de los seis escenarios."""

# ---------------------------------------------------------------------------
# 3. Mercado: tasa de referencia y curva (§6.2, §6.3)
# ---------------------------------------------------------------------------

RATE_REGIMES = [
    {
        "nombre": "tasas_bajas",
        "mes_inicio": 1,
        "mes_fin": 30,
        "theta_inicio": 0.0150,
        "theta_fin": 0.0150,
        "kappa": 0.8,
        "sigma": 0.0035,
        "pendiente_inicio": 0.0150,
        "pendiente_fin": 0.0150,
    },
    {
        "nombre": "ciclo_alzas",
        "mes_inicio": 31,
        "mes_fin": 54,
        "theta_inicio": 0.0150,
        "theta_fin": 0.0550,
        "kappa": 2.5,  # un banco central restrictivo se mueve con convicción
        "sigma": 0.0060,
        "pendiente_inicio": 0.0150,
        "pendiente_fin": -0.0080,
    },
    {
        "nombre": "meseta_alta",
        "mes_inicio": 55,
        "mes_fin": 84,
        "theta_inicio": 0.0525,
        "theta_fin": 0.0525,
        "kappa": 1.5,
        "sigma": 0.0040,
        "pendiente_inicio": -0.0080,
        "pendiente_fin": 0.0010,
    },
    {
        "nombre": "normalizacion",
        "mes_inicio": 85,
        "mes_fin": 120,
        "theta_inicio": 0.0525,
        "theta_fin": 0.0300,
        "kappa": 1.2,
        "sigma": 0.0050,
        "pendiente_inicio": 0.0010,
        "pendiente_fin": 0.0100,
    },
]
"""Régimen por tramos de §6.2, con la pendiente de la curva (10A − 3M) atada al
mismo régimen (§6.3).

La pendiente cruza a territorio negativo dentro del `ciclo_alzas` y se mantiene
invertida durante la mayor parte de `meseta_alta` (≈ meses 55–81). Esa inversión
no es decorativa: sin un tramo de curva invertida, el escenario de empinamiento
del Módulo 4 no tendría contenido económico — sería un choque sobre una curva que
ya estaba empinada.
"""

POLICY_RATE = {
    "r0": 0.0150,
    "escalon_pb": 25.0,
    "histeresis_pb": 37.5,
    "movimiento_max_pb": 75.0,
    "piso": 0.0,
}
"""Capa de *decisión de comité* sobre el Vasicek latente.

Un Vasicek puro produce una tasa de política que se mueve 7 pb cada mes. Ningún
banco central hace eso: las tasas se mueven en escalones de 25/50 pb, en fechas de
comité, y se quedan quietas entre medias. La histéresis (no mover hasta que el
latente se aparte 37,5 pb = escalón y medio) reproduce esa persistencia.

Importa para ALM porque la persistencia es lo que convierte un desfase de
repreciación en pérdida o ganancia sostenida: si la tasa oscilara cada mes, el gap
se promediaría a cero y el riesgo de tasa sería un problema menor.
"""

CURVE = {
    "tenores_a": [1 / 12, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0],
    "tenor_ancla_corto_a": 0.25,   # se fuerza y(3M) = ref_rate
    "tenor_ancla_largo_a": 10.0,   # se fuerza y(10A) = ref_rate + pendiente
    "tau_lambda": 2.5,             # loading de Nelson-Siegel; joroba en ~30 meses
    "curvatura_por_regimen": {
        "tasas_bajas": 0.0050,
        "ciclo_alzas": 0.0100,
        "meseta_alta": 0.0000,
        "normalizacion": -0.0050,
    },
    "curvatura_phi": 0.85,
    "curvatura_sigma": 0.0020,
    "ruido_tenor_pb": 5.0,
    "piso_tasa_cero": 0.0005,
}
"""Nelson-Siegel con los dos anclajes resueltos como sistema lineal 2×2 en
(β₀, β₁) dados β₂ y τλ. El anclaje corto hace que el control #5 (curva 3M dentro
de ±25 pb de la referencia) pase **por construcción**, no por calibración a ciegas.

`piso_tasa_cero` evita tasas cero no positivas, que romperían la monotonía de los
factores de descuento del control #6. El piso *post-choque* de los seis escenarios
IRRBB es otro objeto distinto y vive en `src/scenarios.py` (Módulo 4).
"""

# ---------------------------------------------------------------------------
# 4. Tasas pagadas al depósito — ajuste parcial asimétrico (§6.4)
# ---------------------------------------------------------------------------

DEPOSIT_RATES = {
    "vista": {"alpha": 0.0005, "beta_up": 0.25, "beta_down": 0.55, "lambda_": 0.30},
    "ahorro": {"alpha": 0.0015, "beta_up": 0.45, "beta_down": 0.75, "lambda_": 0.35},
    "plazo": {"alpha": 0.0020, "beta_up": 0.85, "beta_down": 0.90, "lambda_": 0.70},
    "sigma_eps": 0.00015,
    "piso": 0.0,
    "kappa_ancla": 0.02,
}
"""Ground truth de §6.4. β⁻ > β⁺ codifica el hecho estilizado central del negocio de
depósitos: **el banco traslada las bajadas rápido y las subidas lento**. No es
picardía comercial, es la respuesta racional de una franquicia con poder de fijación
de precios (Hannan-Berger, Neumark-Sharpe).

**Corrección de especificación respecto a la fórmula literal de §6.4.**

§6.4 escribe  `d_t = d_{t-1} + λ·(α + β^± · r_t − d_{t-1}) + ε_t`,  con β^± aplicado
al **nivel objetivo**. Tomada al pie de la letra, esa fórmula tiene una patología de
signo: con r = 5%, el objetivo en régimen de subida es `α + 0,25·5% = 1,30%` y en
régimen de bajada `α + 0,55·5% = 2,80%`. El primer recorte de tasas hace que el
objetivo **suba** 150 pb. Es decir: la fórmula literal dice que cuando el banco
central baja, el banco le sube la tasa al depositante. Al generar con ella, la beta
estimada de vista sale en −0,15.

La especificación correcta aplica la asimetría al **traspaso del cambio**, con un
ancla de largo plazo simétrica que impide que el efecto trinquete lleve la tasa a
cero tras un ciclo completo:

    d*_t = d*_{t-1} + β^± · Δr_t                 traspaso asimétrico de corto plazo
    d*_t = d*_t + κ · (α + β̄·r_t − d*_t)         reversión lenta al nivel competitivo
    d_t  = d_{t-1} + λ · (d*_t − d_{t-1}) + ε_t   ajuste parcial

con β̄ = (β⁺ + β⁻)/2. Es la forma en que la industria usa la palabra "beta": *de una
subida de 100 pb trasladamos 25*. Y es lo que hace que el control #7 (regresión de
Δd sobre Δr) tenga sentido — si el proceso no estuviera dirigido por Δr, esa
regresión no podría recuperar nada.

`kappa_ancla` = 0,05 mensual: el trinquete es real pero acotado. Sin ancla, tras un
ciclo completo de +400/−225 pb la tasa de vista terminaría en 19 pb, y aunque eso no
es absurdo para una cuenta transaccional, sí lo es para una cuenta de ahorro. Con el
ancla, la competencia acaba imponiéndose — despacio, que es como ocurre.

Consecuencia práctica: con β⁺(vista) = 0,25 el banco retiene el 75% de una subida de
tasas en el margen. Eso es la mitad de por qué el NII mejora ante +200 pb.

Una beta estimada con OLS simple sobre estos datos está mal especificada: devuelve un
promedio ponderado por la frecuencia de subidas y bajadas de la muestra, no un
parámetro estructural. El Módulo 2 debe demostrarlo contra estos números.
"""

# ---------------------------------------------------------------------------
# 5. Saldos de depósitos a la vista y ahorro (§6.5)
# ---------------------------------------------------------------------------

NMD_PARAMS = {
    "vista": {
        "vida_promedio_a": 4.2,
        "core_share": 0.85,
        "crecimiento_anual": 0.045,
        "sigma_volatil": 0.012,
        "gamma_migracion": 2.0,
    },
    "ahorro": {
        "vida_promedio_a": 3.0,
        "core_share": 0.90,
        "crecimiento_anual": 0.045,
        "sigma_volatil": 0.010,
        "gamma_migracion": 3.0,
    },
    "ar1_phi": 0.60,
    "phi_migracion": 0.70,
    "sigma_originacion": 0.05,
    "estacionalidad": {
        1: -0.020,   # enero: resaca de diciembre
        2: -0.015,
        3: 0.000,
        4: -0.025,   # abril: salidas fiscales
        5: 0.000,
        6: 0.005,
        7: 0.005,
        8: 0.000,
        9: 0.000,
        10: 0.000,
        11: 0.005,
        12: 0.035,   # diciembre: aguinaldos y cierre de ejercicio
    },
    "spread_normal": 0.0150,
    "estres": {
        "mes": 96,
        "caidas": [-0.05, -0.04],
        "meses_recuperacion": 6,
        "fraccion_recuperada": 2 / 3,
        "erosion_permanente_core": 0.02,
    },
}
"""Descomposición estructural  B_t = C_t (core) + V_t (volátil).

**`vida_promedio_a` es vida promedio (mean life), NO half-life.** Con decaimiento
exponencial las dos difieren en un factor 1/ln2 ≈ 1,443. La distinción no es
pedante: IRRBB capea la **vida promedio de repreciación** del core en 5 años para
minorista transaccional, 4,5 para minorista no transaccional y 4 para mayorista.
Si 4,2 años fuese half-life, la vida promedio sería 6,06 años y el core verdadero
excedería el tope regulatorio. `build_ground_truth` reporta ambas cifras para que
el Módulo 2 no las confunda.

El core se genera como **stock con runoff y originación nueva**:

    C_t = C_{t-1} · exp(−λ_d/12) + N_t,    λ_d = 1 / vida_promedio_a

y no como una simple tendencia creciente. La razón es de identificación: si el core
sólo fuera una tendencia, la vida promedio no estaría presente en los datos y el
Módulo 2 estaría "recuperando" un parámetro que nunca se usó para generar. Con
runoff más originación el 4,2 años es un parámetro estructural real, y el error de
estimación del Módulo 2 mide algo.

`gamma_migracion`: cuando `r − d` se abre por encima de `spread_normal`, sale saldo
de vista y ahorro **y entra a plazo** — el fondeo se traslada, no se evapora. En la
mesa esto se llama *deposit mix shift*, y es media razón por la que el costo de
fondeo sube en un ciclo de alzas aunque las betas por producto no se muevan. Ahorro
migra más que vista: el saldo transaccional está ahí por servicio, no por precio.
"""

# ---------------------------------------------------------------------------
# 6. Instrumentos: cohortes por producto (§6.6)
# ---------------------------------------------------------------------------

N_COHORTES_OBJETIVO = 3000
"""Cada instrumento es una **cohorte (vintage)** de producto, no un contrato. Lo que
importa es que las fechas de originación estén repartidas en los 120 meses: si todo
se originara el mismo día, el balance entero repactaría en una sola banda temporal y
tanto el gap como el EVE serían artefactos del generador."""

INSTRUMENT_SPECS = {
    # ------------------------------- ACTIVOS -------------------------------
    "efectivo": {
        "lado": "activo",
        "share": 0.040,
        "n_cohortes": 10,
        "tipo_tasa": "mixta",
        "frac_fija": 0.0,     # 100% variable: repacta overnight
        "plazo_meses": (1, 2),
        "repricing_meses": 1,
        "amortizacion": "bullet",
        "frecuencia_pago_meses": 1,
        "spread_pb": -25.0,   # encaje remunerado por debajo de la referencia
    },
    "inversiones": {
        "lado": "activo",
        "share": 0.180,
        "n_cohortes": 350,
        "tipo_tasa": "mixta",
        "frac_fija": 0.80,
        "plazo_meses": (45, 156),
        "repricing_meses": 3,
        "amortizacion": "bullet",
        "frecuencia_pago_meses": 6,
        "spread_pb": 35.0,
        "duracion_objetivo_a": 3.5,
    },
    "comercial": {
        "lado": "activo",
        "share": 0.324,
        "n_cohortes": 800,
        "tipo_tasa": "mixta",
        "frac_fija": 0.15,
        "plazo_meses": (36, 84),
        "repricing_meses": 3,
        "amortizacion": "bullet",
        "frecuencia_pago_meses": 1,
        "spread_pb": 120.0,
    },
    "hipotecario": {
        "lado": "activo",
        "share": 0.180,
        "n_cohortes": 600,
        "tipo_tasa": "mixta",
        "frac_fija": 0.90,
        "plazo_meses": (180, 240),
        "repricing_meses": 12,
        "amortizacion": "frances",
        "frecuencia_pago_meses": 1,
        "spread_pb": 100.0,
    },
    "consumo": {
        "lado": "activo",
        "share": 0.086,
        "n_cohortes": 350,
        "tipo_tasa": "mixta",
        "frac_fija": 1.0,
        "plazo_meses": (36, 60),
        "repricing_meses": None,
        "amortizacion": "frances",
        "frecuencia_pago_meses": 1,
        "spread_pb": 650.0,
    },
    "vehiculos": {
        "lado": "activo",
        "share": 0.058,
        "n_cohortes": 250,
        "tipo_tasa": "mixta",
        "frac_fija": 1.0,
        "plazo_meses": (48, 60),
        "repricing_meses": None,
        "amortizacion": "frances",
        "frecuencia_pago_meses": 1,
        "spread_pb": 350.0,
    },
    "tarjetas": {
        "lado": "activo",
        "share": 0.072,
        "n_cohortes": 120,
        "tipo_tasa": "administrada",
        "frac_fija": 0.0,
        "plazo_meses": (3, 6),
        "repricing_meses": 4,
        "amortizacion": "bullet",
        "frecuencia_pago_meses": 1,
        "spread_pb": 1400.0,
        "revolving_ratio": 0.35,
    },
    "otros_activos": {
        "lado": "activo",
        "share": 0.060,
        "n_cohortes": 10,
        "tipo_tasa": "no_sensible",
        "frac_fija": 0.0,
        "plazo_meses": (1, 1),
        "repricing_meses": None,
        "amortizacion": "bullet",
        "frecuencia_pago_meses": 1,
        "spread_pb": 0.0,
    },
    # ------------------------------- PASIVOS -------------------------------
    "vista": {
        "lado": "pasivo",
        "share": 0.220,
        "n_cohortes": 1,
        "tipo_tasa": "administrada",
        "frac_fija": 0.0,
        "plazo_meses": (1, 1),
        "repricing_meses": 1,
        "amortizacion": "bullet",
        "frecuencia_pago_meses": 1,
        "spread_pb": None,   # la tasa viene de DEPOSIT_RATES, no de la curva
        "es_nmd": True,
    },
    "ahorro": {
        "lado": "pasivo",
        "share": 0.220,
        "n_cohortes": 1,
        "tipo_tasa": "administrada",
        "frac_fija": 0.0,
        "plazo_meses": (1, 1),
        "repricing_meses": 1,
        "amortizacion": "bullet",
        "frecuencia_pago_meses": 1,
        "spread_pb": None,
        "es_nmd": True,
    },
    "plazo": {
        "lado": "pasivo",
        "share": 0.260,
        "n_cohortes": 400,
        "tipo_tasa": "mixta",
        "frac_fija": 1.0,
        "plazo_meses": (3, 24),
        "repricing_meses": None,
        "amortizacion": "bullet",
        "frecuencia_pago_meses": 1,
        "spread_pb": None,
    },
    "interbancario": {
        "lado": "pasivo",
        "share": 0.150,
        "n_cohortes": 120,
        "tipo_tasa": "mixta",
        "frac_fija": 0.45,
        "plazo_meses": (1, 60),
        "repricing_meses": 3,
        "amortizacion": "bullet",
        "frecuencia_pago_meses": 3,
        "spread_pb": 40.0,
    },
    "otros_pasivos": {
        "lado": "pasivo",
        "share": 0.060,
        "n_cohortes": 10,
        "tipo_tasa": "no_sensible",
        "frac_fija": 0.0,
        "plazo_meses": (1, 1),
        "repricing_meses": None,
        "amortizacion": "bullet",
        "frecuencia_pago_meses": 1,
        "spread_pb": 0.0,
    },
    # ----------------------------- PATRIMONIO ------------------------------
    "patrimonio": {
        "lado": "patrimonio",
        "share": 0.090,
        "n_cohortes": 1,
        "tipo_tasa": "no_sensible",
        "frac_fija": 0.0,
        "plazo_meses": (1, 1),
        "repricing_meses": None,
        "amortizacion": "bullet",
        "frecuencia_pago_meses": 1,
        "spread_pb": 0.0,
    },
}
"""Un `share` por categoría, exactamente el mix de §5. Los saldos generados se
reescalan al share objetivo, de modo que Activos = Pasivos + Patrimonio (control #1)
se cumple por construcción y no por suerte del muestreo.

`spread_pb` es un **parámetro de calibración**: se mueve para alcanzar los objetivos
de §7, no para producir un resultado bonito.

Nota de práctica sobre `tarjetas`. La tasa nominal de tarjeta (referencia + 1800 pb
≈ 21%) no se devenga sobre todo el saldo: una parte grande de la cartera es
transaccional y se paga dentro del período de gracia. `revolving_ratio` = 0,35 es la
fracción que efectivamente devenga. Aplicar el 21% pleno a los 3.600 M infla el NIM
del banco a ~4,7% y hace inalcanzable el objetivo de 2,8–3,6%. Es un error frecuente
al construir balances sintéticos y vale la pena dejarlo escrito.

Nota sobre `efectivo`. §5 lo etiqueta "no sensible / overnight", que son dos cosas
distintas. Se modela como **variable con repreciación mensual**: el encaje y las
disponibilidades en el banco central están remunerados y repactan de inmediato, así
que son de los activos más sensibles del balance, no de los menos. Lo que no son es
*fuente de margen* — rinden por debajo de la referencia (spread negativo), y por eso
un banco con exceso de liquidez ve caer su NIM. Tratarlos como tasa cero regalaría
~60 M de ingreso anual y sesgaría el NII a la baja.

`vista` y `ahorro` llevan una sola cohorte porque su saldo sale de la serie generada
en `generate_deposit_balances`, y su repreciación real no la fija el contrato sino el
modelo de comportamiento del Módulo 2. `repricing_meses = 1` es el tratamiento
contractual ingenuo (todo repacta mañana): el Módulo 2 lo sustituye por el perfil de
runoff del core, y la diferencia entre ambos tratamientos es el resultado central del
proyecto.
"""

# ---------------------------------------------------------------------------
# 7. Objetivos de calibración (§6.1) — criterios de aceptación
# ---------------------------------------------------------------------------

CALIBRATION_TARGETS = {
    "duracion_activos_sensibles_a": (2.2, 2.5),
    "duracion_pasivos_sensibles_a": (1.8, 2.1),
    "gap_duracion_a": (0.4, 0.6),
    "delta_eve_peor_sobre_tier1": (-0.13, -0.09),
    "delta_nii_12m_up200": (0.015, 0.035),
    "nim": (0.028, 0.036),
}
"""Fijados ANTES de generar, para no acomodar parámetros al resultado deseado.
Si un objetivo resulta inalcanzable con parámetros defendibles, se reporta como tal
en el README — no se mueve la meta.

Convención de medición de las duraciones (importa, y no es obvia): se calculan sobre
los **flujos contractuales y de runoff, sin ajustar por beta**. La duración
"efectiva" de un depósito, que multiplica por (1 − β) para reconocer que el banco no
traslada el choque completo, es un número bastante menor y se reporta aparte. Ambas
son legítimas; mezclarlas es lo que no se puede hacer. Los rangos de arriba
corresponden a la primera convención.

En cualquier caso la duración aquí es un **diagnóstico**: el ΔEVE del Módulo 4 se
calcula por revaluación completa, no por aproximación de duración. La comparación
entre ambos métodos es precisamente uno de los entregables del Módulo 4.
"""

# ---------------------------------------------------------------------------
# 8. Umbrales de validación (§6.8)
# ---------------------------------------------------------------------------

VALIDATION_THRESHOLDS = {
    "balance_tol_relativa": 1e-8,          # #1
    "n_meses_esperado": 120,               # #2
    "spread_minimo_pb": 0.0,               # #4
    "curva_3m_vs_ref_pb": 25.0,            # #5
    "regresion_t_stat_min": 3.0,           # #7
    "regresion_r2_min": 0.50,              # #7
    "regresion_rezagos": 6,                # #7
    "beta_ols_tolerancia": 0.15,           # #8  (WARNING deliberado)
    "salto_saldo_max": 0.15,               # #9
    "meses_exentos_salto": (95, 96, 97, 98),  # ventana del episodio de estrés
}

SEVERIDAD = {"ERROR": "ERROR", "WARNING": "WARNING"}

# ---------------------------------------------------------------------------
# 9. Barridos de sensibilidad (§10) — declarados de antemano
# ---------------------------------------------------------------------------

SENSITIVITY_GRID = {
    "share_hipotecario": [0.180, 0.210, 0.250],
    "beta_up_vista": [0.25, 0.35, 0.45],
    "core_share_vista": [0.65, 0.75, 0.85],
    "vida_promedio_vista_a": [3.0, 4.2, 5.0],
}
"""Se declaran aquí, no en el Módulo 4, para dejar constancia de que el barrido se
diseñó antes de ver los resultados.

`share_hipotecario` recoge el punto §11.1: el caso base es 18%, y el 25% se presenta
como sensibilidad que responde "¿con qué mix de balance este banco se vuelve outlier
del test del 15%?". Mover el caso base hasta que el número saliera dramático sería
exactamente lo que la pre-registración de §7 existe para impedir.

`beta_up_vista` recoge §11.2. `vida_promedio_vista_a` llega hasta 5,0 porque ahí está
el tope IRRBB para minorista transaccional: el barrido muestra cuánto EVE depende de
un supuesto conductual que el regulador ya decidió limitar.
"""

# ---------------------------------------------------------------------------
# 10. Rutas
# ---------------------------------------------------------------------------

from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.json"


# ---------------------------------------------------------------------------
# 11. Configuración con overrides (para los barridos de §9)
# ---------------------------------------------------------------------------

import copy  # noqa: E402
from types import SimpleNamespace  # noqa: E402

_EXPORTABLES = (
    "SEED", "DATES", "CONVENTIONS", "BANK_PROFILE", "RATE_REGIMES", "POLICY_RATE",
    "CURVE", "DEPOSIT_RATES", "NMD_PARAMS", "N_COHORTES_OBJETIVO", "INSTRUMENT_SPECS",
    "CALIBRATION_TARGETS", "VALIDATION_THRESHOLDS", "SENSITIVITY_GRID",
    "ROOT", "DATA_DIR", "GROUND_TRUTH_PATH",
)


def make_config(**overrides) -> SimpleNamespace:
    """Copia profunda de los parámetros, con overrides por ruta punteada.

    El caso base se obtiene con ``make_config()``. Un barrido de sensibilidad se
    obtiene tocando una sola perilla y dejando todo lo demás idéntico::

        cfg = make_config(**{"NMD_PARAMS.vista.core_share": 0.65})
        cfg = make_config(**{"INSTRUMENT_SPECS.hipotecario.share": 0.25})

    La copia profunda es deliberada: un barrido que mutara el módulo global
    contaminaría las corridas siguientes y rompería la reproducibilidad de §9.7 de
    la peor manera posible — de forma silenciosa y dependiente del orden.

    Args:
        **overrides: Pares ``"RUTA.PUNTEADA"=valor``. Cada segmento debe existir.

    Returns:
        ``SimpleNamespace`` con los parámetros, listo para pasar como ``cfg``.

    Raises:
        KeyError: Si una ruta no existe en la configuración base.
    """
    cfg = SimpleNamespace(**{k: copy.deepcopy(globals()[k]) for k in _EXPORTABLES})
    for ruta, valor in overrides.items():
        partes = ruta.split(".")
        nodo = getattr(cfg, partes[0])
        for p in partes[1:-1]:
            nodo = nodo[p]
        if len(partes) == 1:
            setattr(cfg, partes[0], valor)
        else:
            if partes[-1] not in nodo:
                raise KeyError(f"Ruta de override inexistente: {ruta}")
            nodo[partes[-1]] = valor
    return cfg
