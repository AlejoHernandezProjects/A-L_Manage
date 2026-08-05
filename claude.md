# Simulador ALM / IRRBB — Contexto de proyecto

## 1. Qué es esto

Simulador de Asset and Liability Management para un banco comercial genérico,
como proyecto de portafolio de ciencia de datos. Implementa el núcleo cuantitativo
del marco de Basilea sobre riesgo de tasa de interés en el libro bancario
(IRRBB, Comité de Basilea, 2016): las métricas NII y EVE bajo los seis escenarios
de choque prescritos por el estándar.

Perfil del autor: matemáticas aplicadas + ingeniería en sistemas. Objetivo doble:
(1) demostrar comprensión del problema de gestión de activos y pasivos a nivel de
tesorería, y (2) aprender cómo se maneja en la práctica profesional real.

Lenguaje: Python. Moneda única estable. Sin coyuntura de ningún país específico.

## 2. Cómo quiero que trabajes conmigo

Actúa como **mentor de mesa de ALM**, no solo como generador de código:

- Explica cada decisión como la tomaría un equipo de ALM / riesgo de mercado real.
- Señala explícitamente dónde la práctica profesional difiere de la teoría de libro.
- Usa y explica las convenciones de industria: nomenclatura, indicadores estándar,
  umbrales regulatorios, buenas prácticas de reporting al ALCO.
- Por cada paso: qué se hace y **por qué**.

**Protocolo de fases (inviolable).** Vamos en orden 0 → 1 → 2 → 3 → 4 → 5.
Para cada fase: (1) propón el diseño — funciones, firmas, supuestos, parámetros
de calibración — y explica las decisiones; (2) **espera mi visto bueno**;
(3) implementa con docstrings y tests. No avances a la siguiente fase sin que yo
confirme. En el Módulo 0, además, los datos deben pasar sus validaciones antes de
avanzar.

Usa **plan mode** para proponer diseño antes de tocar archivos.

## 3. Estado actual

- [x] Módulo 0 — Diseño propuesto y perfil del banco aprobado (ver §5 y §6)
- [x] Módulo 0 — Implementación y validaciones: **gate superado** (0 ERROR, 1 WARNING).
      Ver §12 para las decisiones tomadas durante la implementación.
- [x] Módulo 1 — Balance sintético y brecha de repreciación. Ver §13.
- [x] Módulo 2 — Modelo de depósitos a la vista (NMD). Ver §14.
- [x] Módulo 3 — Margen financiero (NII) + `src/scenarios.py`. Ver §15.
- [x] Módulo 4 — Valor económico del patrimonio (EVE). Ver §16.
- [ ] Módulo 5 — Alcance vs. estándar IRRBB **← AQUÍ ESTAMOS**

## 4. Estructura del repositorio

```
config/params.py       BANK_PROFILE, MARKET, NMD_PARAMS, CALIBRATION_TARGETS,
                       VALIDATION_THRESHOLDS, SEED, make_config(**overrides)
src/curves.py          Nelson-Siegel, interpolación, descuento, duración,
                       calendario de amortización de principal  [añadido]
src/data_gen.py        Módulo 0 — generación de datos
src/validation.py      Módulo 0 — controles de calidad
run_module0.py         Gate ejecutable: genera → valida → exit≠0 si hay ERROR  [añadido]
run_module1.py         Informe de brecha de repreciación → data/gap_report.md  [añadido]
run_module2.py         Informe del modelo conductual de NMD → data/nmd_report.md  [añadido]
run_module3.py         Informe de margen financiero → data/nii_report.md  [añadido]
run_module4.py         Informe de valor económico → data/eve_report.md  [añadido]
src/balance.py         Módulo 1
src/deposits.py        Módulo 2
src/nii.py             Módulo 3
src/eve.py             Módulo 4
src/scenarios.py       Los seis escenarios IRRBB (definidos UNA sola vez)
data/ground_truth.json Parámetros verdaderos, escrito por el generador
tests/
notebooks/             Solo exploración, nunca lógica de producción
README.md              Problema, supuestos, resultados, alcance vs. IRRBB
```

## 5. Perfil del banco (decisiones ya cerradas)

**Banco universal grande, ~USD 50.000 M en activos, corporativo + minorista.**
**Posicionamiento: cuasi calzado con descalce moderado.**
**Series históricas: 120 meses, con un episodio de estrés de liquidez.**

### Tensión pedagógica objetivo (deliberada, no accidental)

El banco debe calibrarse de modo que ante **+200 pb paralelo**:
- El **NII mejore** (cartera comercial variable repacta rápido; los depósitos
  trasladan solo una fracción vía beta).
- El **EVE se deteriore** (cartera hipotecaria fija a 15–20 años, duración alta).

Ese signo opuesto es la razón por la que Basilea exige ambas perspectivas y es el
resultado central que el proyecto debe demostrar con números propios.

### Activos: USD 50.000 M

| Rubro | % | USD M | Estructura de tasa | Plazo / repricing |
|---|---|---|---|---|
| Efectivo y encaje | 4,0 | 2.000 | No sensible / overnight | Inmediato |
| Portafolio de inversiones | 18,0 | 9.000 | 80% fija, 20% variable | Duración objetivo 3,5 a |
| Crédito comercial | 32,4 | 16.200 | 85% variable (ref + spread) | Repacta trimestral, vence 3–7 a |
| Crédito hipotecario | 18,0 | 9.000 | 90% fija | 15–20 a |
| Consumo | 8,6 | 4.320 | Fija | 3–5 a |
| Vehículos | 5,8 | 2.880 | Fija | 4–5 a |
| Tarjetas de crédito | 7,2 | 3.600 | Tasa administrada | Repricing conductual ~3–6 m |
| Otros activos no sensibles | 6,0 | 3.000 | — | — |

### Pasivos y patrimonio: USD 50.000 M

| Rubro | % | USD M | Comentario |
|---|---|---|---|
| Depósitos a la vista (NMD transaccional) | 22,0 | 11.000 | Beta baja, core alto |
| Cuentas de ahorro (NMD no transaccional) | 22,0 | 11.000 | Beta media |
| Depósitos a plazo | 26,0 | 13.000 | Beta alta, plazo contractual 3–24 m |
| Interbancario y deuda emitida | 15,0 | 7.500 | Repacta rápido; colchón de calce |
| Otros pasivos no sensibles | 6,0 | 3.000 | — |
| Patrimonio | 9,0 | 4.500 | Capital nivel 1 ≈ 4.000 |

**Justificación.** Loan-to-deposit = 36.000/35.000 ≈ 103%, dentro del rango
90–110% típico de un universal financiado con depósitos minoristas. Los NMD
(vista + ahorro) son 44% del fondeo: casi la mitad del pasivo no tiene vencimiento
contractual, por lo que el EVE depende de un supuesto de comportamiento y no de un
contrato. Capital nivel 1 ≈ 4.000 M fija el **umbral de alerta supervisora del 15%
en −600 M de ΔEVE**.

## 6. Módulo 0 — Diseño aprobado

### 6.1 Objetivos de calibración (criterios de aceptación)

Definidos ANTES de generar, para no acomodar parámetros al resultado deseado.

| Métrica | Objetivo |
|---|---|
| Duración modificada de activos sensibles | 2,2 – 2,5 a |
| Duración modificada de pasivos (post-NMD) | 1,8 – 2,1 a |
| Gap de duración | +0,4 a +0,6 a |
| ΔEVE peor escenario / Tier 1 | −9% a −13% |
| ΔNII 12m (+200 pb) | +1,5% a +3,5% |
| NIM | 2,8% – 3,6% |

### 6.2 Tasa de referencia (120 meses)

Vasicek discreto con régimen por tramos y movimientos en escalones de 25/50 pb
(las tasas de política se mueven en decisiones de comité, no en browniano continuo).

| Meses | Régimen | Nivel |
|---|---|---|
| 1–30 | Tasas bajas estables | ~1,5% |
| 31–54 | Ciclo de alzas | 1,5% → 5,5% |
| 55–84 | Meseta alta | ~5,25% |
| 85–120 | Normalización a la baja | 5,25% → 3,0% |

### 6.3 Curva de rendimiento

Nelson-Siegel con nivel y pendiente atados al régimen. Empinada en tasas bajas,
**invertida en el pico del ciclo (meses 60–80)**, para que el escenario de
empinamiento del Módulo 4 tenga contenido económico real.

Debe ser la MISMA curva que descuenta en el Módulo 4 y que alimenta las tasas
variables del Módulo 1.

### 6.4 Tasas pagadas al depósito — ajuste parcial asimétrico

    d_t = d_{t-1} + λ · (α + β^± · r_t − d_{t-1}) + ε_t

con β⁺ en subidas de r y β⁻ en bajadas.

**Ground truth (registrar en `data/ground_truth.json`):**

| Producto | β⁺ (sube) | β⁻ (baja) | λ |
|---|---|---|---|
| Vista | 0,25 | 0,55 | 0,30 |
| Ahorro | 0,45 | 0,75 | 0,35 |
| Plazo | 0,85 | 0,90 | 0,70 |

β⁻ > β⁺ codifica el hecho estilizado central: el banco traslada las bajadas rápido
y las subidas lento. Es también por qué una beta estimada con OLS simple está mal
especificada — el Módulo 2 debe demostrarlo empíricamente contra este ground truth.

### 6.5 Saldos de depósitos a la vista

Generación estructuralmente descompuesta: `B_t = C_t (core) + V_t (volátil)`.

- **C_t**: tendencia log-lineal (+4,5% anual) con decaimiento exponencial lento.
  Vida media verdadera: 4,2 años (vista), 3,0 años (ahorro).
- **V_t**: estacionalidad (pico diciembre, valle enero–febrero, salidas fiscales
  en abril), AR(1) de media cero, y sensibilidad al diferencial de tasas
  (migración de saldos hacia plazo cuando r sube y d no lo sigue).
- **Episodio de estrés en el mes 96**: caída de 9% en dos meses, recuperación de
  dos tercios en seis meses, erosión permanente de ~2% de C_t.

**Nota conceptual para el README:** el core estadístico y el core regulatorio no
son el mismo objeto. IRRBB define core como la porción que no *repactaría* ante un
choque; la descomposición de series identifica la porción que no *se va* en volumen.
Un depósito puede ser estable en saldo y aun así repactar. El error entre el core
verdadero y el estimado es el resultado del experimento, no un defecto.

### 6.6 Instrumentos

~3.000 instrumentos, cada uno una **cohorte (vintage)** de producto, no un contrato
individual. Fechas de originación distribuidas a lo largo de los 120 meses para que
repactaciones y vencimientos queden escalonados (si todo se origina el mismo día,
el balance entero repacta en una banda y las métricas son artefactos del generador).

Convenciones: base 30/360; pagos mensuales en cartera, semestrales en inversiones;
sin convención de días hábiles (calendario ideal); moneda única.

### 6.7 Firmas propuestas

```python
generate_reference_rate(cfg, rng) -> pd.Series
generate_yield_curves(ref_rate, cfg, rng) -> pd.DataFrame   # fechas × tenores
generate_deposit_rates(ref_rate, cfg, rng) -> pd.DataFrame
generate_deposit_balances(ref_rate, dep_rates, cfg, rng) -> pd.DataFrame
generate_instruments(cfg, curve_cutoff, rng) -> pd.DataFrame
build_ground_truth(cfg) -> dict
generate_dataset(cfg, seed) -> DatasetBundle                # orquestador
```

### 6.8 Validaciones automáticas (gates)

Cada control devuelve `Check(nombre, valor, umbral, severidad)`.
**ERROR bloquea el avance al Módulo 1; WARNING solo informa.**

| # | Control | Severidad |
|---|---|---|
| 1 | Activos = Pasivos + Patrimonio (tol. 1e-8 relativa) | ERROR |
| 2 | 120 observaciones mensuales consecutivas, sin NaN ni duplicados | ERROR |
| 3 | `fecha_repreciacion` ≤ `fecha_vencimiento`; tasa fija ⇒ repricing = vencimiento | ERROR |
| 4 | Tasa activa por producto > tasa de fondeo asignada (spread > 0) | ERROR |
| 5 | Curva a 3M dentro de ±25 pb de `ref_rate` en cada fecha | ERROR |
| 6 | Factores de descuento positivos y decrecientes en el plazo | ERROR |
| 7 | Regresión Δd ~ Δr: t-stat > 3 y R² > 0,5 | ERROR |
| 8 | Beta OLS ingenua dentro de ±0,15 del promedio ponderado de β⁺/β⁻ | WARNING |
| 9 | Saldos > 0; sin saltos > 15% fuera del mes 96 | WARNING |
| 10 | Objetivos de calibración de §6.1 dentro de rango | WARNING |
| 11 | Reproducibilidad: dos corridas con misma semilla → hash idéntico | ERROR |

El control 8 es WARNING deliberadamente: **esperamos que falle un poco**, porque una
regresión simétrica sobre un proceso asimétrico está mal especificada. Que dispare
la alerta es información, no un defecto.

## 7. Alcance de los módulos siguientes

**Módulo 1 — Balance sintético.** DataFrame de instrumentos con al menos: id,
categoria, lado, saldo, tasa, tipo_tasa, fecha_repreciacion, fecha_vencimiento,
spread_sobre_referencia, frecuencia_pago, moneda. Informe de brecha de repreciación
clasificando por **cuándo repacta**, no cuándo vence, usando el esquema de bandas
temporales de IRRBB (agrupables si simplificamos).

**Módulo 2 — NMD.** Separar core/volátil, estimar vida media efectiva y beta
asimétrica, **validar contra el ground truth** reportando el error, y reasignar
depósitos a bandas según el modelo y no según el contrato. Comentar los topes de
plazo promedio que IRRBB impone por categoría de cliente: 5 años minorista
transaccional, 4,5 minorista no transaccional, 4 mayorista. Este es el corazón
conceptual del proyecto.

**Módulo 3 — NII.** Proyección a 12 meses. Los instrumentos que repactan toman la
tasa nueva (activos con beta ≈ 1, depósitos con la beta del Módulo 2); los que
vencen se reinvierten bajo regla de balance explícita (constante o crecimiento
parametrizable). Reportar ΔNII en % vs. base.

**Módulo 4 — EVE.** EVE = VP(activos) − VP(pasivos) descontando a la curva del
Módulo 0. Medir ΔEVE / capital nivel 1 y comentar el umbral de alerta supervisora
del 15% (outlier test). Introducir duración y convexidad: usar la duración del gap
como aproximación de primer orden y compararla contra la revaluación completa para
mostrar dónde falla la aproximación lineal ante choques grandes.

**Módulo 5 — Alcance vs. IRRBB.** Sección documentada que mapea con honestidad qué
se implementó y qué quedó fuera y por qué. Fuera de alcance a explicar con
suficiente profundidad para defenderlo en entrevista: opcionalidad conductual de
prepago de créditos, retiro anticipado de depósitos a plazo, riesgo de base entre
índices distintos, tratamiento multi-moneda completo, y la capa de gobernanza
(política de ALM, límites, validación independiente de modelos estilo SR 11-7,
backtesting periódico, documentación auditable).

## 8. Escenarios de tasas — los seis de Basilea IRRBB

Definidos **una sola vez** en `src/scenarios.py` y consumidos por igual por los
Módulos 3 y 4:

1. Paralelo hacia arriba
2. Paralelo hacia abajo
3. Empinamiento (cortas bajan, largas suben)
4. Aplanamiento (cortas suben, largas bajan)
5. Cortas hacia arriba
6. Cortas hacia abajo

- Los seis se aplican como **choques instantáneos** sobre la curva.
- **Suelo post-choque (post-shock floor)** parametrizado en el config: parte de un
  valor negativo cerca del tramo corto y sube gradualmente hacia cero en los plazos
  largos.
- Convención: el NII usa típicamente solo los dos paralelos; el EVE usa los seis.
  Seguir esa convención pero dejar ambos configurables.
- El **resultado headline de EVE es el PEOR de los seis** (máximo deterioro), que
  es lo que se compara contra el umbral del 15%.

## 9. Reglas de consistencia entre módulos (inviolables)

1. Activos = Pasivos + Patrimonio, siempre.
2. Las tasas activas por producto son mayores que las pasivas correspondientes.
3. La curva que descuenta en el Módulo 4 es la misma referencia que alimenta las
   tasas variables del Módulo 1.
4. La beta estimada en el Módulo 2 alimenta el NII del Módulo 3.
5. Los seis escenarios se definen una sola vez.
6. Todos los supuestos y parámetros viven en `config/params.py`. **Nunca**
   hardcodeados en medio de la lógica.
7. Todo reproducible con semilla fija.

## 10. Estándares de ingeniería

- Código limpio y modular, un archivo por módulo, docstrings en todo lo público.
- Notebooks solo para exploración, nunca lógica de producción.
- **Tests unitarios de las funciones críticas de valoración**, como mínimo:
  - un bono a tasa par vale su nominal;
  - el balance cuadra;
  - el modelo de depósitos recupera la beta del ground truth dentro de un margen;
  - un balance perfectamente calzado da ΔNII y ΔEVE cercanos a cero.
- **Análisis de sensibilidad como práctica estándar**: mostrar cómo se mueven los
  resultados ante rangos de parámetros inciertos (sobre todo beta y porción core),
  no un único número puntual.
- **Reporting estilo ALCO** al cierre: presentar los resultados como lo haría un
  equipo de tesorería ante su comité de activos y pasivos, incluyendo la tabla de
  los seis escenarios con su ΔNII y ΔEVE.

## 11. Puntos abiertos a cuestionar antes de codificar el Módulo 0

1. **18% de hipotecario** es el principal motor del gap de duración. Subirlo a 25%
   acerca el banco al umbral del 15% y cambia el tono del caso.
2. **β⁺ = 0,25 para vista** es conservador (franquicia minorista fuerte). Un banco
   con clientela más sensible al precio estaría en 0,40–0,50.
3. **Vida media core de 4,2 años** queda holgada bajo el tope IRRBB de 5 años para
   minorista transaccional, pero relativamente cerca del techo.

**Los tres quedaron resueltos** al arrancar la implementación: (1) mix hipotecario
se mantiene en 18% y el 25% pasa a ser sensibilidad declarada; (2) β⁺ vista se
mantiene en 0,25 con barrido 0,25/0,35/0,45; (3) "vida media" se define como **vida
promedio** (mean life), no half-life. Ver §12.3.

## 12. Módulo 0 — Decisiones tomadas durante la implementación

Todo lo de esta sección se decidió *después* del diseño de §6, al chocar con los
datos. Se documenta aquí porque un lector externo (o el yo de dentro de seis meses)
tiene que poder distinguir qué se planeó de qué se aprendió.

### 12.1 Corrección de especificación en §6.4 — patología de signo

§6.4 escribe `d_t = d_{t-1} + λ·(α + β^± · r_t − d_{t-1}) + ε_t`, con β^± aplicado
al **nivel objetivo**. Tomada al pie de la letra la fórmula está mal: con r = 5%, el
objetivo en régimen de subida es `α + 0,25·5% = 1,30%` y en régimen de bajada
`α + 0,55·5% = 2,80%`. El primer recorte de tasas hace **subir** el objetivo 150 pb.
Generando así, la beta estimada de vista salía en **−0,15**.

La asimetría pertenece al **traspaso del cambio**, no al nivel:

    d*_t = d*_{t-1} + β^± · Δr_t                 traspaso asimétrico de corto plazo
    d*_t = d*_t + κ · (α + β̄·r_t − d*_t)         ancla competitiva, β̄ = (β⁺+β⁻)/2
    d_t  = d_{t-1} + λ · (d*_t − d_{t-1}) + ε_t   ajuste parcial

Es además cómo la industria usa la palabra: *de una subida de 100 pb trasladamos 25*.
El ancla (`κ = 0,02` mensual) impide que el efecto trinquete lleve la tasa a cero tras
un ciclo completo. El ground truth β⁺/β⁻/λ de §6.4 queda intacto.

### 12.2 Respecificación del control #7 — el umbral era inalcanzable

§6.8 pedía R² > 0,5 en la regresión contemporánea Δd ~ Δr. **Medido con ruido cero,
vista topa en R² = 0,50 y no puede pasar de ahí**: con λ = 0,30 sólo un tercio de la
respuesta ocurre en el mes del movimiento y el resto llega después, cuando Δr ya vale
cero. No era un problema de calibración sino del propio modelo de §6.4.

El control ahora regresa Δd sobre Δr **y sus 6 rezagos**, y exige t > 3 y R² > 0,5
sobre la **suma** de coeficientes. Medido: R² = 0,69 / 0,89 / 0,98 y Σβ̂ = 0,32 / 0,52
/ 0,86 (vista/ahorro/plazo). Σβ̂ es el traspaso acumulado — el número que un ALCO pide.

Efecto colateral bueno: #7 y #8 quedan como un par. **#7 demuestra que el proceso
generador SÍ es recuperable con la especificación correcta; #8 que la ingenua falla.**
Eso es exactamente la tesis del Módulo 2.

### 12.3 Vida promedio ≠ half-life

Bajo decaimiento exponencial difieren en 1/ln2 ≈ 1,443. **IRRBB capea la vida
promedio de repreciación** del core: 5 años minorista transaccional, 4,5 minorista no
transaccional, 4 mayorista. Con 4,2 años de vida promedio el banco queda holgado; si
esos 4,2 fueran half-life, la vida promedio sería 6,06 años y el core verdadero
**excedería el tope**. `ground_truth.json` registra ambas cifras.

### 12.4 Otras decisiones de modelado

- **Efectivo es sensible, no insensible.** §5 lo etiqueta "no sensible / overnight",
  que son dos cosas distintas. El encaje remunerado repacta de inmediato: es de los
  activos más sensibles del balance. Lo que no es, es fuente de margen — rinde por
  debajo de la referencia. Tratarlo como tasa cero regalaba ~60 M de ingreso anual.
- **Las cohortes amortizables entran por su principal vivo**, no por el monto
  originado. Una cohorte hipotecaria de hace ocho años ya devolvió cerca de un tercio
  del principal. Sin ese factor el libro parece más viejo — y más corto — de lo que es.
- **Duración de pasivos: dos convenciones, reportadas por separado.** La de
  runoff/contractual (sin ajustar por beta) es la de los objetivos de §6.1; la
  *efectiva* multiplica por (1−β⁺) y sale bastante más corta. Mezclarlas es lo que hace
  que dos áreas del mismo banco reporten duraciones de pasivo distintas por un factor
  de dos y nadie sepa cuál usar.
- **Los saldos anclan su nivel en la fecha de corte** al share de §5, de modo que la
  serie histórica y la tabla de instrumentos digan lo mismo el último día.
- **Las series latentes** (core verdadero, migración) viajan aparte de las
  observables. El Módulo 2 no debe verlas salvo para reportar su error.

### 12.5 Estado de calibración: 5 de 6 objetivos en rango

| Métrica | Medido | Objetivo §6.1 | |
|---|---|---|---|
| Duración activos sensibles | 2,15 a | 2,2 – 2,5 | fuera por 0,05 |
| Duración pasivos (post-NMD) | 1,84 a | 1,8 – 2,1 | ✓ |
| Gap de duración | +0,49 a | +0,4 – +0,6 | ✓ |
| ΔEVE peor / Tier 1 | −11,5% | −9% a −13% | ✓ |
| ΔNII 12m (+200 pb) | +1,72% | +1,5% – +3,5% | ✓ |
| NIM | 3,33% | 2,8% – 3,6% | ✓ |

**La tensión pedagógica se sostiene**: ante +200 pb el NII mejora (+1,72%) y el EVE se
deteriora (−11,5% del Tier 1). Ese signo opuesto es el resultado central del proyecto.

La duración de activos queda 0,05 a por debajo del rango, y **no se movió la meta para
taparlo** (§6.1 se pre-registró justamente para impedirlo). La causa es identificable:
con 32,4% de cartera comercial variable que repacta trimestralmente, 4% de efectivo y
7,2% de tarjetas, el activo de este banco es estructuralmente corto. El desvío es
sensible a la convención de añejamiento de las cohortes y va al README como tal.

Advertencia sobre estas dos últimas filas: ΔEVE y ΔNII son **aproximaciones de primer
orden** (duración del gap y gap estático a 12 meses), calculadas como diagnóstico de
calibración. Los Módulos 3 y 4 las recalculan con proyección y revaluación completas,
y la diferencia entre ambos métodos es uno de los entregables.

### 12.6 Cómo correr el gate

```
pip install -r requirements.txt
python run_module0.py     # exit≠0 si falla algún ERROR; informe en data/
python -m pytest tests/ -q
```

## 13. Módulo 1 — Brecha de repreciación

`src/balance.py`, `run_module1.py` → `data/gap_report.md`. 126 tests en verde.

### 13.1 El hallazgo

| | |
|---|---|
| Gap contractual acumulado a 12 meses | **−10.280 M = −20,6% de los activos** |
| Estado vs. política interna | **ALERTA** (umbral de alerta: 20%) |
| RSA/RSL a 12 meses | 0,72 |
| Lo que eso predice | El margen financiero **cae** cuando suben las tasas |
| Lo que dice el Módulo 0 | **ΔNII +1,72%** ante +200 pb |

**Los dos números son correctos. El mal especificado es el gap.**

La causa está en una sola línea del tratamiento: vista y ahorro —22.000 M, el 44% del
fondeo— entran en la banda más corta porque contractualmente el cliente retira mañana.
Eso equivale a suponer que el banco traslada el **100%** de cualquier movimiento de
tasas a esos depósitos. La beta verdadera de vista es 0,25. El gap contractual está
midiendo un banco que no existe.

Ésta es, históricamente, la razón por la que la industria dejó de usar el análisis de
brechas como herramienta única. El hallazgo está bajo test
(`test_el_gap_contradice_al_margen_financiero`).

El problema es **doble**, y el Módulo 2 debe arreglar las dos mitades: la **cuantía**
del traspaso (la beta) y el **momento** en que ocurre (el plazo conductual del core).
Corregir sólo la primera dejaría el EVE del Módulo 4 igual de mal.

### 13.2 Decisiones de diseño

- **Ninguna beta ni supuesto conductual en el Módulo 1.** La beta es producto del
  Módulo 2; usarla antes de estimarla sería importar el resultado al módulo anterior y
  dejar al Módulo 2 sin nada que demostrar.
- **Las 19 bandas del marco estandarizado IRRBB**, con sus **puntos medios en años**
  (`IRRBB_BANDS`), más un agrupamiento a 8 para el informe (`BANDAS_ALCO`). Los puntos
  medios se definen ahora porque son exactamente los que el Módulo 4 necesita para el
  EVE estandarizado. Intervalos abiertos por abajo y cerrados por arriba.
- **Reparto amortizado del saldo, no bullet.** El principal devuelto antes del
  horizonte de repreciación se asigna a la banda en que se cobra; sólo el remanente va
  a la banda del repricing. Una hipoteca francesa a 20 años no expone su saldo íntegro
  dentro de 20 años. Con 16.200 M de cartera amortizable la diferencia es material: el
  gap a 12 meses pasa de −13.192 M (bullet) a −10.280 M (amortizado). El informe
  publica ambos.
- **Gap por vencimiento publicado al lado**, no como métrica sino como el error que el
  módulo existe para no cometer. Sí es la tabla relevante para riesgo de **liquidez**,
  que es otro problema y no el de este proyecto.
- **Reconciliación obligatoria** de los 50.000 M contables a RSA/RSL. Un informe de gap
  que no cuadra con el balance es un informe que nadie audita.
- **No sensibles y patrimonio quedan fuera de las bandas.** No repactan nunca;
  asignarles una banda sería inventar exposición.
- **`GAP_THRESHOLDS` son de apetito interno, no regulatorios.** El único umbral que
  Basilea fija es el del *outlier test* de EVE (15% del Tier 1, Módulo 4). Se incluyen
  porque un informe de gap sin umbral es un número sin decisión asociada.

### 13.3 Dos notas honestas

- Con datos mensuales la banda **overnight queda vacía**. Los NMD son overnight de
  verdad; la granularidad del generador los agrupa en O/N–1M.
- El punto medio que Basilea da para overnight, 0,0028 años, es 1/365 redondeado, y
  bajo base 30/360 cae tres diezmilésimas de mes fuera del tope de su propia banda. Se
  conserva el número regulatorio en lugar de "arreglarlo": el punto medio es el plazo
  al que el marco estandarizado descuenta, y ahí manda el texto.

### 13.4 Cómo correrlo

```
python run_module1.py     # informe en data/gap_report.md
```

## 14. Módulo 2 — Modelo conductual de NMD

`src/deposits.py`, `run_module2.py` → `data/nmd_report.md`. 158 tests en verde.

### 14.1 La beta se recupera; el plazo no

| Producto | β̂⁺ | real | error | β̂⁻ | real | error | R² |
|---|---|---|---|---|---|---|---|
| vista | 0,275 | 0,25 | +0,025 | 0,505 | 0,55 | −0,045 | 0,81 |
| ahorro | 0,466 | 0,45 | +0,016 | 0,744 | 0,75 | −0,006 | 0,94 |
| plazo | 0,848 | 0,85 | −0,002 | 0,938 | 0,90 | +0,038 | 0,98 |

Estimador: `Δd_t = a₀ + Σa_k·Δr⁺_{t−k} + Σb_k·Δr⁻_{t−k}`, 12 rezagos. Separar la parte
positiva de la negativa es lo único que hace visible la asimetría; los 12 rezagos son
necesarios porque con λ = 0,30 sólo un tercio del traspaso ocurre en el mes del
movimiento.

Proporción estable (destendenciada, mínimo de la muestra): **0,897** vista y **0,913**
ahorro, contra `core_share` estructural de 0,85 y 0,90. La sobreestimación no es ruido:
el estimador mide *lo que no se va en volumen* y el parámetro describe *composición*.

### 14.2 El hallazgo: la vida del núcleo no es identificable

Cuatro bancos idénticos salvo la vida promedio **verdadera** del núcleo de vista:

| Vida verdadera | Correlación vs. base | Dif. relativa máx. |
|---|---|---|
| 3,0 a | 0,999994 | 0,11% |
| 6,0 a | 0,999997 | 0,09% |
| 10,0 a | 0,999987 | 0,17% |

Una vida de 3 años y una de 10 producen la misma serie observable. **No es
identificación débil: es información cero.** El saldo agregado es la suma de un stock
que se va y otro que entra, y no permite separar los dos flujos.

**Esto corrige §12.4.** Allí argumenté que la construcción runoff + originación hacía
λ recuperable. La hace *estructural*, no *identificable desde el agregado*.

Y es el argumento de fondo de por qué IRRBB **acota** el plazo del núcleo en vez de
pedir una estimación mejor: es un parámetro que el banco no puede falsar con los datos
que suele tener, y que además empuja el EVE en la dirección que al banco le conviene —
un núcleo más largo abarata el descalce en el papel. Los topes no son conservadurismo
arbitrario, son la respuesta correcta a un problema de identificación.

**Consecuencia de método**: cuando un parámetro no está identificado, el entregable
honesto no es un número puntual sino el rango que produce. La sensibilidad deja de ser
un apéndice.

### 14.3 Tres plazos distintos, tres preguntas distintas

| Concepto | Pregunta | Vista |
|---|---|---|
| Vida de volumen | ¿cuánto tarda en irse el dinero? | 4,0 a supuesto (real 4,2) |
| Plazo de réplica | ¿cuánto tarda en repactar el precio? | 0,37 a estimado |
| Plazo IRRBB del núcleo | ¿qué asigno a bandas para EVE? | 4,0 a (tope 5,0) |

El portafolio de réplica se publica **como contraste, no como insumo**. Usarlo para
asignar bandas mandaría los 22.000 M a la banda corta y dejaría el gap tan mal
especificado como en el Módulo 1.

**La vida supuesta no sale del ground truth** (4,0 y 3,5 contra 4,2 y 3,0 reales).
Copiar el valor verdadero fabricaría un acierto.

### 14.4 Núcleo bajo el marco estandarizado

`núcleo = estable × (1 − β⁺)`, luego topes por categoría de cliente:

| Producto | Estable | β⁺ | Núcleo | Tope | ¿Muerde? |
|---|---|---|---|---|---|
| vista (minorista transaccional) | 0,897 | 0,275 | 0,651 | 0,90 / 5,0 a | no |
| ahorro (minorista no transaccional) | 0,913 | 0,466 | 0,488 | 0,70 / 4,5 a | no |

El paso `× (1 − β)` es la nota conceptual de §6.5 hecha operación: la porción que no
repacta no es la que no se va.

### 14.5 El gap corregido — y hasta dónde llega

| Tratamiento de los NMD | Gap acum. 12m | Sobre activos | Estado |
|---|---|---|---|
| Contractual (Módulo 1) | −10.280 M | −20,6% | **ALERTA** |
| Conductual (Módulo 2) | −675 M | −1,3% | **DENTRO DE POLÍTICA** |

De un descalce que exigiría plan de acción a un banco esencialmente calzado. Lo que
cambió no fue el balance: fue reconocer que de una subida de 100 pb el banco traslada
del orden de 27 pb a las cuentas transaccionales, no 100.

**Sin sobreafirmar.** El gap conductual sigue levemente negativo mientras el margen
mejora (+1,72%). Un gap cercano a cero es *compatible* con eso, pero no lo predice, y
la razón es estructural: **cualquier** análisis de brechas trata cada peso que repacta
dentro de su banda como si trasladara el 100% del choque. El modelo conductual arregla
el *momento* y la *proporción*; no puede arreglar que dentro de la banda el traspaso se
suponga completo. El gap pasó de mal especificado a bien especificado — y sigue siendo
una aproximación de primer orden. El signo del margen sale de proyectarlo: Módulo 3.
Está bajo test (`test_el_gap_conductual_sigue_sin_predecir_el_signo_del_margen`).

### 14.6 Sensibilidad: el entregable

| Vida supuesta | Vista aplicada | ¿Tope muerde? | Gap 12m / activos |
|---|---|---|---|
| 2,0 a | 2,0 a | no | −5,4% |
| 3,0 a | 3,0 a | no | −2,6% |
| 4,0 a | 4,0 a | no | −1,1% |
| 5,0 a | 5,0 a | sí (ahorro) | −0,2% |
| 7,0 a | **5,0 a** | sí | **−0,2%** |

El rango honesto del gap a 12 meses es **−5,4% a −0,2%**. Y la última fila es el
exhibit: **a partir de 5 años el resultado se congela**, porque el tope impide seguir
estirando el supuesto. Ahí se ve exactamente qué compra la regulación.

Otros dos ejes: la proporción estable ±10 pp mueve el gap entre −3,5% y +0,7%; usar β̄
en lugar de β⁺ para el corte lo lleva de −1,3% a −5,2%. Elegir la beta del corte no es
un detalle técnico.

### 14.7 Cómo correrlo

```
python run_module2.py     # informe en data/nmd_report.md
```

## 15. Módulo 3 — Margen financiero (NII)

`src/scenarios.py` (los seis, definidos UNA vez), `src/nii.py`, `run_module3.py` →
`data/nii_report.md`. 202 tests en verde.

Choques: calibración **USD** del estándar — paralelo 200 pb, cortas 300, largas 150.
Suelo post-choque de −100 pb subiendo 5 pb/año hasta 0% a los 20 años.

### 15.1 ΔNII a 12 meses — NII base 1.566 M

| Escenario | Δ% | | Escenario | Δ% |
|---|---|---|---|---|
| **Paralelo arriba** | **+2,77%** | | Aplanamiento | +4,44% |
| **Paralelo abajo** | **+4,13%** | | Cortas arriba | +5,17% |
| Empinamiento | +2,22% | | Cortas abajo | +2,26% |

### 15.2 El resultado: el banco gana en las dos direcciones

Ante +200 pb el margen mejora. Ante −200 pb **también**. Un choque simétrico subiendo
el margen en ambos sentidos parece error de signo; no lo es. En la bajada el ingreso
cae 343 M pero **el costo de fondeo cae 407 M**.

El mecanismo es la asimetría del Módulo 2: al subir el banco traslada β⁺ ≈ 0,28 y
retiene el resto; al bajar traslada β⁻ ≈ 0,51 y se queda la diferencia. **Es una
posición larga en volatilidad de tasas**: la franquicia de depósitos no es sólo fondeo
barato, es una opción del banco contra sus depositantes que paga en ambos sentidos.

| Beta usada | Paralelo arriba | Paralelo abajo |
|---|---|---|
| Asimétrica (β̂⁺ / β̂⁻) | +2,77% | +4,13% |
| Simétrica (promedio) | **+0,05%** | +1,10% |

Con beta única la ganancia **no se reparte mal: se borra**, y el banco parece neutral
al riesgo de tasa en el margen. Ésa es la respuesta a por qué el Módulo 2 estimó dos
betas.

### 15.3 La acotación: el piso de la tasa de depósito

β⁻ sólo paga mientras quede tasa que recortar. Un banco no puede cobrarle al minorista
por depositar.

| Nivel de partida (3M) | ΔNII arriba | ΔNII abajo | Holgura al piso |
|---|---|---|---|
| 3,27% (hoy) | +2,77% | **+4,13%** | **5 pb** |
| 2,27% | +2,96% | +0,45% | 0 pb |
| 1,27% | +3,18% | **−10,26%** | 0 pb |
| 0,27% | +3,70% | **−12,46%** | 0 pb |

Al banco le quedan **5 pb** de holgura antes de que el producto más barato choque
contra cero. Es la razón por la que una franquicia de depósitos vale mucho menos en un
entorno de tasas cero, y por la que la banca europea y japonesa pasó una década sin
poder monetizar la suya. Reportar el ΔNII sin esta acotación vendería como estructural
una ganancia que depende del nivel de partida.

### 15.4 Reconciliación con el Módulo 0

| Método | ΔNII (+200 pb) |
|---|---|
| Gap estático a 12 meses (Módulo 0) | +1,72% |
| Proyección completa (Módulo 3) | +2,77% |

Miden lo mismo por caminos distintos. Coincidir en signo y orden de magnitud es la
validación; diferir en signo indicaría un error en uno de los dos. Bajo test.

### 15.5 Decisiones de diseño

- **Balance constante y margen constante**: cada instrumento conserva su spread sobre
  la curva y sólo se mueve la curva. Es el estándar regulatorio; no predice cómo
  respondería un banco real, mide la sensibilidad del balance *actual*.
- **Traspaso con rezagos**: los coeficientes de rezago del Módulo 2 **son** el perfil
  mensual de traspaso. Suponerlo inmediato sobrestima el costo de fondeo del primer
  trimestre y sesga el ΔNII a la baja. Bajo test.
- **Los seis se calculan siempre**; la convención de §8 decide qué se titula, no qué se
  computa.
- **`src/scenarios.py` es la única fuente de verdad** sobre qué significa cada
  escenario. El Módulo 4 lo consume sin redefinir nada — es lo que permite poner NII y
  EVE en la misma fila del informe al comité sin comparar peras con manzanas.

### 15.6 Cómo correrlo

```
python run_module3.py     # informe en data/nii_report.md
```

## 16. Módulo 4 — Valor económico del patrimonio (EVE)

`src/eve.py`, `run_module4.py` → `data/eve_report.md`. 228 tests en verde.

EVE base **4.500 M** (= patrimonio contable, por construcción) · Tier 1 4.000 M ·
umbral del *outlier test* −600 M.

### 16.1 El resultado que el proyecto existe para demostrar

| Escenario | ΔNII 12m | ΔEVE / Tier 1 | ¿Outlier? |
|---|---|---|---|
| **Paralelo arriba** | **+2,77%** | **−17,0%** | **SÍ** ⟵ peor |
| Paralelo abajo | +4,13% | +19,9% | no |
| Empinamiento | +2,22% | −7,7% | no |
| Aplanamiento | +4,44% | +3,9% | no |
| Cortas arriba | +5,17% | −4,0% | no |
| Cortas abajo | +2,26% | +4,1% | no |

**Ante +200 pb el margen mejora y el valor económico se deteriora.** Son dos
horizontes: el NII mira quién repacta antes, el EVE mira quién tiene más duración.
Este banco cobra rápido en el activo indexado y paga despacio en sus depósitos —gana
margen— mientras carga 9.000 M de hipotecas fijas originadas con la referencia en
1,5% —pierde valor—. Gestionar solo por NII llevaría a celebrar una subida que le
está destruyendo patrimonio. Bajo test en `test_la_tension_central_del_proyecto`.

### 16.2 Convención: calibración a la par

Para cada instrumento se resuelve el spread `s` tal que `VP(flujos, curva + s) = saldo`
hoy. Así el EVE de partida no inventa plusvalías latentes y el choque mueve **solo la
parte libre de riesgo**: el ΔEVE es riesgo de tasa puro, no una mezcla con crédito.

**Flujos de repreciación, no contractuales.** Un crédito comercial a 5 años que repacta
cada trimestre vuelve a valer su nominal en cada reset. Valorarlo hasta vencimiento
—como hice en la primera versión— infla la duración del activo de 1,98 a 2,96 años y
lleva el ΔEVE de −17% a −37%. Es la misma regla del gap del Módulo 1, y esa coherencia
no es opcional: si gap y EVE usaran horizontes distintos describirían balances
distintos. Bajo test.

### 16.3 Duración y convexidad: dónde falla lo lineal

Duración efectiva (por diferencias finitas, no analítica, porque los flujos de los NMD
salen de un modelo de comportamiento): activo **1,98 a**, pasivo **1,37 a**, gap
**+0,74 a**.

| Choque | Revaluación completa | 1er orden | 2do orden | Error lineal |
|---|---|---|---|---|
| +50 pb | −181 | −184 | −180 | −2,0% |
| +200 pb | −682 | −736 | −679 | −8,0% |
| +400 pb | −1.265 | −1.472 | −1.244 | −16,4% |
| +800 pb | −2.190 | −2.944 | −2.033 | **−34,4%** |

El error crece con el **cuadrado** del choque. Es el argumento cuantitativo de por qué
un límite de ALM expresado solo en duración es insuficiente para estrés: mide bien el
riesgo del día a día y subestima justo el que motiva tener límites. La convexidad
recupera casi todo.

### 16.4 Reconciliación con el Módulo 0 — un error conceptual costoso

El diagnóstico del Módulo 0 daba **−11,5%**; aquí sale **−17,0%**. La diferencia no es
de método sino **de definición de núcleo**.

Aquel diagnóstico usaba el núcleo **de volumen** (0,90 en vista) como si fuera el de
repreciación. IRRBB define el núcleo como la porción que **no repacta**: estable ×
(1−β) = **0,65**. Un cuarto del depósito a la vista es dinero que se queda **y aun así
sigue a la tasa de mercado**.

Es exactamente la nota de §6.5, y aquí se ve cuánto cuesta ignorarla: casi seis puntos
de Tier 1, la diferencia entre pasar el *outlier test* y no pasarlo. El objetivo
pre-registrado de §6.1 (−9% a −13%) **no se cumple**, y no se movió la meta: el rango
se fijó sobre una definición de núcleo que resultó ser la equivocada.

### 16.5 El hallazgo que cierra el proyecto

| Vida supuesta del núcleo | Aplicada | ΔEVE / Tier 1 | ¿Outlier? |
|---|---|---|---|
| 2,0 a | 2,0 a | −26,5% | SÍ |
| 3,0 a | 3,0 a | −21,1% | SÍ |
| 4,0 a (caso base) | 4,0 a | −16,0% | SÍ |
| 5,0 a | 5,0 a | −12,2% | **no** |
| 7,0 a | **5,0 a (tope)** | −12,2% | **no** |

**El veredicto regulatorio cambia dentro del rango de un supuesto que el Módulo 2
demostró no identificable desde los datos.** El banco es o no es *outlier* según una
hipótesis que no puede falsar.

Y ahí se ve para qué sirve el tope de 5 años: es lo único que impide al banco
suponerse fuera del problema. Con el tope, el mejor caso alcanzable es −12,2% — justo
por debajo del umbral. El supervisor acotó exactamente el margen de maniobra que
importaba. Bajo test en
`test_el_veredicto_regulatorio_depende_del_supuesto_no_identificado`.

### 16.6 Cómo correrlo

```
python run_module4.py     # informe en data/eve_report.md
```
