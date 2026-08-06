# Simulador ALM / IRRBB

Implementación del núcleo cuantitativo del marco de Basilea sobre **riesgo de tasa de
interés en el libro bancario** (IRRBB, Comité de Basilea, 2016) sobre un banco
universal sintético de USD 50.000 M: las métricas **NII** y **EVE** bajo los seis
escenarios de choque prescritos.

Python, sin dependencias más allá de numpy / pandas / scipy. 238 tests.

```bash
pip install -r requirements.txt
python run_module0.py    # genera el banco y corre 11 controles de calidad (gate)
python run_module1.py    # brecha de repreciación
python run_module2.py    # modelo conductual de depósitos sin vencimiento
python run_module3.py    # margen financiero proyectado
python run_module4.py    # valor económico del patrimonio
python -m pytest tests/ -q
```

> **En Windows sin `python` en el PATH**, usar el lanzador: `py run_module0.py`,
> `py -m pytest tests/ -q`. Dentro de un entorno virtual `python` funciona en
> cualquier sistema operativo y es la forma canónica.

**Documento de teoría:** [`docs/Teoria_ALM_IRRBB.pdf`](docs/Teoria_ALM_IRRBB.pdf) (68 pp.)
desarrolla la teoría detrás de cada paso y el porqué de cada decisión: la matemática de
curvas, la literatura de beta de depósitos, las derivaciones de los hallazgos, la
metodología de validación y una crítica honesta del alcance. Fuente editable en
[`docs/teoria.html`](docs/teoria.html).

---

## 1. El problema

Un banco toma depósitos que puede repactar cuando quiera y presta a plazos largos con
tasa fija. Cuando las tasas de mercado se mueven, esas dos patas del balance no
reaccionan igual, y la diferencia se manifiesta de **dos formas distintas que pueden
apuntar en direcciones opuestas**:

- el **margen financiero (NII)** de los próximos doce meses, que depende de quién
  repacta antes;
- el **valor económico del patrimonio (EVE)**, valor presente de todos los flujos
  futuros, que depende de quién tiene más duración.

Basilea exige ambas porque un banco puede estar ganando margen este año mientras
destruye patrimonio económico. Este proyecto lo demuestra con números propios.

## 2. Resultado principal — la tabla que va al comité

| Escenario | ΔNII 12m | ΔEVE / Tier 1 | ¿Outlier? |
|---|---|---|---|
| **Paralelo arriba (+200 pb)** | **+2,77%** | **−17,0%** | **SÍ** ⟵ peor |
| Paralelo abajo (−200 pb) | +4,13% | +19,9% | no |
| Empinamiento | +2,22% | −7,7% | no |
| Aplanamiento | +4,44% | +3,9% | no |
| Cortas arriba | +5,17% | −4,0% | no |
| Cortas abajo | +2,26% | +4,1% | no |

NII base 1.566 M · EVE base 4.500 M · Tier 1 4.000 M · umbral del *outlier test* −600 M.

**Ante +200 pb el margen mejora y el valor económico se deteriora.** Este banco cobra
rápido en su cartera comercial indexada y paga despacio en sus depósitos —gana
margen— mientras carga 9.000 M de hipotecas a tasa fija originadas cuando la
referencia estaba en 1,5% —pierde valor—. Gestionar solo por NII llevaría a celebrar
una subida de tasas que le está destruyendo patrimonio.

## 3. Los cinco hallazgos

### 3.1 El gap de repreciación contractual mide un banco que no existe

El gap acumulado a 12 meses da **−20,6% de los activos** (estado ALERTA) y predice que
el margen cae. El margen sube. Los dos números son correctos; el mal especificado es
el gap: mete los 22.000 M de depósitos sin vencimiento en la banda más corta porque
contractualmente el cliente retira mañana, lo que equivale a suponer que el banco
traslada el **100%** de cualquier movimiento. Traslada el 27%.

Con el tratamiento conductual del Módulo 2 el gap pasa a **−1,3%** y sale de alerta.

Es la razón histórica por la que la industria abandonó el análisis de brechas como
herramienta única.

### 3.2 La beta asimétrica es una posición larga en volatilidad de tasas

| Producto | β̂⁺ | real | β̂⁻ | real |
|---|---|---|---|---|
| vista | 0,275 | 0,25 | 0,505 | 0,55 |
| ahorro | 0,466 | 0,45 | 0,744 | 0,75 |
| plazo | 0,848 | 0,85 | 0,938 | 0,90 |

El banco traslada poco de una subida y mucho de una bajada, así que **gana margen en
ambas direcciones**. Con una beta simétrica la ganancia cae a +0,05% / +1,10%: no se
reparte mal, **se borra**.

Y tiene fecha de caducidad. La ganancia por bajadas necesita que quede tasa que
recortar, y la remuneración de un depósito se detiene en cero:

| Nivel de partida (3M) | ΔNII arriba | ΔNII abajo |
|---|---|---|
| 3,27% (hoy) | +2,77% | **+4,13%** |
| 1,27% | +3,18% | **−10,26%** |

Al banco le quedan **5 puntos básicos** de holgura. Es por lo que una franquicia de
depósitos vale mucho menos en un entorno de tasas cero.

### 3.3 El plazo conductual del núcleo no es estimable

Cuatro bancos idénticos salvo la vida promedio **verdadera** del núcleo (3 / 4,2 / 6 /
10 años) producen series de saldo con correlación **0,99999** y diferencia máxima del
0,17%. No es identificación débil: es información cero. El saldo agregado es la suma
de un stock que se va y otro que entra, y no permite separar los dos flujos.

Ése es el argumento de fondo por el que IRRBB **acota** el plazo del núcleo (5 años
minorista transaccional, 4,5 no transaccional, 4 mayorista) en vez de pedir una
estimación mejor: es un parámetro que el banco no puede falsar con los datos que
suele tener, y que empuja el EVE en la dirección que le conviene.

**Consecuencia de método**: cuando un parámetro no está identificado, el entregable
honesto no es un número puntual sino el rango que produce.

### 3.4 El veredicto regulatorio depende de ese supuesto no identificado

| Vida supuesta del núcleo | Aplicada | ΔEVE / Tier 1 | ¿Outlier? |
|---|---|---|---|
| 3,0 a | 3,0 a | −21,1% | SÍ |
| 4,0 a (caso base) | 4,0 a | −16,0% | SÍ |
| 5,0 a | 5,0 a | −12,2% | **no** |
| 7,0 a | **5,0 a (tope)** | −12,2% | **no** |

El banco **es o no es** *outlier* según una hipótesis que no puede verificar. Y ahí se
ve para qué sirve el tope de 5 años: es lo único que impide al banco suponerse fuera
del problema. Con el tope, el mejor caso alcanzable queda justo por debajo del umbral.

### 3.5 La aproximación por duración falla donde más importa

| Choque | Revaluación completa | Primer orden | Error |
|---|---|---|---|
| +50 pb | −181 | −184 | −2,0% |
| +200 pb | −682 | −736 | −8,0% |
| +800 pb | −2.190 | −2.944 | **−34,4%** |

El error crece con el **cuadrado** del choque. Un límite de ALM expresado solo en
duración mide bien el riesgo del día a día y subestima precisamente el de estrés, que
es el que motiva tener límites.

## 4. El banco sintético

Universal, USD 50.000 M, corporativo + minorista, financiado con depósitos.
Loan-to-deposit ≈ 103%. Series de 120 meses con un ciclo completo de tasas
(1,5% → 5,5% → 3,25%), curva invertida en el pico y un episodio de estrés de liquidez.

| Activo | % | | Pasivo y patrimonio | % |
|---|---|---|---|---|
| Efectivo y encaje | 4,0 | | Depósitos a la vista | 22,0 |
| Inversiones | 18,0 | | Cuentas de ahorro | 22,0 |
| Crédito comercial (85% variable) | 32,4 | | Depósitos a plazo | 26,0 |
| Crédito hipotecario (90% fija, 15–20 a) | 18,0 | | Interbancario y deuda | 15,0 |
| Consumo, vehículos, tarjetas | 21,6 | | Otros pasivos | 6,0 |
| Otros activos | 6,0 | | Patrimonio | 9,0 |

**Por qué sintético.** El objetivo no es estimar bien un banco concreto, es demostrar
que las estimaciones son correctas. Con datos reales la beta verdadera de los
depósitos es inobservable y el modelo solo podría reportar un número sin contraste.
Aquí la fijamos nosotros en `data/ground_truth.json`, así que el modelo reporta su
**error** — que es lo que pediría un validador independiente.

Los 11 controles de calidad del Módulo 0 actúan como **gate**: los de severidad ERROR
bloquean el avance con código de salida distinto de cero.

## 5. Alcance frente al estándar IRRBB

### Implementado

| Componente | Estado |
|---|---|
| Los seis escenarios de choque prescritos | Sí, calibración USD (200/300/150 pb) |
| Suelo post-choque (−100 pb → 0% a 20 a) | Sí |
| Bandas temporales del marco estandarizado | Las 19, con sus puntos medios |
| ΔNII a 12 meses, balance y margen constantes | Sí |
| ΔEVE con revaluación completa | Sí |
| *Outlier test* contra el 15% del Tier 1 | Sí |
| Tratamiento conductual de NMD | Sí, con los topes de proporción y plazo |
| Separación núcleo / no núcleo vía `estable × (1−β)` | Sí |
| Beta asimétrica de traspaso | Sí, estimada y validada contra ground truth |
| Duración, convexidad y error de la aproximación lineal | Sí |
| Análisis de sensibilidad sobre supuestos no identificados | Sí, es el entregable central |

### Fuera de alcance, y por qué

**Opcionalidad de prepago de créditos.** Es la omisión que más pesaría en este banco.
Un deudor hipotecario tiene una opción de compra sobre su propio préstamo: cuando las
tasas bajan, refinancia. El efecto es **convexidad negativa** — la cartera se acorta
justo cuando alargarla sería valioso y se alarga cuando el banco preferiría
recuperar el capital. Va en contra del banco en ambas direcciones. Con 9.000 M de
hipotecas fijas a 15–20 años, modelarlo empeoraría el EVE en los escenarios de bajada
y probablemente cambiaría cuál es el peor de los seis. Requiere un modelo de
prepago (típicamente función del incentivo de refinanciación, la antigüedad de la
cohorte y la estacionalidad) que es un proyecto en sí mismo.

**Retiro anticipado de depósitos a plazo.** La opción simétrica del lado del pasivo:
cuando las tasas suben, el depositante rompe el plazo, paga la penalización y
recoloca. Acorta el pasivo justo cuando el banco necesitaba que fuera largo. Con
13.000 M en depósitos a plazo, ignorarlo sobrestima la cobertura del banco en el
escenario que ya es el peor.

**Riesgo de base.** Aquí todo lo indexado sigue una única referencia. En un banco real
el activo se indexa a una tasa (digamos, la interbancaria a 3 meses) y el fondeo a
otra (la de política, o un índice de depósitos), y el diferencial entre ambas se
mueve por su cuenta. Ese *basis* generó pérdidas muy reales en 2007–2008, cuando los
spreads interbancarios se dispararon mientras las tasas de política caían. Modelarlo
exige múltiples curvas y una estructura de correlación entre ellas.

**Multi-moneda.** El estándar exige calcular el IRRBB por moneda y agregarlo con un
tratamiento asimétrico: las pérdidas se suman enteras y las ganancias solo
parcialmente (habitualmente al 50%), porque no se puede compensar una pérdida en una
moneda con una ganancia en otra que no se piensa convertir. Este proyecto usa moneda
única, así que el problema de agregación no aparece — y es precisamente donde un
banco internacional se juega el resultado.

**Riesgo de crédito.** La calibración a la par lo aísla deliberadamente: el spread de
cada instrumento se fija y solo se mueve la curva libre de riesgo. Un choque de tasas
real vendría acompañado de deterioro crediticio, y el efecto conjunto no es la suma
de los dos por separado.

**Gobernanza y validación (SR 11-7).** Es la mitad del trabajo que este repositorio no
puede contener, y conviene ser explícito porque en una entrevista se pregunta:

- **Política de ALM y límites**: quién aprueba el apetito de riesgo, con qué
  frecuencia se revisa, qué pasa cuando se rompe un límite. En este proyecto los
  umbrales de gap son una constante en `config/params.py`; en un banco son una
  decisión de consejo con un procedimiento de escalamiento documentado.
- **Validación independiente de modelos**: un equipo distinto del que construyó el
  modelo lo replica, cuestiona los supuestos y emite una opinión con hallazgos y
  plazos. El módulo `src/validation.py` es un remedo artesanal de eso: pruebas
  escritas antes de mirar los resultados, que se corren siempre y cuyo veredicto no
  se negocia.
- **Backtesting periódico**: contrastar el NII proyectado contra el realizado y
  explicar la diferencia. Aquí no hay realizado contra el que contrastar.
- **Documentación auditable**: cada supuesto con su fecha, su dueño y su
  justificación. El equivalente aquí es que todos los parámetros viven en un solo
  archivo con su razonamiento escrito al lado, y que el histórico está en git.
- **Gobierno del dato**: linaje, conciliación con la contabilidad, controles de
  completitud. El Módulo 0 reconcilia el balance a cero, que es el primero de esos
  controles y el único implementado.

### Objetivos de calibración: qué se cumplió y qué no

Los criterios de aceptación se fijaron **antes** de generar los datos, para no
acomodar parámetros al resultado. Dos no se cumplen, y no se movió la meta:

| Métrica | Medido | Objetivo | |
|---|---|---|---|
| Duración de activos sensibles | 2,15 a | 2,2 – 2,5 | fuera por 0,05 |
| Duración de pasivos (post-NMD) | 1,84 a | 1,8 – 2,1 | ✓ |
| Gap de duración | +0,49 a | +0,4 – +0,6 | ✓ |
| ΔNII 12m (+200 pb) | +1,72% | +1,5% – +3,5% | ✓ |
| NIM | 3,33% | 2,8% – 3,6% | ✓ |
| **ΔEVE peor / Tier 1** | **−21,9%** | −9% – −13% | **fuera** |

La duración de activos queda corta porque, con 32,4% de cartera comercial que repacta
cada trimestre, este activo es estructuralmente corto.

El ΔEVE incumple por una razón más interesante: el rango se fijó usando el núcleo **de
volumen** de los depósitos como si fuera el de **repreciación**. IRRBB define el
núcleo como la porción que no repacta —estable × (1−β)— que es 0,65 y no 0,90. Un
cuarto del depósito a la vista se queda **y aun así sigue a la tasa de mercado**.
Contarlo como núcleo le atribuye un plazo que no tiene. El objetivo se pre-registró
sobre la definición equivocada; corregirla cuesta casi seis puntos de Tier 1 y es la
diferencia entre pasar el *outlier test* y no pasarlo.

## 6. Estructura

```
config/params.py       Todos los supuestos, con su justificación. Ningún número
                       mágico dentro de la lógica.
src/curves.py          Nelson-Siegel, interpolación, descuento, duración, amortización
src/data_gen.py        Módulo 0 — generador del banco sintético
src/validation.py      Módulo 0 — los 11 controles de calidad
src/balance.py         Módulo 1 — brecha de repreciación
src/deposits.py        Módulo 2 — modelo conductual de NMD
src/scenarios.py       Los seis escenarios IRRBB, definidos UNA sola vez
src/nii.py             Módulo 3 — margen financiero
src/eve.py             Módulo 4 — valor económico del patrimonio
run_module{0..4}.py    Entrypoints; cada uno escribe su informe en data/
tests/                 238 tests
data/ground_truth.json Parámetros verdaderos del generador
```

**Reproducibilidad**: todo sale de `config/params.py` más una semilla. El control #11
del Módulo 0 regenera el dataset y compara hashes SHA-256; si difieren, el gate falla.

## 7. Decisiones de modelado que conviene conocer antes de leer el código

- **Los escenarios se definen una sola vez.** Si el NII y el EVE usaran definiciones
  distintas del mismo choque, la tabla del comité compararía peras con manzanas sin
  que nadie lo notara.
- **Flujos de repreciación, no contractuales.** Un crédito a cinco años que repacta
  cada trimestre no tiene duración de cinco años. Valorarlo hasta vencimiento inflaba
  la duración del activo de 1,98 a 2,96 años y llevaba el ΔEVE de −17% a −37%.
- **Calibración a la par.** El spread de cada instrumento se resuelve para que su
  valor presente iguale su saldo hoy, así el ΔEVE es riesgo de tasa puro.
- **Duración efectiva, no analítica.** La analítica supone que los flujos no cambian
  con la tasa, y eso ya no es cierto para los NMD.
- **El traspaso a depósitos usa el perfil temporal estimado**, no la beta entera desde
  el primer mes: con λ = 0,30 solo un tercio ocurre en el mes del movimiento.
- **La vida supuesta del núcleo no sale del ground truth** (4,0 y 3,5 años contra los
  4,2 y 3,0 verdaderos). Copiar el valor real fabricaría un acierto.

## 8. Nota sobre el proceso

El desarrollo siguió un protocolo de fases con diseño aprobado antes de implementar.
Tres decisiones de diseño se corrigieron al chocar con los datos, y quedan
documentadas en `claude.md` §12–§16 junto con el motivo:

1. La fórmula de traspaso a depósitos tenía una patología de signo que hacía **subir**
   la remuneración del depositante cuando el banco central bajaba tasas.
2. Un umbral de validación era inalcanzable por construcción del propio modelo:
   medido con ruido cero, el R² tope era exactamente el umbral exigido.
3. El núcleo de volumen se usó como núcleo de repreciación en el diagnóstico inicial,
   lo que hacía parecer al banco más cubierto de lo que está.

Las tres están bajo test para que no reaparezcan.
