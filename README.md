# TP3 - Perceptron Simple y Multicapa


## Estructura

- `validation/`: validaciones de las herramientas pedidas en el TP.
- `exercise1/`: Ejercicio 1, distillation / fraude.
- `exercise2/`: Ejercicio 2, clasificación de dígitos con perceptrón multicapa.
- `exercise3/`: Ejercicio 3, mejora con más datos y manejo de desbalanceo de clases.


## Datasets

Los scripts no dependen de rutas absolutas particulares de una instalación local.

Podés usar cualquiera de estas dos opciones:

1. Extraer la carpeta de datos en la raíz del repositorio con alguno de estos nombres:
   - `data and documentation/`
   - `data_and_documentation/`
2. O definir una variable de entorno:

```bash
export TP3_DATA_DIR="/ruta/a/data and documentation"
```

Archivos esperados:

- `fraud_dataset.csv`
- `fraud_dataset_documentation.pdf`
- `digits.csv`
- `digits_test.csv`
- `more_digits.csv`
- `digit_dataset_loader.py`

> **Nota:** `more_digits.csv` es el conjunto adicional utilizado en el Ejercicio 3 (`more_data_digits.csv` según la consigna del TP).

## Dependencias

Mínimas para correr el código:

```bash
pip install numpy
```

Opcionales:

```bash
pip install matplotlib pandas
```

## Ejecución

### Validaciones base

```bash
python3 validation/single-layer-step-perceptron.py
python3 validation/single-layer-step-perceptron.py --xor
python3 validation/single-layer-linear-perceptron.py
python3 validation/single-layer-non-linear-perceptron.py
python3 validation/multi-layer-perceptron.py
python3 -m unittest validation/test_validation_scripts.py
```

### Ejercicio 1

```bash
python3 exercise1/train.py
```

Salida esperada:

- exploración inicial del dataset
- chequeos de calidad
- comparación perceptrón lineal vs no lineal
- estudio de generalización
- recomendación de threshold
- gráficos en `exercise1/plots/` si `matplotlib` está instalado
- datos para la visualización principal en `exercise1/visualizations/data/`

### Visualización principal de Ejercicio 1

Para regenerar los datos que consume la visualización:

```bash
python3 exercise1/train.py
```

Para renderizar la animación de Manim:

```bash
bash exercise1/visualizations/manim/render_final_model.sh
```

Salida esperada:

- datos exportados en `exercise1/visualizations/data/final_model_overview.json`
- video renderizado en `exercise1/visualizations/renders/videos/final_model_scene/1080p60/Exercise1FinalModelOverview.mp4`

Para abrir el video generado:

```bash
xdg-open exercise1/visualizations/renders/videos/final_model_scene/1080p60/Exercise1FinalModelOverview.mp4
```

> **Nota:** el script usa la imagen Docker `manimcommunity/manim:stable`, así que no hace falta instalar Manim localmente.

### Visualización principal de Ejercicio 3

Para regenerar los datos que consume la visualización:

```bash
python3 exercise3/visualization_data.py
```

Para renderizar la animación de Manim (requiere Docker):

```bash
bash exercise3/visualizations/manim/render_nn_overview.sh
```

Salida esperada:

- datos exportados en `exercise3/visualizations/data/nn_overview.json`
- video renderizado en `exercise3/visualizations/renders/videos/nn_overview_scene/1080p60/Exercise3NNOverview.mp4`

Para abrir el video generado:

```bash
xdg-open exercise3/visualizations/renders/videos/nn_overview_scene/1080p60/Exercise3NNOverview.mp4
```

> **Nota:** el script usa la imagen Docker `manimcommunity/manim:stable`, así que no hace falta instalar Manim localmente.

### Ejercicio 2

```bash
python3 exercise2/train.py
```

Salida esperada:

- comparación de variantes de `learning rate`
- comparación de arquitecturas
- comparación de optimizadores (`sgd`, `momentum`, `adaptive_eta`, `rmsprop`, `adam`)
- métricas en entrenamiento, validación y `digits_test.csv`
- modelos guardados en `exercise2/models/`
- resultados en `exercise2/results/`
- gráficos en `exercise2/plots/` si `matplotlib` está instalado

La grilla por defecto está separada por eje de análisis para que las comparaciones sean más defendibles:

- `learning_rate`: mantiene fija la arquitectura base y el optimizador
- `architecture`: mantiene fijo el optimizador y la tasa de aprendizaje
- `optimizer`: mantiene fija la arquitectura base y la tasa de aprendizaje

También se puede correr un solo experimento desde consola, eligiendo optimizador y opcionalmente otros hiperparámetros:

```bash
python3 exercise2/train.py --optimizer momentum
python3 exercise2/train.py --optimizer sgd --learning-rate 0.01 --architecture 784,128,10
python3 exercise2/train.py --optimizer adam --architecture 784,256,128,10 --activation tanh
```

En esas corridas individuales, `digits_test.csv` no se evalúa por defecto para no mezclar ajuste de hiperparámetros con generalización final.

Argumentos disponibles para corridas individuales:

- `--optimizer`: uno de `sgd`, `momentum`, `adaptive_eta`, `rmsprop`, `adam`
- `--learning-rate`: tasa de aprendizaje
- `--architecture`: arquitectura separada por comas, por ejemplo `784,128,10`
- `--activation`: `logistic` o `tanh`
- `--epochs`: cantidad de épocas
- `--batch-size`: tamaño de mini-batch
- `--l2-lambda`: regularización L2
- `--normalization`: `none`, `minmax`, `zscore` o `unit`
- `--name`: nombre explícito para guardar artefactos de esa corrida
- `--evaluate-test`: evalúa en `digits_test.csv` solo si querés medir un modelo final ya elegido

### Análisis opcional de Ejercicio 2

Los opcionales de robustez al ruido e interpretabilidad quedaron separados del entrenamiento, para no mezclar la búsqueda de hiperparámetros con el análisis posterior del mejor modelo:

```bash
python3 exercise2/analysis.py
python3 exercise2/analysis.py --model exercise2/models/mi_modelo.npz
python3 exercise2/analysis.py --model exercise2/models/mi_modelo.npz --noise-levels 0,0.05,0.1,0.2,0.3,0.5
python3 exercise2/analysis.py --model exercise2/models/mi_modelo.npz --attribution-methods gradient_input,integrated_gradients
```

Qué hace este script:

- toma `digits_test.csv` y le agrega ruido gaussiano de distinta intensidad
- mide cómo cae `accuracy` y `F1 macro` al aumentar `sigma`
- guarda gráficos de robustez global y por clase en `exercise2/plots/`
- genera mapas de atribución por dígito usando `Gradient·Input` e `Integrated Gradients`
- visualiza los pesos de la primera capa para inspeccionar qué patrones aprendieron las neuronas ocultas
- guarda un resumen reproducible en `exercise2/results/*_analysis.json`

Argumentos principales:

- `--model`: modelo `.npz` a analizar; si no se pasa, intenta usar el mejor de `summary.json`
- `--noise-levels`: lista de sigmas separada por comas
- `--noise-repeats`: cantidad de repeticiones por sigma para promediar el efecto del ruido
- `--per-class-sigmas`: subconjunto de sigmas a mostrar en el gráfico por clase
- `--attribution-methods`: `gradients`, `gradient_input`, `integrated_gradients`
- `--integrated-steps`: pasos de interpolación para `Integrated Gradients`
- `--max-samples-per-class`: máximo de ejemplos correctamente clasificados usados por clase
- `--skip-noise`: saltea el análisis de robustez
- `--skip-attribution`: saltea los mapas de atribución


### Reanudar entrenamiento de Ejercicio 2

Si ya existe un modelo guardado, se puede continuar el entrenamiento sin empezar desde cero:

```bash
python3 exercise2/train.py --resume-model exercise2/models/archivo.npz --extra-epochs 20
```

Esto:

- carga el modelo y su configuración guardada
- entrena `20` épocas adicionales
- vuelve a evaluar validación y test
- guarda un nuevo modelo, nuevos resultados y nuevos gráficos

Argumentos disponibles para reanudar:

- `--resume-model`: ruta al modelo `.npz` guardado
- `--extra-epochs`: épocas adicionales a entrenar
- `--patience`: override opcional del early stopping durante la reanudación
- `--evaluate-test`: evalúa en `digits_test.csv` después de reanudar si realmente querés medir generalización final

### Ejercicio 3

```bash
python3 exercise3/train.py
```

Salida esperada:

- exploración del dataset combinado (`digits.csv` + `more_digits.csv`)
- distribución de clases y pesos de clase (inverse frequency)
- cinco experimentos comparados: `Baseline`, `Weighted Loss`, `Weighted Sampling`, `Synthetic Balancing`, `SMOTE`
- métricas en entrenamiento, validación y `digits_test.csv` por experimento
- modelos guardados en `exercise3/models/`
- resultados en `exercise3/results/`
- gráficos en `exercise3/plots/` si `matplotlib` está instalado

También se puede correr un solo experimento:

```bash
python3 exercise3/train.py --experiment weighted_sampling
python3 exercise3/train.py --experiment synthetic_balancing
python3 exercise3/train.py --experiment smote
```

La opción `synthetic_balancing` agrega una variante SMOTE-like: genera muestras sintéticas para las clases minoritarias interpolando entre ejemplos del mismo dígito y aplicando una pequeña perturbación.

La opción `smote` agrega una implementación de SMOTE propiamente dicha: para cada muestra minoritaria elige uno de sus `k` vecinos más cercanos de la misma clase y genera una interpolación lineal, sin meter ruido ni shifts extra.

### Análisis opcional de Ejercicio 3

Análoga al análisis del Ejercicio 2, pero evaluando todos los modelos entrenados con `train.py`:

```bash
python3 exercise3/analysis.py
python3 exercise3/analysis.py --noise-levels 0,0.1,0.3,0.5
python3 exercise3/analysis.py --skip-noise
python3 exercise3/analysis.py --skip-attribution
```

Qué hace este script:

- evalúa los modelos del Ejercicio 3 bajo distintos niveles de ruido gaussiano
- genera gráficos de robustez global y por clase en `exercise3/plots/`
- genera mapas de saliencia (∂output/∂pixel) por dígito para cada modelo
- visualiza los pesos de la primera capa de cada modelo
- guarda resúmenes reproducibles en `exercise3/results/*_analysis.json`

Argumentos disponibles:

- `--noise-levels`: lista de sigmas separada por comas (default: `0,0.05,0.1,0.15,0.2,0.3,0.5,0.8,1.0`)
- `--noise-repeats`: repeticiones por sigma para promediar el ruido (default: `3`)
- `--per-class-sigmas`: subconjunto de sigmas para el gráfico por clase
- `--max-samples-per-class`: máximo de muestras correctamente clasificadas por clase para atribución
- `--seed`: semilla aleatoria
- `--skip-noise`: saltea el análisis de robustez al ruido
- `--skip-attribution`: saltea los mapas de saliencia y pesos

### Artefactos generados

Durante las corridas, el proyecto puede generar archivos auxiliares:

- `results/`: sirven para comparar experimentos y conservar métricas.
- `models/`: sirven para reutilizar pesos entrenados sin volver a entrenar desde cero.
- `plots/`: sirven para análisis visual de loss, accuracy o matrices de confusión.
- `visualizations/`: sirven para exportar datos y renderizar animaciones de Manim del modelo final.
