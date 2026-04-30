# TP3 - Perceptron Simple y Multicapa


## Estructura

- `validation/`: validaciones de las herramientas pedidas en el TP.
- `exercise1/`: Ejercicio 1, distillation / fraude.
- `exercise2/`: Ejercicio 2, clasificación de dígitos con perceptrón multicapa.


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

### Artefactos generados

Durante las corridas, el proyecto puede generar archivos auxiliares:

- `results/`: sirven para comparar experimentos y conservar métricas.
- `models/`: sirven para reutilizar pesos entrenados sin volver a entrenar desde cero.
- `plots/`: sirven para análisis visual de loss, accuracy o matrices de confusión.
