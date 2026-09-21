# 🛒 Buscador semantico de productos

Proyecto en Python que utiliza spaCy y el modelo `es_core_news_md` para buscar productos mediante similitud semantica, coincidencia de terminos y coincidencia de n-gramas. Tambien incluye un script para entrenar los pesos del buscador con un algoritmo genetico.

## 📁 Estructura del proyecto

```text
.
├── JSON/
│   ├── best_weights.json   # Historial de pesos entrenados
│   ├── productos.json       # Catalogo de productos
│   └── queries.json         # Consultas etiquetadas para entrenamiento
├── scripts/
│   ├── index.py             # Busqueda semantica de productos
│   └── train_weights.py     # Entrenamiento y evaluacion de pesos
├── README.md                # Documentacion del proyecto
├── requirements.txt         # Dependencias de Python
├── Taskfile.yml             # Tareas automatizadas
├── .gitignore               # Exclusiones del entorno y archivos temporales
└── .venv/                   # Entorno virtual local, ignorado por Git
```

## ⚙️ Que hace cada script

### 🔎 `scripts/index.py`

Carga el catalogo desde `JSON/productos.json`, procesa los textos con spaCy y ejecuta una consulta de ejemplo (`MicroSD 512 GB`). Muestra los tres productos mejor posicionados junto con su puntuacion y el detalle de los componentes de la puntuacion.

### 🧬 `scripts/train_weights.py`

Carga el catalogo y las consultas etiquetadas, divide las consultas en conjuntos de entrenamiento y prueba, y optimiza los seis pesos del buscador mediante un algoritmo genetico. Guarda cada resultado en `JSON/best_weights.json` y muestra una comparacion entre los pesos originales y los entrenados.

## 📦 Instalacion de dependencias (con Taskfile)

Desde la raiz del proyecto, crea primero el entorno e instala todo con:

```bash
task install
```

Esta tarea crea `.venv`, instala las dependencias de `requirements.txt` usando ese entorno y descarga el modelo de espanol de spaCy dentro del mismo entorno.

## ▶️ Ejecucion

Ejecuta los comandos desde la raiz del proyecto, despues de instalar las dependencias.

### 🔍 Buscador

```bash
task search
```

### 🧠 Entrenamiento de pesos

```bash
task train
```

El entrenamiento actualiza `JSON/best_weights.json` con un nuevo registro de resultados.
