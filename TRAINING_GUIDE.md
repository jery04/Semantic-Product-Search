# 🧬 Guía de Entrenamiento con Reanudación de Checkpoints

## Problema Resuelto

El entrenamiento anterior se interrumpía (Ctrl+C) durante la evaluación y era necesario empezar desde cero. Ahora puedes **reanudar desde el último checkpoint guardado** sin perder el progreso.

---

## Cómo Usar

### 1️⃣ **Iniciar Entrenamiento Nuevo**

```bash
# Opción A: Con Taskfile
task train

# Opción B: Directamente
python scripts/train_weights.py
```

Durante el entrenamiento, el mejor estado se guarda automáticamente en `JSON/best_weights_checkpoint.json` cada vez que hay una mejora en fitness.

---

### 2️⃣ **Reanudar desde Checkpoint**

Si el entrenamiento se interrumpe, simplemente ejecuta:

```bash
# Opción A: Con Taskfile (recomendado)
task train-resume

# Opción B: Directamente
python scripts/train_weights.py --resume
```

El script:
- Carga el checkpoint del entrenamiento anterior
- **Continúa exactamente desde la siguiente generación**
- Mantiene el mejor fitness y la población anterior
- Permite llegar a 40 generaciones sin empezar de cero

---

### 3️⃣ **Ver Estado del Checkpoint**

Para verificar en qué generación te encuentras y qué fitness has alcanzado:

```bash
# Opción A: Con Taskfile
task checkpoint-status

# Opción B: Directamente
python scripts/checkpoint_status.py
```

Output esperado:
```
============================================================
📊 ESTADO DEL CHECKPOINT DE ENTRENAMIENTO
============================================================

✓ Generación alcanzada: 15
✓ Mejor fitness: 0.7234
✓ Fitness promedio: 0.6891
✓ Fecha del checkpoint: 2026-09-23 14:35:22

📈 Pesos optimizados:
Componente                           Peso
------------------------------------------
  title_similarity                   0.180000
  description_similarity             0.120000
  title_match                        0.520000
  description_match                  0.080000
  title_ngram_match                  0.050000
  description_ngram_match            0.050000
------------------------------------------
  Total                              1.000000

⏳ Generaciones restantes para completar 40: 25

💡 Comandos útiles:
   • Reanudar entrenamiento: python scripts/train_weights.py --resume
   • Ver estado nuevamente:  python scripts/checkpoint_status.py
```

---

## Opciones de Línea de Comando

| Opción | Descripción |
|--------|-------------|
| `--resume` | Reanuda desde el último checkpoint (si existe) |
| `--generations N` | Establece el número máximo de generaciones (default: 40) |
| `--quiet` | Reduce la salida de consola |

**Ejemplos:**
```bash
# Reanudar y cambiar a 50 generaciones totales
python scripts/train_weights.py --resume --generations 50

# Entrenar 100 generaciones desde cero
python scripts/train_weights.py --generations 100

# Entrenar 40 gen. sin output detallado
python scripts/train_weights.py --quiet
```

---

## Flujo Recomendado para Alcanzar Generación 40

### Escenario 1: Pequeñas Interrupciones
```bash
# Sesión 1: Inicia entrenamiento
task train
# [Se interrumpe en generación 15]

# Sesión 2: Continúa desde donde se quedó
task train-resume
# [Continúa de generación 16 a 40]
```

### Escenario 2: Verificacio Frecuente
```bash
# Verificar estado actual
task checkpoint-status

# Si es necesario, reanudar
task train-resume
```

### Escenario 3: Ajustar Meta de Generaciones
```bash
# Entrenar hasta gen 50 en lugar de 40
python scripts/train_weights.py --resume --generations 50
```

---

## Detalles Técnicos

### Checkpoint (`JSON/best_weights_checkpoint.json`)

Se guarda automáticamente con:
- **weights**: Los 6 pesos optimizados de la mejor solución
- **best**: El fitness de la mejor solución
- **mean**: El fitness promedio de esa generación
- **generations**: Número de generaciones completadas
- **date**: Timestamp del último guardado

### Reanudación

Cuando ejecutas `--resume`:
1. Carga los pesos y fitness del checkpoint
2. Inicia la población desde generación `N+1`
3. Mantiene el mejor genoma conocido
4. Continúa el algoritmo genético como si nunca se hubiera interrumpido

### Automáticamente Guardado

El checkpoint se actualiza automáticamente cuando:
- ✅ Se encuentra una solución mejor (fitness > anterior + tolerancia)
- ✅ Se interrumpe con Ctrl+C (se muestra en consola)

---

## Solución de Problemas

### ❓ "No se encontró checkpoint para reanudar"
- **Causa**: Aún no existe `JSON/best_weights_checkpoint.json`
- **Solución**: Ejecuta `task train` primero para iniciar

### ❓ "¿Me quedé en la generación 15 y necesito llegar a 40?"
```bash
# Simplemente reanuda, se completarán las 25 restantes
task train-resume
```

### ❓ Quiero comenzar un entrenamiento nuevo ignorando el checkpoint anterior
```bash
# Elimina el checkpoint manualmente
rm JSON/best_weights_checkpoint.json
# Luego inicia
task train
```

---

## Ejemplo Completo

```bash
# Día 1: Inicias entrenamiento
$ task train
Cargando modelo spaCy...
Cargando catálogo de productos...
Iniciando evolución...
gen=01: best=0.5234 ; mean=0.4891
gen=05: best=0.6123 ; mean=0.5812
gen=10: best=0.6745 ; mean=0.6234
[Ctrl+C - Se interrumpe]
Interrupción detectada: el mejor estado ha sido guardado en el checkpoint.
Puedes reanudar el entrenamiento con: python scripts/train_weights.py --resume
Checkpoint actual: generación 10, fitness=0.6745

# Día 2: Verificas estado
$ task checkpoint-status
Generación alcanzada: 10
Mejor fitness: 0.6745
Generaciones restantes: 30

# Día 2: Continúas entrenamiento
$ task train-resume
Cargando modelo spaCy...
✓ Checkpoint cargado (fecha: 2026-09-23 14:35:22, generación: 10)
Mejor fitness anterior: 0.6745

Iniciando evolución...
Reanudando desde generación 11...
gen=11: best=0.6789 ; mean=0.6321
gen=15: best=0.7012 ; mean=0.6545
...
[Continúa hasta generación 40]
```

---

## Notas Importantes

⚠️ **Aviso**: 
- No elimines `JSON/best_weights_checkpoint.json` mientras estés entrenando
- El archivo se sobrescribe con cada checkpoint, guardando solo el mejor estado actual
- Si cambias `SEED` en el código, la reanudación podría dar resultados inconsistentes

✨ **Ventaja Clave**: 
Ahora puedes detener el entrenamiento sin miedo, verificar el checkpoint, y continuar cuando lo desees. ¡Perfecta para entrenamientos largos en máquinas con recursos limitados!

