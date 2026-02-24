# Video Intelligence System

Sistema de análisis y organización automática de videos usando IA. Procesa colecciones de videos MP4, los clasifica por fecha y tema usando modelos de IA, y los organiza en una estructura de carpetas `Año/Mes/Tema`.

## Características

- **Detección de duplicados** por hash MD5 - nunca reprocesa un video
- **Extracción de metadata** con FFprobe (duración, resolución, codecs, aspect ratio)
- **Clasificación de formato** automática (reel/short/long, vertical/horizontal)
- **Categorización visual** con CLIP (8 categorías temáticas)
- **Detección facial** con InsightFace + clustering automático con DBSCAN
- **Transcripción de audio** con Whisper (detección de idioma)
- **OCR** con EasyOCR (español + inglés)
- **Extracción de fechas** desde EXIF, nombre de archivo, o fecha del sistema
- **Organización automática** en estructura `Año/Mes/Tema/`
- **Manejo de duplicados** configurable (skip o mover a carpeta)
- **Base de datos SQLite** con FTS5 full-text search, views y triggers
- **Reportes y CSV** por lote de procesamiento
- **Backup automático** de base de datos antes de cada ejecución

## Requisitos

- Ubuntu 24.04 LTS (o similar)
- Python 3.11+
- NVIDIA GPU con CUDA 12.x (recomendado, funciona sin GPU pero más lento)
- FFmpeg 6.x
- 8GB+ RAM (16GB+ recomendado)

## Instalación

```bash
# Clonar o copiar el proyecto
cd /home/claude/video_intelligence

# Ejecutar instalador
chmod +x install.sh
./install.sh

# Activar entorno virtual
source venv/bin/activate
```

## Configuración

Editar `config.yaml` según tu setup:

```yaml
paths:
  input: /mnt/video_hub/00_ENTRADA      # Carpeta de entrada
  output: /mnt/video_hub/01_PROCESADOS   # Carpeta de salida organizada
  analysis: /mnt/video_hub/_ANALYSIS     # Datos de análisis
```

### Parámetros importantes

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `analysis.frames.count_per_video` | 30 | Frames a samplear por video |
| `processing.gpu_batch_size` | 16 | Tamaño de batch para GPU |
| `models.whisper.model_name` | medium | Modelo Whisper (tiny/base/small/medium/large) |
| `models.clip.model_name` | ViT-B/32 | Modelo CLIP para categorización |
| `analysis.themes.confidence_threshold` | 0.6 | Umbral mínimo para asignar tema |
| `analysis.face_clustering.eps` | 0.6 | Umbral de similitud facial (DBSCAN) |
| `organization.operation` | move | Operación: move o copy |
| `duplicates.action` | move_to_folder | Qué hacer con duplicados |
| `error_handling.max_errors_per_batch` | 10 | Máximo de errores antes de abortar |

## Uso

### Procesamiento básico

```bash
# 1. Colocar videos en carpeta de entrada
cp ~/nuevos_videos/*.mp4 /mnt/video_hub/00_ENTRADA/

# 2. Ejecutar análisis
python main.py

# 3. Revisar resultados organizados
ls /mnt/video_hub/01_PROCESADOS/
```

### Opciones de línea de comando

```bash
# Procesar con configuración custom
python main.py --config mi_config.yaml

# Vista previa sin mover archivos
python main.py --dry-run

# Regenerar reporte de un lote anterior
python main.py --report-only abc12345
```

### Ejemplo de salida

```
============================================================
  VIDEO INTELLIGENCE - Batch Report
  Batch ID: a1b2c3d4
  Generated: 2024-01-15 14:30:00
============================================================

SUMMARY
----------------------------------------
  Total videos found:    50
  Successfully processed:47
  Skipped (duplicates):  2
  Duplicates moved:      2
  Errors:                1
  Needs manual review:   3
  Processing time:       45m 23s

THEME DISTRIBUTION
----------------------------------------
  Comedia                    15
  Talking_Head               12
  Tutoriales                  8
  Sketches                    5
  Vlogs                       4
  Otros                       3

DATE DISTRIBUTION
----------------------------------------
  2023/Marzo               8
  2023/Abril              12
  2024/Enero               6
  Sin Fecha                 5
```

## Estructura de Carpetas

### Entrada
```
/mnt/video_hub/00_ENTRADA/
└── [videos.mp4 a procesar]
```

### Salida organizada
```
/mnt/video_hub/01_PROCESADOS/
├── 2023/
│   ├── Enero/
│   │   ├── Comedia/
│   │   ├── Tutoriales/
│   │   └── Talking_Head/
│   ├── Febrero/
│   └── ...
├── 2024/
├── Sin_Fecha/
│   └── Otros/
└── _ESPECIALES/
    ├── Duplicados/
    ├── Videos_Largos_3min+/
    └── Revisar_Manual/
```

### Datos de análisis
```
/mnt/video_hub/_ANALYSIS/
├── database/
│   ├── analysis.db          # Base de datos (NO borrar)
│   └── backups/             # Backups automáticos
├── thumbnails/
│   ├── videos/              # Thumbnails de videos
│   └── faces/               # Thumbnails de clusters faciales
├── reports/
│   └── lote_a1b2c3d4_2024-01-15.txt
├── exports/
│   └── batch_a1b2c3d4.csv
└── logs/
    └── 2024-01-15.log
```

## Categorías Temáticas

| Tema | Descripción |
|------|-------------|
| Comedia | Sketches de humor, videos cómicos |
| Tutoriales | Videos educativos, how-to |
| Sketches | Cortometrajes, escenas dramáticas |
| Vlogs | Diarios personales, behind the scenes |
| Entrevistas | Conversaciones, Q&A |
| Outdoor | Escenas exteriores, naturaleza |
| Producto_Review | Reviews, unboxing |
| Talking_Head | Persona hablando a cámara |
| Otros | No clasificado (requiere revisión manual) |

## Base de Datos

El esquema SQLite incluye:

| Tabla | Descripción |
|-------|-------------|
| `videos` | Metadata principal + clasificación + fechas |
| `faces` | Detecciones faciales con embeddings |
| `person_clusters` | Clusters de personas (agrupación facial) |
| `video_persons` | Relación many-to-many videos ↔ personas |
| `transcriptions` | Transcripciones de audio (Whisper) |
| `ocr_texts` | Texto extraído de frames (OCR) |
| `visual_tags` | Tags CLIP por tema con confianza |
| `processing_log` | Historial de lotes procesados |

### Búsqueda Full-Text (FTS5)

```sql
-- Buscar en transcripciones
SELECT * FROM transcriptions_fts WHERE transcriptions_fts MATCH 'tutorial';

-- Buscar en textos OCR
SELECT * FROM ocr_texts_fts WHERE ocr_texts_fts MATCH 'subscribe';
```

### Vistas útiles

```sql
-- Videos con toda la información
SELECT * FROM videos_full;

-- Estadísticas de clusters de personas
SELECT * FROM person_clusters_stats;

-- Lotes recientes
SELECT * FROM recent_batches;
```

## Procesamiento Incremental

El sistema detecta automáticamente videos ya procesados:

1. Calcula hash MD5 de cada video
2. Busca el hash en la base de datos
3. Si existe → lo salta (o mueve a carpeta de duplicados según config)
4. Si es nuevo → lo procesa completo

Esto permite re-ejecutar el script sin preocuparse por duplicados.

## Manejo de Errores

- Errores individuales no detienen el lote completo (configurable)
- Videos con errores se marcan como `needs_review = true` en la BD
- Videos fallidos se mueven a `_ESPECIALES/Revisar_Manual/` (configurable)
- `max_errors_per_batch` limita errores antes de abortar
- Videos con baja confianza temática se marcan para revisión manual
- Log detallado en `_ANALYSIS/logs/`

## Flujo de Trabajo Completo

```bash
# 1. Copiar nuevos videos al SSD
cp ~/Downloads/nuevos/*.mp4 /mnt/video_hub/00_ENTRADA/

# 2. Procesar
cd /home/claude/video_intelligence
source venv/bin/activate
python main.py

# 3. Revisar resultados
cat /mnt/video_hub/_ANALYSIS/reports/lote_*.txt

# 4. Subir a iCloud (manual)
# ...

# 5. Mover videos subidos al archivo
mv /mnt/video_hub/01_PROCESADOS/2023/ /mnt/video_hub/_SUBIDO_A_NUBE/

# 6. Cuando necesites espacio, borrar del archivo
# (la base de datos y thumbnails siempre se conservan)
```

## Rendimiento Estimado

| Videos | Tiempo estimado |
|--------|----------------|
| 10     | ~20-25 min     |
| 50     | ~1.5-2 hours   |
| 100    | ~2.5-3.5 hours |

*Con GPU NVIDIA GTX 1080 Ti. Sin GPU puede ser 3-5x más lento.*

## Solución de Problemas

**GPU no detectada:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
# Si False, verificar drivers NVIDIA y CUDA
nvidia-smi
```

**FFmpeg no encontrado:**
```bash
sudo apt install ffmpeg
```

**Error de memoria GPU:**
Reducir `performance.batch.frame_batch_size` en config.yaml (e.g., de 32 a 16).

**Videos corruptos:**
Se registran con `error_message` en la BD y se mueven a `_ESPECIALES/Revisar_Manual/`.

**Demasiados errores:**
Ajustar `error_handling.max_errors_per_batch` en config.yaml.
