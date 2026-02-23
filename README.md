# Video Intelligence System

Sistema de análisis y organización automática de videos usando IA.

Procesa colecciones de videos MP4, los clasifica por fecha y tema usando modelos de IA (CLIP, Whisper, InsightFace, EasyOCR), y los organiza automáticamente en una estructura `Año/Mes/Tema`.

## Características

- **Clasificación por tema**: Usa CLIP para categorizar videos (Comedia, Tutoriales, Vlogs, etc.)
- **Detección facial**: Detecta y agrupa rostros con InsightFace + DBSCAN
- **Transcripción de audio**: Whisper para transcripción automática
- **OCR**: Extrae texto visible en los frames
- **Extracción de fecha**: EXIF, nombre de archivo, o fecha de modificación
- **Detección de duplicados**: Hash MD5 para evitar reprocesar
- **Organización automática**: Mueve videos a `Año/Mes/Tema/`

## Requisitos

- Python 3.11+
- NVIDIA GPU con CUDA (recomendado, GTX 1080 Ti o superior)
- FFmpeg 6.x
- ~10GB espacio para modelos de IA

## Instalación

```bash
git clone <repo-url>
cd video_intelligence

# Instalación automática
bash install.sh

# O manual:
python -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Configuración

Editar `config.yaml` para ajustar:

```yaml
paths:
  input: /mnt/video_hub/00_ENTRADA      # Carpeta con videos nuevos
  output: /mnt/video_hub/01_PROCESADOS   # Videos organizados
  analysis: /mnt/video_hub/_ANALYSIS     # Base de datos y reportes
```

## Uso

```bash
# Activar entorno virtual
source venv/bin/activate

# Procesar todos los videos en la carpeta de entrada
python main.py

# Con carpeta de entrada personalizada
python main.py --input /ruta/a/videos

# Con archivo de configuración personalizado
python main.py --config mi_config.yaml
```

### Flujo de trabajo típico

```bash
# 1. Copiar videos a la carpeta de entrada
cp ~/videos_nuevos/*.mp4 /mnt/video_hub/00_ENTRADA/

# 2. Ejecutar análisis
python main.py

# 3. Revisar resultados organizados
ls /mnt/video_hub/01_PROCESADOS/2024/Marzo/Comedia/

# 4. Subir a iCloud (manual)
# 5. Mover subidos a archivo
mv /mnt/video_hub/01_PROCESADOS/2024/ /mnt/video_hub/_SUBIDO_A_NUBE/2024/
```

## Estructura de salida

```
01_PROCESADOS/
├── 2023/
│   ├── Enero/
│   │   ├── Comedia/
│   │   ├── Tutoriales/
│   │   └── Vlogs/
│   └── Febrero/
├── 2024/
└── Sin_Fecha/
    └── Otros/
```

## Temas disponibles

| Tema | Descripción |
|------|-------------|
| Comedia | Sketches cómicos, humor |
| Tutoriales | Videos educativos, how-to |
| Sketches | Cortometrajes, actuación |
| Vlogs | Diarios personales |
| Entrevistas | Conversaciones, Q&A |
| Outdoor | Escenas exteriores |
| Producto_Review | Reviews, unboxing |
| Talking_Head | Persona hablando a cámara |
| Otros | No clasificado (requiere revisión) |

## Pipeline de procesamiento

1. **Metadata**: Extrae duración, resolución, codecs (FFprobe)
2. **Frames**: Samplea 30 frames uniformemente distribuidos
3. **CLIP**: Categoriza visualmente contra prompts de temas
4. **Faces**: Detecta rostros y genera embeddings (InsightFace)
5. **Whisper**: Transcribe audio si detecta diálogo
6. **OCR**: Extrae texto visible (EasyOCR)
7. **Fecha**: Determina fecha de creación
8. **Organización**: Mueve a `Año/Mes/Tema/`
9. **Reporte**: Genera resumen del lote

## Base de datos

SQLite en `_ANALYSIS/database/analysis.db` con:
- Metadata de cada video
- Resultados de detección facial
- Transcripciones
- Textos OCR
- Clasificaciones temáticas
- Historial de procesamiento

## Rendimiento esperado

| Videos | Tiempo estimado |
|--------|----------------|
| 10     | ~20-25 min     |
| 50     | ~1.5-2 horas   |
| 100    | ~3-4 horas     |

(Con GPU NVIDIA GTX 1080 Ti)
