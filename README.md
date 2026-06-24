# FRS — Síntesis de datos SLV2013

Pipeline Python para generar **datos sintéticos** de la encuesta de factores de riesgo Salvador 2013 (`SLV2013_Public_Use.csv`), preservando distribuciones, patrones de missing y coherencia entre variables Q, QN e indicadores derivados.

## Requisitos

- Python 3.11–3.13
- `pip`

## Instalación

```powershell
cd C:\Users\echoe\Desktop\frs

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset de entrada

Coloca el archivo real en `data/input/SLV2013_Public_Use.csv`.

| Grupo | Columnas | Descripción |
|-------|----------|-------------|
| Q1–Q58 | 49 cols | Preguntas originales (edad, sexo, altura, peso, alimentación, etc.) |
| QN6–QN58 | 53 cols | Recodificaciones numéricas de Q6–Q58 |
| Derivadas | 8 cols | `qnowtg`, `qnobeseg`, `qnunwtg`, `qnfrvgg`, `qnpa7g`, `qnpe5g`, `qnc1g`, `qnc2g` |
| Diseño | 3 cols | `weight`, `stratum`, `psu` |

**Missing SPSS/Stata:** el valor `1.79769313486232e+308` (máximo float64) representa nulo. El pipeline lo convierte a `NaN` en memoria y lo restaura al exportar.

## Uso

### 1. Generar perfil de columnas (opcional)

```powershell
python main.py profile --input data/input/SLV2013_Public_Use.csv
```

Escribe [`config/schema.yaml`](config/schema.yaml) con tipos, rangos y % missing por columna.

### 2. Sintetizar datos

```powershell
python main.py synthesize `
  --input data/input/SLV2013_Public_Use.csv `
  --output data/output/synthetic_SLV2013.csv `
  -n 5000 `
  --seed 42 `
  --validate
```


| Flag | Descripción |
|------|-------------|
| `-n` / `--rows` | Número de filas sintéticas (obligatorio) |
| `--input` | CSV fuente (default: `data/input/SLV2013_Public_Use.csv`) |
| `--output` | CSV de salida (default: `data/output/synthetic_{timestamp}.csv`) |
| `--seed` | Semilla para reproducibilidad |
| `--schema` | Ruta al schema YAML |
| `--validate` | Genera reporte JSON/MD de fidelidad |

## Arquitectura

```
CSV real → preproceso (sentinel → NaN)
         → SDV GaussianCopula (Q1–Q58 + weight/stratum/psu)
         → patrones de missing por estrato
         → recodificación Q→QN (determinística)
         → indicadores derivados (IMC + reglas condicionales)
         → export CSV con sentinel SPSS
```

- **No se sintetizan** QN ni derivadas directamente; se calculan desde las Q para mantener coherencia.
- Combinaciones `(stratum, psu)` inválidas se reemplazan por pares observados en el dato real.

## Estructura del proyecto

```
frs/
├── config/schema.yaml      # Metadatos de columnas
├── data/input/             # Datos reales (gitignored)
├── data/output/            # Sintéticos y reportes (gitignored)
├── frs/
│   ├── constants.py        # Sentinel SPSS, grupos de columnas
│   ├── io.py               # Carga/exportación CSV
│   ├── preprocess.py       # Perfilado y missing patterns
│   ├── metadata.py         # Metadatos SDV
│   ├── mappings.py         # Tablas Q→QN
│   ├── derive.py           # Indicadores globales
│   ├── synthesizer.py      # Motor principal
│   └── validate.py         # Métricas de fidelidad
├── main.py                 # CLI
└── requirements.txt
```

## Validación

Con `--validate` se genera un reporte junto al CSV (`*.validation.json` y `*.validation.md`) con:

- Test KS para variables continuas (`Q4`, `Q5`, `weight`)
- Distancia de variación total (TVD) en categóricas
- Diferencia de % missing por columna
- Coherencia Q↔QN e indicadores derivados

> Con pocos registros sintéticos (p. ej. n=100) las métricas marginales pueden variar mucho. Use n ≥ 1.915 para comparaciones más estables.

## Privacidad

Los datos sintéticos reducen exposición de registros reales, pero **no garantizan anonimato**. Evalúe riesgo de re-identificación antes de publicar o compartir.

## Licencia

MIT — ver [LICENSE](LICENSE).
