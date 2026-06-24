# FRS — Síntesis de datos

Proyecto Python para **síntesis de datos**. Por ahora solo contiene la estructura base; el dominio y la lógica de generación están pendientes de definir.

## Requisitos

- Python 3.11 o superior
- `pip`

## Instalación

```powershell
cd frs

# Crear entorno virtual
python -m venv .venv

# Activar (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Instalar dependencias (cuando se definan en requirements.txt)
pip install -r requirements.txt
```

En **CMD**:

```cmd
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

## Estructura del proyecto

```
frs/
├── config/
│   └── schema.yaml       # Esquema de datos (vacío, por definir)
├── data/
│   └── output/           # Salida de datos generados
├── frs/
│   ├── __init__.py
│   └── synthesizer.py    # Motor de síntesis (por implementar)
├── main.py               # Punto de entrada CLI (por implementar)
├── requirements.txt
└── README.md
```

## Próximos pasos

1. Definir qué datos se van a sintetizar.
2. Completar `config/schema.yaml` con el esquema.
3. Añadir dependencias en `requirements.txt`.
4. Implementar `frs/synthesizer.py` y `main.py`.

## Licencia

MIT — ver [LICENSE](LICENSE).
