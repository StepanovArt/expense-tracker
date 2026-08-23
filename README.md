# Bot de Telegram para Registro de Gastos

Bot de Telegram que procesa mensajes de texto y audio en lenguaje natural sobre gastos y los registra automáticamente en Google Sheets usando un LLM (Ollama local o Gemini).

## 🎯 Objetivo

Permitir el registro automático de gastos personales (propios y de pareja) a partir de mensajes de texto o voz enviados por Telegram, cuando no se tiene acceso simple o inmediato a la planilla utilizada para llevar el control financiero.

El foco es reducir fricción: registrar el gasto en el momento en que ocurre, usando lenguaje natural.

## 🌟 Características

- 🤖 **Procesa lenguaje natural**: Envía mensajes como "Compré pan por $500 ayer" y el bot entiende
- 📦 **Múltiples gastos en un mensaje**: "productos 80 y taxi 25" se registra como dos filas separadas; si una es inválida, las demás se guardan igual (partial success)
- 🎙️ **Soporte de audio**: Envía notas de voz o audios; el bot los transcribe automáticamente con Whisper (modelo open-source)
- 📊 **Registro automático**: Guarda directamente en Google Sheets con descripción generada por el LLM
- 🧠 **LLM configurable**: Ollama local (privado, sin costos de API) o Gemini (API de Google)
- 🔐 **Acceso restringido**: Solo el usuario autorizado puede usar el bot (configurado vía `ALLOWED_USER_ID`); sin esta variable el bot no arranca
- ⚙️ **Configurable**: Todo parametrizable vía variables de entorno
- 🍓 **Optimizado para Raspberry Pi**: Bajo consumo de recursos


## 🧠 ¿Por qué usar un LLM?

El sistema utiliza un LLM para interpretar mensajes en lenguaje natural. Soporta dos conectores: Ollama (local) y Gemini (API de Google), seleccionables via variable de entorno.

Ventajas frente a soluciones tradicionales (regex, comandos estructurados o botones):

- Permite entrada libre: "pagué 23 mil en el súper", "compré nafta 45k", "almuerzo con Juan 12.500"
- Categorización automática según un set predefinido de categorías
- Soporte natural para variaciones de formato, moneda y redacción
- Posibilidad de extender a voz sin modificar el modelo de interacción

El LLM actúa únicamente como capa de interpretación. La lógica de negocio y la persistencia siguen siendo determinísticas.

### 🔒 Opciones de conector LLM

El sistema soporta dos conectores, seleccionables mediante la variable `LLM_CONNECTOR`:

| Conector | Ventajas | Requisitos |
|----------|----------|------------|
| `ollama` | Privado, sin costos de API, sin dependencia externa | Ollama instalado con modelo descargado |
| `gemini` | Sin infraestructura local, modelos más capaces | API key de Google Gemini |

Para entornos como Raspberry Pi, Ollama con un modelo liviano (ej. `qwen3:1.7b`) mantiene tiempos de respuesta razonables con bajo consumo de recursos.

### 🏷️ Categorización automática

Uno de los principales beneficios del uso de LLM es la asignación automática de categorías de gasto.

El modelo recibe:

- El mensaje original
- El listado explícito de categorías válidas

Y debe devolver un **array JSON** con uno o varios objetos, cada uno con:

- Monto
- Descripción
- Categoría (forzada a una del listado)
- Fecha (si está implícita)

Si un mensaje contiene varios gastos, el LLM los devuelve como múltiples elementos del array y se registran en una sola llamada a la API de Sheets. Esto permite mantener consistencia en la planilla sin exigirle al usuario recordar comandos o categorías exactas.

## ⚠️ Limitaciones

- La precisión depende del modelo y del prompt.
- No existe interfaz de corrección automática ante errores del modelo.

## 🏗️ Arquitectura

### Diagrama de Contenedores (C2)

Vista de alto nivel de los componentes del sistema y cómo se comunican entre sí:

```mermaid
flowchart LR
    User(["👤 Usuario"])
    Telegram(["📱 Telegram"])
    Sheets(["📊 Google Sheets"])

    Gemini(["✨ Gemini API"])

    subgraph Sistema ["telegram-bot-gastos-llm"]
        direction TB
        Bot["🤖 Bot<br/>python-telegram-bot"]
        Whisper["🎙️ Whisper<br/>transcripción local"]
        LLM["🧠 LLMConnector<br/>ollama | gemini"]
        OllamaLocal["🖥️ Ollama<br/>Qwen3:1.7b"]
    end

    User -->|"Envía texto nota de voz"| Telegram
    Bot -->|"Polling"| Telegram
    Bot -->|"Procesamiento de audio"| Whisper
    Bot -->|"Transformación de texto a JSON con datos del gasto"| LLM
    LLM -->|"ollama"| OllamaLocal
    LLM -->|"gemini"| Gemini
    Bot -->|"Registro de gasto"| Sheets
```

### Diagrama de Secuencia

Flujo detallado de inicialización y ciclo de vida de un mensaje:

```mermaid
sequenceDiagram
    actor User as 👤 Usuario
    participant TG as Telegram API
    participant Main as main.py
    participant Handler as telegram_handler.py
    participant Whisper as openai-whisper
    participant Prompt as prompt_builder.py
    participant LLMFactory as factory.py
    participant LLMConn as LLMConnector<br/>(ollama/gemini)
    participant Val as validators.py
    participant Sheets as sheets_client.py
    participant GS as Google Sheets

    Note over Main: Arranque
    Main->>Main: Carga Config (.env)
    Main->>LLMFactory: create_llm_connector(config)
    LLMFactory-->>Main: LLMConnector (OllamaClient | GeminiClient)
    Main->>Sheets: Inicializa SheetsClient
    Main->>Whisper: whisper.load_model("small")
    Main->>TG: Registra handlers · start polling

    Note over User,GS: Ciclo de vida de un mensaje
    User->>TG: Texto o nota de voz
    TG->>Handler: handle_message(Update)

    alt Audio / Nota de voz
        Handler->>Whisper: transcribe(audio, language="es")
        Whisper-->>Handler: texto transcrito
    end

    Handler->>Prompt: build_prompt(texto, categorías)
    Prompt-->>Handler: prompt con fecha actual
    Handler->>LLMConn: generate(prompt)
    LLMConn-->>Handler: expense_data (JSON con monto, categoría, fecha, descripción)

    Handler->>Val: validate_expense_data(expense_data, categorías)

    alt Datos válidos
        Handler->>Sheets: append_expense(fecha, descripcion, categoria, monto)
        Sheets->>GS: Sheets API
        GS-->>Sheets: OK
        Handler->>TG: ✅ Confirmación con resumen
    else Datos inválidos o error
        Handler->>TG: ❌ Mensaje de error con sugerencia
    end

    TG-->>User: Respuesta final
```

## 📋 Requisitos Previos

1. **Python 3.9+**
2. **LLM** (elegir uno):
   - **Ollama** instalado con modelo `qwen3:1.7b` (si `LLM_CONNECTOR=ollama`)
   - **Gemini API key** de Google (si `LLM_CONNECTOR=gemini`)
3. **Google Cloud** cuenta con Sheets API habilitada
4. **Bot de Telegram** creado via @BotFather
5. **ffmpeg** instalado en el sistema (requerido por Whisper para procesar audio)
6. **uv** package manager (opcional pero recomendado)

## 🚀 Instalación

### 1. Clonar/Copiar el Proyecto

Clonar repositorio

### 2. Instalar Dependencias

#### Usando uv (recomendado)

```bash
cd telegram-bot-gastos-llm

# Instalar uv si no lo tienes
curl -LsSf https://astral.sh/uv/install.sh | sh

# Crear virtual environment
uv venv

# Activar venv
source .venv/bin/activate

# Instalar dependencias
uv pip install -e .
```

#### Usando pip

```bash
cd telegram-bot-gastos-llm
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 3. Configurar el conector LLM

#### Opción A: Ollama (local)

```bash
# Verificar que Ollama está corriendo
ollama list

# Si no está el modelo, descargarlo
ollama pull qwen3:1.7b

# Probar el modelo
ollama run qwen3:1.7b "Hola"
```

#### Opción B: Gemini (API)

Obtener una API key en [Google AI Studio](https://aistudio.google.com/) y configurar `GEMINI_API_KEY` en el `.env`.

## ⚙️ Configuración

### 1. Crear Bot de Telegram

1. Abre Telegram y busca **@BotFather**
2. Envía `/newbot`
3. Sigue las instrucciones (elige nombre y username)
4. **Copia el token** que te da BotFather

### 2. Configurar Google Sheets API

1. Ve a [Google Cloud Console](https://console.cloud.google.com/)
2. Crea un proyecto nuevo o selecciona uno existente
3. Habilita la **Google Sheets API** y **Google Drive API**
4. Crea una **Service Account**:
   - Ve a "IAM & Admin" → "Service Accounts"
   - Crea nueva service account
   - Descarga el archivo JSON de credenciales
5. **Comparte tu Google Sheet** con el email de la service account (con permisos de editor)
6. Copia el **Spreadsheet ID** de la URL:
   ```
   https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/edit
   ```

### 3. Preparar Google Sheet

Tu hoja debe tener las siguientes columnas (en este orden):

| Fecha | Descripción | Categoría | Monto |
|-------|-------------|-----------|-------|

### 4. Configurar Variables de Entorno

```bash
# Copiar template
cp .env.example .env

# Editar con tus valores
nano .env
```

Completa las siguientes variables:

```bash
# Token del bot de Telegram (de @BotFather)
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz

# Tu Telegram user ID — obtenlo enviando /start a @userinfobot
# Sin esto el bot no arranca (fail closed)
ALLOWED_USER_ID=123456789

# Conector LLM: ollama | gemini
LLM_CONNECTOR=ollama

# --- Ollama (si LLM_CONNECTOR=ollama) ---
OLLAMA_MODEL=qwen3:1.7b
OLLAMA_TIMEOUT=45

# --- Gemini (si LLM_CONNECTOR=gemini) ---
GEMINI_API_KEY=tu-api-key-de-gemini
GEMINI_MODEL=gemini-2.0-flash

# Google Sheets (ajusta las rutas)
GOOGLE_CREDENTIALS_PATH=/ruta/a/tu/service-account.json
SPREADSHEET_ID=tu-spreadsheet-id-aqui
SHEET_NAME=Gastos

# Categorías (personaliza según tus necesidades)
EXPENSE_CATEGORIES=Supermercado,Salidas,Juntadas,Compras

# Logging
LOG_LEVEL=INFO

# Directorio de logs (default: logs/ relativo al directorio de ejecución)
# En producción: /var/log/telegram-bot-gastos-llm
LOG_DIR=logs
```

## 🎯 Uso

### Modo Desarrollo (Consola)

```bash
# Activar venv si no está activado
source .venv/bin/activate

# Ejecutar bot
python run.py
```

### Comandos del Bot

- `/start` - Iniciar bot y ver mensaje de bienvenida
- `/help` - Ver categorías disponibles e instrucciones
- **Mensaje de texto** - Registrar un gasto escribiendo
- **Nota de voz / Audio** - Registrar un gasto por voz (Whisper transcribe automáticamente)

### Ejemplos de Mensajes

**Un gasto:**
```
Compré pan por $500 ayer
Gasté 1200 en suplementos
Salida con amigos, 3500 pesos anteayer
Super hoy 2500
Juntada el lunes pasado, 1800
```

**Varios gastos en un solo mensaje:**
```
productos 80 y taxi 25
super 2500 y delivery 1200 ayer
nafta 45k, juntada 3500 y farmacia 800
```

También podés enviar una nota de voz diciendo lo mismo y el bot la transcribirá antes de procesarla.

El bot entiende:
- ✅ Montos en diferentes formatos
- ✅ Referencias temporales (ayer, anteayer, hoy, etc.)
- ✅ Diferentes formas de expresar gastos
- ✅ Mensajes de texto y notas de voz / audios
- ✅ Uno o múltiples gastos en el mismo mensaje
- ✅ Éxito parcial: si un gasto del mensaje es inválido, los demás se registran igual

## 🐳 Deployment en Raspberry Pi

### Como Servicio systemd

```bash
# 1. Crear directorio de logs
sudo mkdir -p /var/log/telegram-bot-gastos-llm

# 2. Copiar servicio
sudo cp systemd/telegram-bot-gastos-llm.service.template /etc/systemd/system/telegram-bot-gastos-llm.service

# 3. Editar si es necesario (ajustar User, WorkingDirectory)
sudo nano /etc/systemd/system/telegram-bot-gastos-llm.service

# 4. Recargar systemd
sudo systemctl daemon-reload

# 5. Habilitar inicio automático
sudo systemctl enable telegram-bot-gastos-llm

# 6. Arrancar servicio
sudo systemctl start telegram-bot-gastos-llm

# 7. Ver estado
sudo systemctl status telegram-bot-gastos-llm

# 8. Ver logs en tiempo real
sudo journalctl -u telegram-bot-gastos-llm -f
```

### Comandos Útiles

```bash
# Reiniciar servicio
sudo systemctl restart telegram-bot-gastos-llm

# Detener servicio
sudo systemctl stop telegram-bot-gastos-llm

# Ver últimas líneas de log
sudo journalctl -u telegram-bot-gastos-llm -n 50

# Deshabilitar inicio automático
sudo systemctl disable telegram-bot-gastos-llm
```

## 🔧 Troubleshooting

### El bot no responde

1. Verifica logs del bot:
   ```bash
   sudo journalctl -u telegram-bot-gastos-llm -f
   ```

2. Si usas Ollama, verifica que está corriendo:
   ```bash
   ollama list
   ```

3. Verifica que el token de Telegram es correcto

### Error de Google Sheets

1. Verifica que compartiste la hoja con la service account
2. Verifica que el SPREADSHEET_ID es correcto
3. Verifica que la pestaña existe y tiene el nombre correcto
4. Verifica permisos del archivo de credenciales:
   ```bash
   chmod 600 /ruta/a/credentials.json
   ```

### Ollama no responde o es muy lento

> Solo aplica si `LLM_CONNECTOR=ollama`

1. En Raspberry Pi, el modelo puede tardar 10-20 segundos
2. Verifica que tienes suficiente RAM libre:
   ```bash
   free -h
   ```
3. Considera ajustar `OLLAMA_TIMEOUT` en `.env`

### Error con Gemini

> Solo aplica si `LLM_CONNECTOR=gemini`

1. Verifica que `GEMINI_API_KEY` está configurada en `.env`
2. Verifica que la API key tiene permisos para el modelo configurado en `GEMINI_MODEL`
3. Revisa los logs para ver el mensaje de error específico

### Categoría no reconocida

El bot valida que las categorías sean las configuradas en `EXPENSE_CATEGORIES`. Si el LLM devuelve una categoría no válida, el bot lo rechazará. Asegúrate de que tus categorías estén bien escritas en `.env`.

### Error al procesar audio / nota de voz

1. Verifica que `ffmpeg` está instalado:
   ```bash
   ffmpeg -version
   ```
   Si no está, instalarlo:
   ```bash
   sudo apt install ffmpeg
   ```

2. El modelo `small` de Whisper se descarga automáticamente en el primer inicio (~460 MB). Asegúrate de tener conexión a internet y espacio en disco.

3. En Raspberry Pi, la transcripción puede tardar varios segundos dependiendo de la longitud del audio. Si es muy lento, considera usar el modelo `tiny` modificando el código en `src/main.py`:
   ```python
   model_transcribe = whisper.load_model("tiny")
   ```

## 📁 Estructura del Proyecto

```
telegram-bot-gastos-llm/
├── src/
│   ├── main.py                 # Entry point
│   ├── config.py               # Configuración
│   ├── bot/
│   │   └── telegram_handler.py # Handlers de Telegram
│   ├── llm/
│   │   ├── base.py             # Interfaz base LLMConnector
│   │   ├── factory.py          # Factory: crea el conector según LLM_CONNECTOR
│   │   ├── ollama_client.py    # Implementación Ollama
│   │   ├── gemini_client.py    # Implementación Gemini
│   │   └── prompt_builder.py   # Constructor de prompts
│   ├── storage/
│   │   └── sheets_client.py    # Cliente Google Sheets
│   └── utils/
│       ├── logger.py           # Logging
│       ├── validators.py       # Validaciones
│       └── exceptions.py       # Excepciones
├── tests/                      # Tests
├── systemd/                    # Templates systemd
├── .env                        # Configuración (no commitear)
├── .env.example                # Template de configuración
├── pyproject.toml              # Dependencias
└── README.md                   # Esta documentación
```

## 🔒 Seguridad

- ❌ **Nunca commitees** el archivo `.env` o credenciales de Google
- ✅ Usa `.gitignore` para excluirlos automáticamente
- ✅ Permisos 600 para archivos sensibles:
  ```bash
  chmod 600 .env
  chmod 600 /ruta/a/credentials.json
  ```
- ✅ **Control de acceso obligatorio**: configura `ALLOWED_USER_ID` con tu Telegram user ID. El bot rechaza cualquier mensaje de otros usuarios a nivel de filtro (no llega al handler). Sin esta variable el bot no arranca. Tu ID lo encontrás enviando `/start` a [@userinfobot](https://t.me/userinfobot).

## 📝 Licencia

Este proyecto es de código abierto. Úsalo y modifícalo como necesites.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor abre un issue o pull request.
