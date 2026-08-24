"""Configuración del bot cargada desde variables de entorno."""
import os
import json
from dotenv import load_dotenv


class Config:
    """Clase de configuración que carga y valida variables de entorno."""

    def __init__(self):
        """Inicializa la configuración cargando el archivo .env"""
        # Cargar .env de la raíz del proyecto
        load_dotenv()

        # Validar y cargar variables requeridas
        self.telegram_bot_token = self._get_required_env('TELEGRAM_BOT_TOKEN')

        # Conector LLM
        self.llm_connector = os.getenv('LLM_CONNECTOR', 'ollama').lower()

        # Ollama
        self.ollama_model = os.getenv('OLLAMA_MODEL', 'qwen3:1.7b')
        self.ollama_timeout = int(os.getenv('OLLAMA_TIMEOUT', '45'))

        # Gemini
        self.gemini_api_key = os.getenv('GEMINI_API_KEY', '')
        self.gemini_model = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')

        # Credenciales Google: JSON inline (Railway) o ruta a archivo (local/Pi)
        self.google_credentials_json = os.getenv('GOOGLE_CREDENTIALS_JSON', '')
        self.google_credentials_path = os.getenv('GOOGLE_CREDENTIALS_PATH', '')
        if not self.google_credentials_json and not self.google_credentials_path:
            raise ValueError(
                "Se requiere GOOGLE_CREDENTIALS_JSON (contenido del JSON) "
                "o GOOGLE_CREDENTIALS_PATH (ruta al archivo)"
            )

        self.spreadsheet_id = self._get_required_env('SPREADSHEET_ID')
        self.sheet_name = os.getenv('SHEET_NAME', 'Gastos')

        # Parsear categorías como lista
        categories_str = os.getenv(
            'EXPENSE_CATEGORIES',
            'Supermercado,Salidas,Combustible,Mascotas,Regalos,Delivery,Servicios,Compras,Juntadas,Salud,Deporte,Vianda'
        )
        self.expense_categories = [cat.strip() for cat in categories_str.split(',')]

        # Descripciones de categorías para guiar al LLM (JSON: {"Categoria": "ejemplos..."})
        category_descriptions_str = os.getenv('EXPENSE_CATEGORY_DESCRIPTIONS', '')
        if category_descriptions_str:
            try:
                self.expense_category_descriptions: dict = json.loads(category_descriptions_str)
            except json.JSONDecodeError:
                self.expense_category_descriptions = {}
        else:
            self.expense_category_descriptions = {}

        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.log_dir = os.getenv('LOG_DIR', 'logs')

        # Control de acceso — fail closed: sin ALLOWED_USER_ID el bot no arranca
        allowed_user_id_str = os.getenv('ALLOWED_USER_ID', '').strip()
        if not allowed_user_id_str:
            raise ValueError(
                "ALLOWED_USER_ID es requerido. El bot no iniciará sin control de acceso. "
                "Configúralo en .env con tu Telegram user ID (obtenlo con @userinfobot)."
            )
        try:
            self.allowed_user_id = int(allowed_user_id_str)
        except ValueError:
            raise ValueError(f"ALLOWED_USER_ID debe ser un número entero, recibido: {allowed_user_id_str!r}")

    def _get_required_env(self, key: str) -> str:
        """
        Obtiene una variable de entorno requerida.

        Args:
            key: Nombre de la variable de entorno

        Returns:
            Valor de la variable de entorno

        Raises:
            ValueError: Si la variable no está definida
        """
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Variable de entorno {key} es requerida")
        return value
