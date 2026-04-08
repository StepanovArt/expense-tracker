"""Configuración del bot cargada desde variables de entorno."""
import os
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
        self.google_credentials_path = self._get_required_env('GOOGLE_CREDENTIALS_PATH')
        self.spreadsheet_id = self._get_required_env('SPREADSHEET_ID')
        self.sheet_name = os.getenv('SHEET_NAME', 'Gastos')

        # Parsear categorías como lista
        categories_str = os.getenv(
            'EXPENSE_CATEGORIES',
            'Supermercado,Salidas,Combustible,Mascotas,Regalos,Delivery,Servicios,Compras,Juntadas,Salud,Deporte,Vianda'
        )
        self.expense_categories = [cat.strip() for cat in categories_str.split(',')]

        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.log_dir = os.getenv('LOG_DIR', 'logs')

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
