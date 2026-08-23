"""Cliente para interactuar con Ollama usando la librería oficial."""
import json
import re
import logging
import ollama
from src.llm.base import LLMConnector
from src.utils.exceptions import OllamaConnectionError, OllamaInvalidJSONError


logger = logging.getLogger('telegram-bot-gastos-llm')


class OllamaClient(LLMConnector):
    """Cliente para generar respuestas usando Ollama."""

    def __init__(self, model: str, timeout: int = 45):
        """
        Inicializa el cliente de Ollama.

        Args:
            model: Nombre del modelo (ej: "qwen3:1.7b")
            timeout: Timeout en segundos para la generación
        """
        self.model = model
        self.timeout = timeout
        logger.info(f"Cliente Ollama inicializado con modelo: {model}")

    def generate(self, prompt: str) -> list:
        """
        Genera una respuesta usando Ollama y extrae el JSON.

        Args:
            prompt: Prompt completo para el modelo

        Returns:
            Diccionario con los datos del gasto (monto, categoria, fecha)

        Raises:
            OllamaConnectionError: Si no se puede conectar con Ollama
            OllamaInvalidJSONError: Si no se puede parsear el JSON
        """
        try:
            logger.debug(f"Enviando prompt a Ollama (modelo: {self.model})")

            # Llamar a Ollama usando la librería oficial
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={'timeout': self.timeout}
            )

            # response['response'] contiene el texto generado
            response_text = response['response']
            logger.debug(f"Respuesta de Ollama: {response_text}")

            # Extraer y parsear JSON
            return self._extract_json_from_text(response_text)

        except Exception as e:
            logger.error(f"Error al conectar con Ollama: {e}")
            raise OllamaConnectionError(f"No se pudo conectar con Ollama: {e}")

    def _extract_json_from_text(self, text: str) -> list:
        """
        Extrae un array JSON de un texto, incluso si viene rodeado de texto adicional.
        Si el modelo devuelve un objeto en lugar de un array, lo envuelve en una lista.

        El modelo puede devolver:
        - Array JSON: [{"monto": 100, ...}]
        - Array con múltiples gastos: [{"monto": 100, ...}, {"monto": 50, ...}]
        - Objeto (fallback): {"monto": 100, ...}

        Args:
            text: Texto que contiene JSON

        Returns:
            Lista de diccionarios con los datos de los gastos

        Raises:
            OllamaInvalidJSONError: Si no se encuentra o no se puede parsear el JSON
        """
        array_match = re.search(r'\[.*\]', text, re.DOTALL)
        if array_match:
            try:
                return json.loads(array_match.group())
            except json.JSONDecodeError as e:
                logger.error(f"Error al parsear JSON array: {e}")
                raise OllamaInvalidJSONError(f"JSON inválido: {e}")

        # Fallback: el modelo devolvió un objeto en lugar de array
        obj_match = re.search(r'\{.*\}', text, re.DOTALL)
        if not obj_match:
            logger.error(f"No se encontró JSON en la respuesta: {text}")
            raise OllamaInvalidJSONError("No se encontró JSON en la respuesta del modelo")

        try:
            return [json.loads(obj_match.group())]
        except json.JSONDecodeError as e:
            logger.error(f"Error al parsear JSON: {e}")
            raise OllamaInvalidJSONError(f"JSON inválido: {e}")
