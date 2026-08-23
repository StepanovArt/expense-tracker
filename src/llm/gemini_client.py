"""Cliente para interactuar con Gemini usando la librería oficial de Google."""
import json
import re
import logging
import google.generativeai as genai
from src.llm.base import LLMConnector
from src.utils.exceptions import GeminiConnectionError, GeminiInvalidJSONError


logger = logging.getLogger('telegram-bot-gastos-llm')


class GeminiClient(LLMConnector):
    """Cliente para generar respuestas usando Gemini."""

    def __init__(self, api_key: str, model: str = 'gemini-2.0-flash'):
        """
        Inicializa el cliente de Gemini.

        Args:
            api_key: API key de Google Gemini
            model: Nombre del modelo (ej: "gemini-2.0-flash")
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.model_name = model
        logger.info(f"Cliente Gemini inicializado con modelo: {model}")

    def generate(self, prompt: str) -> list:
        """
        Genera una respuesta usando Gemini y extrae el JSON.

        Args:
            prompt: Prompt completo para el modelo

        Returns:
            Diccionario con los datos del gasto (monto, categoria, fecha)

        Raises:
            GeminiConnectionError: Si no se puede conectar con Gemini
            GeminiInvalidJSONError: Si no se puede parsear el JSON
        """
        try:
            logger.debug(f"Enviando prompt a Gemini (modelo: {self.model_name})")
            response = self.model.generate_content(prompt)
            response_text = response.text
            logger.debug(f"Respuesta de Gemini: {response_text}")
            return self._extract_json_from_text(response_text)

        except Exception as e:
            logger.error(f"Error al conectar con Gemini: {e}")
            raise GeminiConnectionError(f"No se pudo conectar con Gemini: {e}")

    def _extract_json_from_text(self, text: str) -> list:
        """
        Extrae un array JSON de un texto, incluso si viene rodeado de texto adicional.
        Si el modelo devuelve un objeto en lugar de un array, lo envuelve en una lista.

        Args:
            text: Texto que contiene JSON

        Returns:
            Lista de diccionarios con los datos de los gastos

        Raises:
            GeminiInvalidJSONError: Si no se encuentra o no se puede parsear el JSON
        """
        array_match = re.search(r'\[.*\]', text, re.DOTALL)
        if array_match:
            try:
                return json.loads(array_match.group())
            except json.JSONDecodeError as e:
                logger.error(f"Error al parsear JSON array: {e}")
                raise GeminiInvalidJSONError(f"JSON inválido: {e}")

        # Fallback: el modelo devolvió un objeto en lugar de array
        obj_match = re.search(r'\{.*\}', text, re.DOTALL)
        if not obj_match:
            logger.error(f"No se encontró JSON en la respuesta: {text}")
            raise GeminiInvalidJSONError("No se encontró JSON en la respuesta del modelo")

        try:
            return [json.loads(obj_match.group())]
        except json.JSONDecodeError as e:
            logger.error(f"Error al parsear JSON: {e}")
            raise GeminiInvalidJSONError(f"JSON inválido: {e}")
