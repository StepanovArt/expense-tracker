"""Interfaz base para conectores LLM."""
from abc import ABC, abstractmethod


class LLMConnector(ABC):
    """Interfaz común para todos los conectores LLM."""

    @abstractmethod
    def generate(self, prompt: str) -> list:
        """
        Genera una respuesta a partir de un prompt y retorna lista de gastos parseados.

        Args:
            prompt: Prompt completo para el modelo

        Returns:
            Lista de dicts con los datos de los gastos (monto, categoria, fecha, descripcion)

        Raises:
            OllamaConnectionError / GeminiConnectionError: Si no se puede conectar
            OllamaInvalidJSONError / GeminiInvalidJSONError: Si no se puede parsear el JSON
        """
