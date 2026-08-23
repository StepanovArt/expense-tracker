"""Tests para _extract_json_from_text en clientes LLM."""
import sys
import pytest
from unittest.mock import MagicMock
from src.utils.exceptions import GeminiInvalidJSONError, OllamaInvalidJSONError


def _make_gemini_client():
    """Crea GeminiClient sin llamar __init__, mockeando google.generativeai si no está instalado."""
    genai_mock = MagicMock()
    sys.modules.setdefault('google', MagicMock(generativeai=genai_mock))
    sys.modules.setdefault('google.generativeai', genai_mock)

    # Importar después de parchear sys.modules
    import importlib
    import src.llm.gemini_client as mod
    importlib.reload(mod)
    return mod.GeminiClient.__new__(mod.GeminiClient)


# ── GeminiClient ───────────────────────────────────────────────────────────────

class TestGeminiExtract:
    def setup_method(self):
        self.client = _make_gemini_client()

    def _extract(self, text):
        from src.llm.gemini_client import GeminiClient
        return GeminiClient._extract_json_from_text(self.client, text)

    def test_array_single(self):
        text = '[{"monto": 100, "categoria": "Supermercado", "fecha": "2025-08-01", "descripcion": "pan"}]'
        result = self._extract(text)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["monto"] == 100

    def test_array_multiple(self):
        text = '[{"monto": 80, "categoria": "Supermercado", "fecha": "2025-08-01", "descripcion": "pan"}, {"monto": 25, "categoria": "Salidas", "fecha": "2025-08-01", "descripcion": "taxi"}]'
        result = self._extract(text)
        assert len(result) == 2

    def test_fallback_object(self):
        # Si el modelo devuelve objeto en lugar de array, lo envuelve
        text = '{"monto": 100, "categoria": "Supermercado", "fecha": "2025-08-01", "descripcion": "pan"}'
        result = self._extract(text)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_array_with_surrounding_text(self):
        text = 'Aquí está el resultado: [{"monto": 50, "categoria": "Delivery", "fecha": "2025-08-01", "descripcion": "pizza"}]'
        result = self._extract(text)
        assert len(result) == 1
        assert result[0]["monto"] == 50

    def test_no_json_raises(self):
        with pytest.raises(GeminiInvalidJSONError):
            self._extract("No hay JSON aquí")


# ── OllamaClient ──────────────────────────────────────────────────────────────

class TestOllamaExtract:
    def setup_method(self):
        from src.llm.ollama_client import OllamaClient
        self.client = OllamaClient.__new__(OllamaClient)

    def _extract(self, text):
        from src.llm.ollama_client import OllamaClient
        return OllamaClient._extract_json_from_text(self.client, text)

    def test_array_single(self):
        text = '[{"monto": 200, "categoria": "Combustible", "fecha": "2025-08-01", "descripcion": "nafta"}]'
        result = self._extract(text)
        assert isinstance(result, list)
        assert result[0]["monto"] == 200

    def test_array_multiple(self):
        text = '[{"monto": 80, "categoria": "Supermercado", "fecha": "2025-08-01", "descripcion": "pan"}, {"monto": 25, "categoria": "Salidas", "fecha": "2025-08-01", "descripcion": "taxi"}]'
        result = self._extract(text)
        assert len(result) == 2

    def test_fallback_object(self):
        text = '{"monto": 150, "categoria": "Delivery", "fecha": "2025-08-01", "descripcion": "sushi"}'
        result = self._extract(text)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_no_json_raises(self):
        with pytest.raises(OllamaInvalidJSONError):
            self._extract("texto sin json")
