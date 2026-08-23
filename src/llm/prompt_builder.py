"""Constructor de prompts para Ollama con contexto temporal."""
from datetime import datetime


def build_prompt(user_message: str, categories: list) -> str:
    """
    Construye el prompt completo con fecha actual y mensaje del usuario.

    IMPORTANTE: Inyecta la fecha actual para que el modelo pueda interpretar
    referencias temporales como "ayer", "anteayer", "la semana pasada", etc.

    Args:
        user_message: Mensaje del usuario sobre el gasto
        categories: Lista de categorías válidas

    Returns:
        Prompt completo listo para enviar a Ollama
    """
    today = datetime.now().strftime("%Y-%m-%d")
    categories_str = '", "'.join(categories)

    system_prompt = f"""Eres un asistente contable.
HOY ES {today}.

Tu única función es recibir frases de gastos y responder EXCLUSIVAMENTE con un array JSON.
Formato: [{{"monto": <float>, "categoria": <string>, "fecha": <string formato Y-m-d>, "descripcion": <string>}}]

Si el mensaje contiene múltiples gastos, incluye un objeto por cada gasto dentro del mismo array.
La descripción implica un breve resumen del gasto, por ejemplo si se hace mención a compra de algo particular, eso que se compró, o si se hace mención a un lugar al que se fue, se menciona ese lugar. Es todo aquello que ayude a identificar el gasto más allá de la categoría

Si no hay fecha explícita, asume hoy.
Las categorías posibles son: "{categories_str}".

IMPORTANTE: Responde SOLO con el array JSON, sin texto adicional."""

    return f"{system_prompt}\n\nUsuario: {user_message}\n\nAsistente:"
