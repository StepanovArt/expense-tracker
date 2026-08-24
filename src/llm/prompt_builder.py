"""Constructor de prompts para Ollama con contexto temporal."""
from datetime import datetime


def build_prompt(user_message: str, categories: list, category_descriptions: dict = None) -> str:
    today = datetime.now().strftime("%Y-%m-%d")

    if category_descriptions:
        category_lines = '\n'.join(
            f'  - {cat}: {category_descriptions[cat]}' if cat in category_descriptions else f'  - {cat}'
            for cat in categories
        )
        categories_section = f"Las categorías posibles son (con ejemplos de qué incluye cada una):\n{category_lines}\nUsa EXACTAMENTE uno de estos nombres de categoría."
    else:
        categories_str = '", "'.join(categories)
        categories_section = f'Las categorías posibles son: "{categories_str}".'

    system_prompt = f"""Eres un asistente contable.
HOY ES {today}.

Tu única función es recibir frases de gastos y responder EXCLUSIVAMENTE con un array JSON.
Formato: [{{"monto": <float>, "categoria": <string>, "fecha": <string formato Y-m-d>, "descripcion": <string>}}]

Si el mensaje contiene múltiples gastos, incluye un objeto por cada gasto dentro del mismo array.
La descripción es un breve resumen del gasto: qué se compró o a qué lugar se fue, lo que ayude a identificar el gasto más allá de la categoría.

Si no hay fecha explícita, asume hoy.
{categories_section}

IMPORTANTE: Responde SOLO con el array JSON, sin texto adicional."""

    return f"{system_prompt}\n\nUsuario: {user_message}\n\nAsistente:"
