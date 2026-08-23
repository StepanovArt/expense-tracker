"""Validadores para datos de gastos."""
from datetime import datetime
from typing import List, Tuple


def validate_expense_data(data: dict, valid_categories: list) -> Tuple[bool, str]:
    """
    Valida los datos de un gasto.

    Args:
        data: Diccionario con monto, categoria y fecha
        valid_categories: Lista de categorías válidas

    Returns:
        Tupla (es_valido, mensaje_error)
        Si es válido: (True, "")
        Si es inválido: (False, "mensaje de error")
    """
    # Verificar campos requeridos
    required_fields = ['monto', 'categoria', 'fecha', 'descripcion']
    for field in required_fields:
        if field not in data:
            return False, f"Campo requerido '{field}' no encontrado"

    # Validar monto
    try:
        monto = float(data['monto'])
        if monto <= 0:
            return False, "El monto debe ser mayor que 0"
    except (ValueError, TypeError):
        return False, "El monto debe ser un número válido"

    # Validar categoría
    categoria = data['categoria']
    if categoria not in valid_categories:
        return False, f"Categoría '{categoria}' no válida. Use una de: {', '.join(valid_categories)}"

    # Validar formato de fecha
    try:
        datetime.strptime(data['fecha'], '%Y-%m-%d')
    except ValueError:
        return False, "La fecha debe tener formato YYYY-MM-DD"

    return True, ""


def validate_expense_list(data: list, valid_categories: list) -> List[Tuple[dict, str]]:
    """
    Valida cada gasto de una lista de forma independiente.

    Args:
        data: Lista de diccionarios con datos de gastos
        valid_categories: Lista de categorías válidas

    Returns:
        Lista de tuplas (gasto_valido_o_None, error_o_None).
        Si el elemento es válido: (dict, None).
        Si es inválido: (None, mensaje_de_error).
    """
    results = []
    for i, item in enumerate(data):
        is_valid, error = validate_expense_data(item, valid_categories)
        if is_valid:
            results.append((item, None))
        else:
            results.append((None, f"gasto #{i + 1}: {error}"))
    return results
