"""Tests para src/utils/validators.py"""
import pytest
from src.utils.validators import validate_expense_data, validate_expense_list

VALID_CATEGORIES = ["Supermercado", "Salidas", "Combustible", "Delivery"]

VALID_EXPENSE = {
    "monto": 150.0,
    "categoria": "Supermercado",
    "fecha": "2025-08-01",
    "descripcion": "pan y leche",
}


# ── validate_expense_data ──────────────────────────────────────────────────────

class TestValidateExpenseData:
    def test_valid_expense(self):
        ok, err = validate_expense_data(VALID_EXPENSE, VALID_CATEGORIES)
        assert ok is True
        assert err == ""

    def test_missing_field(self):
        data = {k: v for k, v in VALID_EXPENSE.items() if k != "monto"}
        ok, err = validate_expense_data(data, VALID_CATEGORIES)
        assert ok is False
        assert "monto" in err

    def test_monto_zero(self):
        ok, err = validate_expense_data({**VALID_EXPENSE, "monto": 0}, VALID_CATEGORIES)
        assert ok is False
        assert "mayor que 0" in err

    def test_monto_negative(self):
        ok, err = validate_expense_data({**VALID_EXPENSE, "monto": -10}, VALID_CATEGORIES)
        assert ok is False

    def test_monto_string_invalid(self):
        ok, err = validate_expense_data({**VALID_EXPENSE, "monto": "abc"}, VALID_CATEGORIES)
        assert ok is False
        assert "número válido" in err

    def test_monto_string_numeric(self):
        # "150" debería ser aceptado (cast a float)
        ok, err = validate_expense_data({**VALID_EXPENSE, "monto": "150"}, VALID_CATEGORIES)
        assert ok is True

    def test_invalid_category(self):
        ok, err = validate_expense_data({**VALID_EXPENSE, "categoria": "Vacaciones"}, VALID_CATEGORIES)
        assert ok is False
        assert "Vacaciones" in err

    def test_invalid_date_format(self):
        ok, err = validate_expense_data({**VALID_EXPENSE, "fecha": "01/08/2025"}, VALID_CATEGORIES)
        assert ok is False
        assert "YYYY-MM-DD" in err

    def test_invalid_date_value(self):
        ok, err = validate_expense_data({**VALID_EXPENSE, "fecha": "2025-13-01"}, VALID_CATEGORIES)
        assert ok is False


# ── validate_expense_list ──────────────────────────────────────────────────────

class TestValidateExpenseList:
    def test_all_valid(self):
        data = [VALID_EXPENSE, {**VALID_EXPENSE, "monto": 50.0, "descripcion": "taxi"}]
        results = validate_expense_list(data, VALID_CATEGORIES)
        assert len(results) == 2
        for item, err in results:
            assert item is not None
            assert err is None

    def test_all_invalid(self):
        data = [
            {**VALID_EXPENSE, "monto": -1},
            {**VALID_EXPENSE, "categoria": "Inexistente"},
        ]
        results = validate_expense_list(data, VALID_CATEGORIES)
        for item, err in results:
            assert item is None
            assert err is not None

    def test_partial_valid(self):
        data = [
            VALID_EXPENSE,                               # válido
            {**VALID_EXPENSE, "categoria": "NoExiste"},  # inválido
            {**VALID_EXPENSE, "monto": 30.0},            # válido
        ]
        results = validate_expense_list(data, VALID_CATEGORIES)
        valid = [item for item, _ in results if item is not None]
        errors = [err for _, err in results if err is not None]
        assert len(valid) == 2
        assert len(errors) == 1
        assert "gasto #2" in errors[0]

    def test_empty_list(self):
        results = validate_expense_list([], VALID_CATEGORIES)
        assert results == []

    def test_single_expense(self):
        results = validate_expense_list([VALID_EXPENSE], VALID_CATEGORIES)
        assert len(results) == 1
        item, err = results[0]
        assert item == VALID_EXPENSE
        assert err is None
