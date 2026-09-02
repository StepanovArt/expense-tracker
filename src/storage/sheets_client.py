"""Cliente para interactuar con Google Sheets."""
import json
import logging
import time
from functools import wraps
import gspread
from google.oauth2.service_account import Credentials
from src.utils.exceptions import GoogleSheetsError


logger = logging.getLogger('telegram-bot-gastos-llm')


def rate_limit(max_calls: int, period: int):
    """
    Decorador para limitar la cantidad de llamadas por período.

    Args:
        max_calls: Máximo número de llamadas permitidas
        period: Período en segundos
    """
    calls = []

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            # Limpiar llamadas antiguas
            calls[:] = [call_time for call_time in calls if now - call_time < period]

            if len(calls) >= max_calls:
                sleep_time = period - (now - calls[0])
                logger.warning(f"Rate limit alcanzado, esperando {sleep_time:.2f} segundos")
                time.sleep(sleep_time)
                calls[:] = []

            calls.append(now)
            return func(*args, **kwargs)

        return wrapper
    return decorator


class SheetsClient:
    """Cliente para escribir gastos en Google Sheets."""

    def __init__(self, spreadsheet_id: str, sheet_name: str,
                 credentials_path: str = '', credentials_json: str = ''):
        """
        Inicializa el cliente de Google Sheets.

        Args:
            credentials_path: Ruta al archivo JSON de credenciales
            spreadsheet_id: ID del spreadsheet de Google Sheets
            sheet_name: Nombre de la pestaña donde escribir

        Raises:
            GoogleSheetsError: Si no se puede autenticar o acceder al spreadsheet
        """
        self.spreadsheet_id = spreadsheet_id
        self.sheet_name = sheet_name

        try:
            logger.info("Autenticando con Google Sheets...")

            scopes = [
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]

            if credentials_json:
                credentials = Credentials.from_service_account_info(
                    json.loads(credentials_json), scopes=scopes
                )
            else:
                credentials = Credentials.from_service_account_file(
                    credentials_path, scopes=scopes
                )

            self.client = gspread.authorize(credentials)

            # Abrir el spreadsheet
            self.spreadsheet = self.client.open_by_key(spreadsheet_id)

            # Verificar que la pestaña existe
            try:
                self.worksheet = self.spreadsheet.worksheet(sheet_name)
                logger.info(f"Conectado a Google Sheets: '{sheet_name}'")
            except gspread.exceptions.WorksheetNotFound:
                raise GoogleSheetsError(f"Pestaña '{sheet_name}' no encontrada en el spreadsheet")

        except FileNotFoundError:
            raise GoogleSheetsError(f"Archivo de credenciales no encontrado: {credentials_path}")
        except Exception as e:
            logger.error(f"Error al conectar con Google Sheets: {e}")
            raise GoogleSheetsError(f"Error de autenticación con Google Sheets: {e}")

    @rate_limit(max_calls=50, period=60)
    def append_expense(self, fecha: str, descripcion: str, categoria: str, monto: float) -> None:
        """
        Agrega una fila con el gasto al final de la hoja.

        Rate limiting: Máximo 50 llamadas por minuto (Google limita a 60/min).

        Args:
            fecha: Fecha en formato YYYY-MM-DD
            descripcion: Descripción del gasto (mensaje original del usuario)
            categoria: Categoría del gasto
            monto: Monto del gasto

        Raises:
            GoogleSheetsError: Si hay un error al escribir en Sheets
        """
        try:
            logger.info(f"Registrando gasto: {categoria} - ${monto} - {fecha}")

            # Agregar fila al final de la hoja
            row = [fecha, descripcion, categoria, monto]
            self.worksheet.append_row(row, value_input_option='USER_ENTERED')

            logger.info("Gasto registrado exitosamente en Google Sheets")

        except Exception as e:
            logger.error(f"Error al escribir en Google Sheets: {e}")
            raise GoogleSheetsError(f"Error al registrar gasto en Sheets: {e}")

    def get_spending_stats(self) -> dict:
        """
        Lee todas las filas de la hoja y devuelve estadísticas para el mes
        y la semana calendario actual (lunes–domingo), en un solo recorrido.

        Returns:
            Dict con:
              'month_total'   float
              'week_total'    float
              'by_category'   dict categoria -> float  (totales del mes)
        """
        from datetime import date, timedelta
        today = date.today()
        current_month = today.strftime('%Y-%m')
        week_start = today - timedelta(days=today.weekday())   # lunes
        week_end = week_start + timedelta(days=6)              # domingo

        try:
            all_rows = self.worksheet.get_all_values()
            month_total = 0.0
            week_total = 0.0
            by_category: dict[str, float] = {}

            for row in all_rows:
                if len(row) < 4:
                    continue
                fecha_str, categoria, monto_str = row[0], row[2], row[3]
                if not fecha_str.startswith(current_month):
                    continue
                try:
                    monto = float(monto_str)
                    row_date = date.fromisoformat(fecha_str)
                except (ValueError, TypeError):
                    continue

                month_total += monto
                by_category[categoria] = by_category.get(categoria, 0.0) + monto

                if week_start <= row_date <= week_end:
                    week_total += monto

            return {
                'month_total': month_total,
                'week_total': week_total,
                'by_category': by_category,
            }
        except Exception as e:
            logger.error(f"Error al obtener estadísticas: {e}")
            return {'month_total': 0.0, 'week_total': 0.0, 'by_category': {}}

    @rate_limit(max_calls=50, period=60)
    def append_expenses(self, expenses: list) -> None:
        """
        Agrega múltiples gastos en un solo llamado a la API usando append_rows.

        Args:
            expenses: Lista de dicts con keys fecha, descripcion, categoria, monto

        Raises:
            GoogleSheetsError: Si hay un error al escribir en Sheets
        """
        try:
            logger.info(f"Registrando {len(expenses)} gastos en lote")
            rows = [
                [e['fecha'], e['descripcion'], e['categoria'], e['monto']]
                for e in expenses
            ]
            self.worksheet.append_rows(rows, value_input_option='USER_ENTERED')
            logger.info(f"{len(expenses)} gastos registrados exitosamente en Google Sheets")
        except Exception as e:
            logger.error(f"Error al escribir en Google Sheets: {e}")
            raise GoogleSheetsError(f"Error al registrar gastos en Sheets: {e}")
