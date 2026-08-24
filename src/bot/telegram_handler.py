"""Handlers del bot de Telegram."""
import logging
from telegram import Update
from telegram.ext import ContextTypes
from src.llm.prompt_builder import build_prompt
from src.llm.base import LLMConnector
from src.storage.sheets_client import SheetsClient
from src.utils.validators import validate_expense_data, validate_expense_list
from src.utils.exceptions import (
    OllamaConnectionError,
    OllamaInvalidJSONError,
    GeminiConnectionError,
    GeminiInvalidJSONError,
    InvalidExpenseDataError,
    GoogleSheetsError
)


logger = logging.getLogger('telegram-bot-gastos-llm')


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handler para el comando /start.

    Args:
        update: Update de Telegram
        context: Contexto del bot
    """
    welcome_message = """¡Hola! 👋

Soy tu asistente personal de gastos. Escríbeme tus gastos como quieras y los registro automáticamente en tu planilla.

📝 Ejemplos:
• "super 45"
• "taxi 12 ayer"
• "spotify y chatgpt 30 en total"
• "cena con amigos 80 el viernes"

Podés mandar varios gastos en un solo mensaje, cada uno se registra por separado.

/help — categorías disponibles"""

    await update.message.reply_text(welcome_message)
    logger.info(f"Comando /start recibido de usuario {update.effective_user.id}")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handler para el comando /help.

    Args:
        update: Update de Telegram
        context: Contexto del bot
    """
    categories = context.bot_data.get('categories', [])
    category_descriptions = context.bot_data.get('category_descriptions', {})

    if category_descriptions:
        categories_lines = '\n'.join(
            f'• {cat} — {category_descriptions[cat]}' if cat in category_descriptions else f'• {cat}'
            for cat in categories
        )
    else:
        categories_lines = '\n• '.join(categories)
        categories_lines = '• ' + categories_lines

    help_message = f"""ℹ️ Ayuda

📂 Categorías:
{categories_lines}

💡 Consejos:
• No hace falta escribir la categoría, el bot la infiere
• Si no ponés fecha, asume hoy
• Varios gastos en un mensaje → cada uno en su fila"""

    await update.message.reply_text(help_message)
    logger.info(f"Comando /help recibido de usuario {update.effective_user.id}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler para mensajes de texto con gastos."""
    user_id = update.effective_user.id
    logger.info(f"Mensaje recibido del usuario {user_id}")

    await update.message.reply_text("⏳ Procesando tu mensaje, puede tardar un momento...")
    await handle_text_message(update.message.text, update, context)


async def handle_text_message(user_message, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handler para mensajes de texto con gastos.

    Este es el flujo completo:
    1. Construir prompt con fecha actual
    2. Generar respuesta con Ollama
    3. Validar datos extraídos
    4. Guardar en Google Sheets
    5. Confirmar al usuario

    Args:
        user_message: String mensaje del usuario
        update: Update de Telegram
        context: Contexto del bot
    """
    user_id = update.effective_user.id

    logger.info(f"Mensaje del usuario {user_id}: {user_message}")

    # Obtener clientes del contexto
    llm_connector: LLMConnector = context.bot_data['llm_connector']
    sheets_client: SheetsClient = context.bot_data['sheets_client']
    categories = context.bot_data['categories']
    category_descriptions = context.bot_data.get('category_descriptions', {})

    try:
        # Enviar indicador de "escribiendo..."
        await update.message.chat.send_action(action="typing")

        # 1. Construir prompt con fecha actual
        prompt = build_prompt(user_message, categories, category_descriptions)

        # 2. Generar respuesta del LLM (ahora devuelve lista)
        expenses = llm_connector.generate(prompt)
        logger.debug(f"Datos extraídos: {expenses}")

        # 3. Validar cada gasto individualmente
        validation_results = validate_expense_list(expenses, categories)
        valid_expenses = [item for item, err in validation_results if item is not None]
        errors = [err for _, err in validation_results if err is not None]

        if not valid_expenses:
            error_detail = '\n'.join(f"• {e}" for e in errors)
            logger.warning(f"Todos los gastos inválidos: {errors}")
            await update.message.reply_text(
                f"❌ No se pudo registrar ningún gasto:\n{error_detail}\n\n"
                f"💡 Intenta reformular tu mensaje con monto y categoría claros."
            )
            return

        # 4. Guardar en Google Sheets en un solo llamado API
        sheets_client.append_expenses(valid_expenses)

        # 5. Confirmar al usuario (con advertencia de parciales si aplica)
        confirmation_message = format_confirmation_message_list(valid_expenses)
        if errors:
            error_detail = '\n'.join(f"• {e}" for e in errors)
            confirmation_message += f"\n\n⚠️ No se registraron:\n{error_detail}"

        await update.message.reply_text(confirmation_message)

        logger.info(
            f"Gastos para usuario {user_id}: {len(valid_expenses)} registrados, {len(errors)} inválidos"
        )

    except (OllamaConnectionError, GeminiConnectionError):
        error_msg = "❌ Error de conexión con el LLM.\n🔧 El servicio no está disponible. Intenta más tarde."
        await update.message.reply_text(error_msg)
        logger.error("Error de conexión con el conector LLM")

    except (OllamaInvalidJSONError, GeminiInvalidJSONError):
        error_msg = (
            "❌ No pude entender tu mensaje.\n\n"
            "💡 Intenta ser más específico:\n"
            "• 'Compré [cosa] por [monto] AED'\n"
            "• 'Gasté [monto] en [categoría]'"
        )
        await update.message.reply_text(error_msg)
        logger.error("JSON inválido desde el conector LLM")

    except GoogleSheetsError as e:
        error_msg = f"❌ Error al guardar en Google Sheets.\n🔧 {str(e)}"
        await update.message.reply_text(error_msg)
        logger.error(f"Error de Google Sheets: {e}")

    except Exception as e:
        error_msg = "❌ Ocurrió un error inesperado.\n🔧 Intenta nuevamente más tarde."
        await update.message.reply_text(error_msg)
        logger.exception(f"Error inesperado: {e}")


def format_confirmation_message_list(expenses: list) -> str:
    """
    Formatea el mensaje de confirmación para uno o varios gastos.

    Args:
        expenses: Lista de dicts con monto, categoria, fecha, descripcion

    Returns:
        Mensaje de confirmación formateado
    """
    if len(expenses) == 1:
        e = expenses[0]
        return (
            f"✅ Gasto registrado correctamente\n\n"
            f"💰 Monto: AED {e['monto']:.2f}\n"
            f"📂 Categoría: {e['categoria']}\n"
            f"📅 Fecha: {e['fecha']}\n"
            f"📝 Descripción: {e['descripcion']}\n\n"
            f"Registro completado en Google Sheets ✨"
        )

    lines = [f"✅ {len(expenses)} gastos registrados en Google Sheets ✨\n"]
    for e in expenses:
        lines.append(f"• AED {e['monto']:.2f} — {e['categoria']} ({e['fecha']}): {e['descripcion']}")
    return '\n'.join(lines)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Manejador global para excepciones no capturadas.
    """
    logger.error("Excepción no manejada capturada por el manejador global:", exc_info=context.error)
