"""Entry point del bot de Telegram."""

import logging
import signal
import sys
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from src.config import Config
from src.utils.logger import setup_logger
from src.llm.factory import create_llm_connector
from src.storage.sheets_client import SheetsClient
from src.bot.telegram_handler import start_command, help_command, handle_message, error_handler


logger = None


def main():
    """Función principal que inicializa y arranca el bot."""
    global logger

    try:
        # Cargar configuración
        print("Cargando configuración...")
        config = Config()

        # Inicializar logger
        logger = setup_logger(config)
        logger.info("=" * 50)
        logger.info("Iniciando bot de Telegram para registro de gastos")
        logger.info("=" * 50)

        # Crear clientes
        logger.info(f"Inicializando conector LLM ({config.llm_connector})...")
        llm_client = create_llm_connector(config)

        logger.info(
            f"Inicializando cliente Google Sheets (spreadsheet: {config.spreadsheet_id})..."
        )
        sheets_client = SheetsClient(
            spreadsheet_id=config.spreadsheet_id,
            sheet_name=config.sheet_name,
            credentials_path=config.google_credentials_path,
            credentials_json=config.google_credentials_json,
        )

        logger.info("Configurando bot de Telegram...")

        # Crear aplicación de Telegram
        application = Application.builder().token(config.telegram_bot_token).build()

        # Almacenar clientes y configuración en bot_data para acceso en handlers
        application.bot_data["llm_connector"] = llm_client
        application.bot_data["sheets_client"] = sheets_client
        application.bot_data["categories"] = config.expense_categories

        # Agregar handlers — solo para el usuario autorizado
        user_filter = filters.User(user_id=[config.allowed_user_id])
        application.add_handler(CommandHandler("start", start_command, filters=user_filter))
        application.add_handler(CommandHandler("help", help_command, filters=user_filter))
        application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND & user_filter, handle_message)
        )
        
        # Manejador de errores global
        application.add_error_handler(error_handler)

        # Configurar manejo de señales para shutdown graceful
        def signal_handler(sig, frame):
            logger.info("Señal de terminación recibida. Deteniendo bot...")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # IMPORTANTE: Usar polling, NO webhooks
        logger.info("Bot iniciado exitosamente. Escuchando mensajes...")
        logger.info(f"Categorías configuradas: {', '.join(config.expense_categories)}")
        logger.info("Presiona Ctrl+C para detener el bot") 

        application.run_polling(
            poll_interval=30.0,
            timeout=30,
            drop_pending_updates=False,
            allowed_updates=Update.ALL_TYPES,
            bootstrap_retries=-1,  # Reintentar infinitamente en errores de red
            read_timeout=60,
            connect_timeout=30,
        )

    except ValueError as e:
        print(f"Error de configuración: {e}")
        if logger:
            logger.error(f"Error de configuración: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"Error fatal: {e}")
        if logger:
            logger.exception(f"Error fatal al iniciar bot: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
