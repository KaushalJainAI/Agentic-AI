# import logging
# from telegram import Update
# from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
# import os




# class TelegramBot:
#     def __init__(self, token):
#         """Initialize the bot with the provided token."""
#         self.token = token
#         self.application = None
#         self._setup_logging()
#         self._build_application()
        
#     def _setup_logging(self):
#         """Set up logging configuration."""
#         logging.basicConfig(
#             format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#             level=logging.INFO
#         )
#         self.logger = logging.getLogger(__name__)
    
#     def _build_application(self):
#         """Build the application and add all handlers."""
#         self.application = Application.builder().token(self.token).build()
        
#         # Add command handlers
#         self.application.add_handler(CommandHandler("start", self.start))
#         self.application.add_handler(CommandHandler("help", self.help_command))
#         self.application.add_handler(CommandHandler("echo", self.echo))
#         self.application.add_handler(CommandHandler("info", self.info))
        
#         # Add message handlers
#         self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
#         self.application.add_handler(MessageHandler(filters.Document.ALL, self.handle_document))
    
#     async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
#         """Handle the /start command."""
#         await update.message.reply_text('Hello! Welcome to my bot. Type /help to see available commands.')
    
#     async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
#         """Handle the /help command."""
#         help_text = """
#             Available Commands:
#             /start - Start the bot
#             /help - Show this help message
#             /echo - I'll repeat what you say
#             /info - Get bot information
#             """
#         await update.message.reply_text(help_text)
    
#     async def echo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
#         """Handle the /echo command."""
#         await update.message.reply_text(f"You said: {update.message.text}")
    
#     async def info(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
#         """Handle the /info command."""
#         await update.message.reply_text("I'm a simple bot running on a local computer!")
    
#     async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
#         """Handle regular text messages."""
#         user_message = update.message.text
#         await update.message.reply_text(f"I received your message: '{user_message}'. Try using /help for commands!")
    
#     async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
#         """Handle document uploads."""
#         document = update.message.document
#         await update.message.reply_text(f"I received a file: {document.file_name}")
    
#     def add_custom_handler(self, handler):
#         """Add a custom handler to the bot."""
#         self.application.add_handler(handler)
    
#     def run(self):
#         """Start the bot and run it until stopped."""
#         print("Bot is starting...")
#         self.application.run_polling(allowed_updates=Update.ALL_TYPES)
    
#     def stop(self):
#         """Stop the bot."""
#         if self.application:
#             self.application.stop()
#             print("Bot stopped.")

# # Usage
# if __name__ == '__main__':
#     # Enable logging
#     logging.basicConfig(
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#         level=logging.INFO
#     )
#     logger = logging.getLogger(__name__)

#     # Replace 'YOUR_BOT_TOKEN' with the token you got from BotFather
#     TOKEN = os.getenv('TELEGRAM_BOT_API_KEY')
#     print(TOKEN)

#     os.environ["TELEGRAM_BOT_API_KEY"] = TOKEN



#     # Create and run the bot
#     bot = TelegramBot(TOKEN)
#     bot.run()





