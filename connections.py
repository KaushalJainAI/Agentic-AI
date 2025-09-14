import logging
import os
import asyncio
import aiohttp
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
from config import SuperAgentConfig


# Load environment variables
load_dotenv()

class TelegramBot:
    def __init__(self, name, token, backend_url):
        self.token = token
        self.name = name
        self.backend_url = backend_url
        self.application = None
        self._setup_logging()
        self._build_application()
        
    def _setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)
    
    def _build_application(self):
        """Build the Telegram application with handlers"""
        try:
            self.application = Application.builder().token(self.token).build()
            self.application.add_handler(CommandHandler("start", self.start))
            self.application.add_handler(CommandHandler("help", self.help_command))
            self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
            self.logger.info("Application built successfully")
        except Exception as e:
            self.logger.error(f"Failed to build application: {e}")
            raise
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        welcome_message = (
            f'Hello! Welcome to {self.name}! 🤖\n'
            'I\'m your AI assistant bot. Type /help to see available commands.'
        )
        await update.message.reply_text(welcome_message)
        self.logger.info(f"Start command used by user {update.effective_user.id}")
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = (
            "🤖 **Available Commands:**\n\n"
            "• `/start` - Start the bot\n"
            "• `/help` - Show this help message\n\n"
            "💬 **How to use:**\n"
            "Just type any message, and I will process it using AI and reply to you!"
        )
        await update.message.reply_text(help_text, parse_mode='Markdown')
        self.logger.info(f"Help command used by user {update.effective_user.id}")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle regular text messages"""
        if not update.message or not update.message.text:
            return
            
        user_message = update.message.text
        chat_id = update.message.chat_id
        user_id = update.effective_user.id
        
        self.logger.info(f"Message from user {user_id}: {user_message[:50]}...")
        
        # Send typing action to show bot is processing
        await context.bot.send_chat_action(chat_id=chat_id, action='typing')
        
        try:
            # Set up timeout for HTTP requests
            timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                payload = {
                    "message": user_message,
                    "userId": str(chat_id),
                    "username": update.effective_user.username or "Unknown"
                }
                
                async with session.post(
                    self.backend_url,
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        reply = data.get("reply", "Sorry, I didn't get a proper response.")
                        
                        # Limit reply length (Telegram has a 4096 character limit)
                        if len(reply) > 4000:
                            reply = reply[:4000] + "...\n\n(Message truncated due to length)"
                            
                    elif response.status == 404:
                        reply = "Sorry, the AI service is not available right now. Please try again later."
                    elif response.status == 500:
                        reply = "The AI service is experiencing issues. Please try again in a moment."
                    else:
                        reply = f"Unexpected server response: {response.status}. Please try again."
                        self.logger.warning(f"Unexpected status code: {response.status}")
                        
        except asyncio.TimeoutError:
            self.logger.error("Request to backend timed out")
            reply = "Sorry, the request took too long. Please try again with a shorter message."
            
        except aiohttp.ClientConnectorError:
            self.logger.error("Could not connect to backend server")
            reply = "Sorry, I can't connect to the AI service right now. Please try again later."
            
        except aiohttp.ClientError as e:
            self.logger.error(f"HTTP client error: {e}")
            reply = "Sorry, there was a connection error. Please try again later."
            
        except Exception as e:
            self.logger.error(f"Unexpected error in handle_message: {e}")
            reply = "Sorry, something unexpected happened. Please try again."
        
        try:
            await update.message.reply_text(reply)
        except Exception as e:
            self.logger.error(f"Failed to send reply: {e}")
    
    async def post_init(self, application: Application) -> None:
        """Post initialization hook"""
        self.logger.info(f"Bot {self.name} initialized successfully")
    
    async def shutdown(self, application: Application) -> None:
        """Graceful shutdown hook"""
        self.logger.info(f"Bot {self.name} is shutting down...")
    
    def run(self):
        """Start the bot"""
        try:
            self.logger.info(f"{self.name} is starting...")
            
            # Add post init and shutdown handlers
            self.application.post_init = self.post_init
            self.application.post_shutdown = self.shutdown
            
            # Run the bot
            self.application.run_polling(
                drop_pending_updates=True,  # Ignore messages sent while bot was offline
                allowed_updates=Update.ALL_TYPES
            )
            
        except Exception as e:
            self.logger.error(f"Failed to start bot: {e}")
            raise


def validate_environment():
    """Validate required environment variables"""
    required_vars = ['TELEGRAM_BOT_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")


if __name__ == '__main__':
    try:
        # Validate environment variables
        validate_environment()
        
        # Get configuration
        TOKEN = os.getenv('TELEGRAM_BOT_API_KEY')
        BOT_USERNAME = SuperAgentConfig.BOT_USERNAME
        BACKEND_URL = f"http://{SuperAgentConfig.API_HOST}:{SuperAgentConfig.API_PORT}/chat"
        
        print("🤖 Initializing Telegram Bot...")
        print(f"   Bot Username: {BOT_USERNAME}")
        print(f"   Backend URL: {BACKEND_URL}")
        
        # Create and run the bot
        bot = TelegramBot(
            name=BOT_USERNAME,
            token=TOKEN,
            backend_url=BACKEND_URL
        )
        
        bot.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        logging.error(f"❌ Fatal error: {e}")
        raise
