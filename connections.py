import os
import asyncio
import aiohttp
import logging
from typing import Optional
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler, 
    filters, ContextTypes
)
from telegram.constants import ChatAction
from dotenv import load_dotenv
load_dotenv()


class SuperAgentTelegramBot:
    """
    Enhanced Telegram bot that connects to the SuperAgent Flask server
    with improved error handling and user experience.
    """
    
    def __init__(self, name: str, token: str, backend_url: str):
        self.token = token
        self.name = name
        self.backend_url = backend_url
        self.application = None
        self._setup_logging()
        self._build_application()
        self.active_tasks = {}
        
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
            
            # Add command handlers
            self.application.add_handler(CommandHandler("start", self.start))
            self.application.add_handler(CommandHandler("help", self.help_command))
            self.application.add_handler(CommandHandler("status", self.status_command))
            
            # Add message handler
            self.application.add_handler(
                MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
            )
            
            self.logger.info("Telegram application built successfully")
        except Exception as e:
            self.logger.error(f"Failed to build application: {e}")
            raise
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        welcome_message = (
            f"🤖 **Welcome to {self.name}!**\n\n"
            "I'm an intelligent AI assistant powered by multiple specialized agents. "
            "I can help you with:\n\n"
            "• 💬 General conversations and questions\n"
            "• 🌐 Web scraping and research\n"
            "• 📊 Database queries and analysis\n"
            "• 📁 Data processing tasks\n\n"
            "Just type your question or request, and I'll route it to the best agent for the job!\n\n"
            "Type /help to see available commands."
        )
        await update.message.reply_text(welcome_message, parse_mode='HTML')
        self.logger.info(f"Start command used by user {update.effective_user.id}")
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = (
            "🤖 **SuperAgent Commands:**\n\n"
            "• `/start` - Welcome message and introduction\n"
            "• `/help` - Show this help message\n"
            "• `/status` - Check system status\n\n"
            "💡 **How to use:**\n"
            "Simply type your message, and I'll automatically select the best agent to help you:\n\n"
            "• **Chatbot**: General conversations, Q&A\n"
            "• **Web Scraping**: Research, data collection from websites\n"
            "• **Database**: Query and analyze data\n\n"
            "**Examples:**\n"
            "• \"Tell me about artificial intelligence\"\n"
            "• \"Scrape product prices from Amazon\"\n"
            "• \"Extract customer data from database\"\n"
        )
        await update.message.reply_text(help_text, parse_mode='HTML')
        self.logger.info(f"Help command used by user {update.effective_user.id}")
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        try:
            # Check if backend is accessible
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{self.backend_url.replace('/chat', '/health')}") as response:
                    if response.status == 200:
                        status_message = "✅ **System Status: Online**\n\nAll services are running normally."
                    else:
                        status_message = f"⚠️ **System Status: Degraded**\n\nBackend returned status code: {response.status}"
        except Exception as e:
            status_message = "❌ **System Status: Offline**\n\nCannot connect to backend services."
            self.logger.error(f"Status check failed: {e}")
        
        await update.message.reply_text(status_message, parse_mode='HTML')
        self.logger.info(f"Status command used by user {update.effective_user.id}")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle regular text messages with enhanced error handling"""
        if not update.message or not update.message.text:
            return
            
        user_message = update.message.text.strip()
        chat_id = update.message.chat_id
        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"
        
        self.logger.info(f"Message from user {user_id} (@{username}): {user_message[:100]}...")
        
        await update.message.reply_text("Working on your request... This may take a moment.")

        # Send typing action
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

        task_id = f"{user_id}_{update.message.message_id}"

        task = asyncio.create_task(self._process_and_reply(user_message, chat_id, user_id, username,  task_id))
        self.active_tasks[task_id] = task

    
    async def _process_and_reply(self, user_message: str, user_id: int, task_id: int, context: ContextTypes.DEFAULT_TYPE, username: Optional[str] = None):
        """New async function to handle the API call and reply."""

        username = username or "Unknown"
        
        # Prepare payload for SuperAgent Flask server
        payload = {
            "message": user_message,
            "userId": str(user_id),
            "username": username
        }

        try:    
            # Send request to Flask server
            timeout = aiohttp.ClientTimeout(total=300)  # Longer timeout for complex queries
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.backend_url.replace('/chat', '/process'),
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        reply = data.get("reply", "I processed your request but couldn't generate a response.")
                        success = data.get("success", False)
                        metadata = data.get("metadata", {})
                        
                        # Add metadata info for transparency
                        if success and metadata.get("agent_used"):
                            agent_used = metadata["agent_used"]
                            reply += f"\n\n_Processed by: {agent_used}_"
                        
                    else:
                        if response.headers.get("Content-Type", "").startswith("application/json"):
                            error_data = await response.json()
                        else:
                            error_data = {}
                        reply = self._get_error_message(response.status, error_data)
                        
        except asyncio.TimeoutError:
            self.logger.error("Request to backend timed out")
            reply = (
                "⏱️ Your request is taking longer than expected. "
                "This might be a complex query - Wait for the response"
            )
            
        except aiohttp.ClientConnectorError:
            self.logger.error("Could not connect to backend server")
            reply = (
                "🔧 I'm currently unable to connect to my processing services. "
                "Please try again in a few moments."
            )
            
        except Exception as e:
            self.logger.error(f"Unexpected error in handle_message: {e}")
            reply = (
                "❌ I encountered an unexpected error while processing your request. "
                "Please try again or contact support if the issue persists."
            )
        
        # Ensure reply isn't too long for Telegram
        if len(reply) > 4000:
            reply = reply[:4000] + "...\n\n📝 _(Message truncated due to length)_"
        
        try:
            await context.bot.send_message(
                chat_id=user_id,
                text=reply,
                parse_mode='HTML'
            )
        except Exception as e:
            self.logger.error(f"Failed to send reply to user {user_id}: {e}")

        self.active_tasks.pop(task_id, None)

    
    def _get_error_message(self, status_code: int, error_data: dict) -> str:
        """Generate user-friendly error messages based on status code"""
        error_messages = {
            400: "❌ There was an issue with your request format. Please try rephrasing your message.",
            404: "🔍 The service endpoint is not available. Please try again later.",
            500: "⚙️ I'm experiencing internal processing issues. Please try again in a moment.",
            502: "🌐 There's a communication issue with my backend services. Please try again.",
            503: "🚧 My services are temporarily unavailable. Please try again in a few minutes."
        }
        
        base_message = error_messages.get(status_code, 
            f"🔧 I received an unexpected response (HTTP {status_code}). Please try again.")
        
        # Add specific error details if available
        if error_data.get("error"):
            base_message += f"\n\nDetails: {error_data['error']}"
            
        return base_message
    
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
            
            # Add hooks
            self.application.post_init = self.post_init
            self.application.post_shutdown = self.shutdown
            
            # Run the bot
            self.application.run_polling(
                drop_pending_updates=True,
                allowed_updates=Update.ALL_TYPES,
                # pool_timeout=30,
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
        # Validate environment
        validate_environment()
        
        # Configuration
        TOKEN = os.getenv('TELEGRAM_BOT_API_KEY')
        BOT_USERNAME = os.getenv('BOT_USERNAME', 'SuperAgentBot')
        BACKEND_URL = f"http://{os.getenv('SUPERAGENT_HOST', 'localhost')}:{os.getenv('SUPERAGENT_PORT', 5000)}/chat"
        
        print("🤖 Initializing SuperAgent Telegram Bot...")
        print(f"   Bot Username: {BOT_USERNAME}")
        print(f"   Backend URL: {BACKEND_URL}")
        
        # Create and run the bot
        bot = SuperAgentTelegramBot(
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
