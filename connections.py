import os
import asyncio
import aiohttp
import logging
import sqlite3
from typing import Optional, Dict, Any
from datetime import datetime
from telegram import Update, Bot
from telegram.ext import (
    Application, CommandHandler, MessageHandler, 
    filters, ContextTypes
)
from telegram.constants import ChatAction
from dotenv import load_dotenv

load_dotenv()

class UserDatabase:
    """Simple SQLite database to store user information"""
    
    def __init__(self, db_path: str = "users.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize the database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    chat_id INTEGER PRIMARY KEY,
                    username TEXT,
                    first_name TEXT,
                    last_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
    
    def save_user(self, chat_id: int, username: str = None, first_name: str = None, last_name: str = None):
        """Save or update user information"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO users 
                (chat_id, username, first_name, last_name, last_active)
                VALUES (?, ?, ?, ?, ?)
            ''', (chat_id, username, first_name, last_name, datetime.now()))
            conn.commit()
    
    def get_all_users(self):
        """Get all users for broadcasting"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT chat_id, username FROM users')
            return [{"chat_id": row[0], "username": row[1]} for row in cursor.fetchall()]


class SuperAgentTelegramBot:
    """
    Enhanced Telegram bot with background processing and automatic responses
    """
    
    def __init__(self, name: str, token: str, backend_url: str, max_queue_size: int = 5):
        self.token = token
        self.name = name
        self.backend_url = backend_url
        self.max_queue_size = max_queue_size
        self.application = None
        self.bot = None
        self.db = UserDatabase()
        self._setup_logging()
        self._build_application()
        
        # Task management
        self.task_queue = asyncio.Queue(maxsize=max_queue_size)
        self.active_tasks = {}
        self.processing_users = set()  # Track users with active tasks
        
        # Background processing
        self.background_task = None
        
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
            self.bot = Bot(token=self.token)
            
            # Add command handlers
            self.application.add_handler(CommandHandler("start", self.start))
            self.application.add_handler(CommandHandler("help", self.help_command))
            self.application.add_handler(CommandHandler("status", self.status_command))
            self.application.add_handler(CommandHandler("queue", self.queue_status))
            
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
        # Save user to database
        user = update.effective_user
        self.db.save_user(
            chat_id=update.effective_chat.id,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name
        )
        async with aiohttp.ClientSession() as session:
            async with session.post(f"http://{os.getenv('SUPERAGENT_HOST')}:{os.getenv('SUPERAGENT_PORT')}/clearMem") as resp:
                result = await resp.json()
                self.logger.info(f"Memory cleared response: {result}")
        self.logger.info(f"User {user.id} started the bot. Previous Memory Cleared")
        
        welcome_message = (
            f"🤖 **Welcome to {self.name}!**\n\n"
            "I'm an intelligent AI assistant powered by multiple specialized agents. "
            "I can help you with:\n\n"
            "• 💬 General conversations and questions\n"
            "• 🌐 Web scraping and research\n"
            "• 📊 Database queries and analysis\n"
            "• 📁 Data processing tasks\n\n"
            "Just type your question or request, and I'll process it automatically!\n\n"
            "Type /help to see available commands."
        )
        await update.message.reply_text(welcome_message, parse_mode=None)
        self.logger.info(f"Start command used by user {update.effective_user.id}")

    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = (
            "🤖 **SuperAgent Commands:**\n\n"
            "• `/start` - Welcome message and introduction\n"
            "• `/help` - Show this help message\n"
            "• `/status` - Check system status\n"
            "• `/queue` - Check queue status\n\n"
            "💡 **How to use:**\n"
            "Simply type your message, and I'll automatically process it and respond!\n\n"
            "**Examples:**\n"
            "• \"Tell me about artificial intelligence\"\n"
            "• \"What's the weather like?\"\n"
            "• \"Help me with data analysis\""
        )
        await update.message.reply_text(help_text, parse_mode=None)
        self.logger.info(f"Help command used by user {update.effective_user.id}")
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        try:
            # Check if backend is accessible
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                health_url = f"{self.backend_url.replace('/chat', '/health')}"
                async with session.get(health_url) as response:
                    if response.status == 200:
                        status_message = "✅ **System Status: Online**\n\nAll services are running normally."
                    else:
                        status_message = f"⚠️ **System Status: Degraded**\n\nBackend returned status code: {response.status}"
        except Exception as e:
            status_message = "❌ **System Status: Offline**\n\nCannot connect to backend services."
            self.logger.error(f"Status check failed: {e}")
        
        await update.message.reply_text(status_message, parse_mode=None)
        self.logger.info(f"Status command used by user {update.effective_user.id}")
    
    async def queue_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /queue command - show current queue status"""
        queue_size = self.task_queue.qsize()
        active_tasks = len(self.active_tasks)
        
        status_message = (
            f"📊 **Queue Status:**\n\n"
            f"• Queue size: {queue_size}/{self.max_queue_size}\n"
            f"• Active tasks: {active_tasks}\n"
            f"• Processing users: {len(self.processing_users)}"
        )
        await update.message.reply_text(status_message, parse_mode=None)
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle regular text messages with immediate response and background processing"""
        if not update.message or not update.message.text:
            return
            
        user_message = update.message.text.strip()
        chat_id = update.message.chat_id
        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"
        
        # Save user to database
        self.db.save_user(
            chat_id=chat_id,
            username=username,
            first_name=update.effective_user.first_name,
            last_name=update.effective_user.last_name
        )
        
        self.logger.info(f"Message from user {user_id} (@{username}): {user_message[:100]}...")
        
        # Check if user already has an active task
        if user_id in self.processing_users:
            await update.message.reply_text(
                "⏳ I'm currently processing your previous request. Please wait for it to complete before sending a new one.",
                parse_mode=None
            )
            return
        
        # Check queue size
        if self.task_queue.qsize() >= self.max_queue_size:
            await update.message.reply_text(
                "🚦 The system is currently busy processing other requests. Please try again in a moment.",
                parse_mode=None
            )
            return
        
        # Send immediate acknowledgment
        await update.message.reply_text(
            "🤖 I've received your request and started processing it. I'll send you the response automatically when it's ready!",
            parse_mode=None
        )
        
        # Send typing action
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        
        # Create task data
        task_data = {
            "message": user_message,
            "chat_id": chat_id,
            "user_id": user_id,
            "username": username,
            "timestamp": datetime.now()
        }
        
        # Add to processing users set
        self.processing_users.add(user_id)
        
        # Add task to queue
        try:
            await self.task_queue.put(task_data)
            self.logger.info(f"Task queued for user {user_id}")
        except Exception as e:
            self.logger.error(f"Failed to queue task for user {user_id}: {e}")
            self.processing_users.discard(user_id)
            await context.bot.send_message(
                chat_id=chat_id,
                text="❌ Sorry, I couldn't queue your request. Please try again.",
                parse_mode=None
            )
    
    async def background_processor(self):
        """Background task processor that handles queued tasks"""
        self.logger.info("Background processor started")
        
        while True:
            try:
                # Get task from queue
                task_data = await self.task_queue.get()
                
                chat_id = task_data["chat_id"]
                user_id = task_data["user_id"]
                user_message = task_data["message"]
                username = task_data["username"]
                
                self.logger.info(f"Processing task for user {user_id}")
                
                # Process the task
                try:
                    reply = await self._process_with_backend(user_message, user_id, username)
                    
                    # Send automatic response
                    await self._send_automatic_response(chat_id, reply)
                    
                except Exception as e:
                    self.logger.error(f"Error processing task for user {user_id}: {e}")
                    error_reply = "❌ Sorry, I encountered an error processing your request. Please try again."
                    await self._send_automatic_response(chat_id, error_reply)
                
                finally:
                    # Remove from processing users
                    self.processing_users.discard(user_id)
                    # Mark task as done
                    self.task_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Background processor error: {e}")
                await asyncio.sleep(1)  # Wait before retrying
    
    async def _process_with_backend(self, message: str, user_id: int, username: str) -> str:
        """Process message with backend API"""
        payload = {
            "message": message,
            "userId": str(user_id),
            "username": username
        }
        
        timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes timeout
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self.backend_url,  # Using /chat endpoint directly
                json=payload,
                headers={'Content-Type': 'application/json'}
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    reply = data.get("reply", "I processed your request but couldn't generate a response.")
                    success = data.get("success", False)
                    metadata = data.get("metadata", {})
                    
                    # # Add metadata info for transparency
                    # if success and metadata.get("agent_used"):
                    #     agent_used = metadata["agent_used"]
                    #     reply += f"\n\n_Processed by: {agent_used}_"
                    
                    return reply
                    
                else:
                    if response.headers.get("Content-Type", "").startswith("application/json"):
                        error_data = await response.json()
                    else:
                        error_data = {}
                    return self._get_error_message(response.status, error_data)
    
    async def _send_automatic_response(self, chat_id: int, message: str):
        """Send automatic response to user"""
        print(message)
        try:
            # Ensure message isn't too long for Telegram
            if len(message) > 4000:
                message = message[:4000] + "...\n\n📝 _(Message truncated due to length)_"
            
            await self.bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode=None
            )
            
            self.logger.info(f"Automatic response sent to chat {chat_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send automatic response to chat {chat_id}: {e}")
            return "❌ Failed to send automatic response. Please try again."

        
    async def broadcast_message(self, message: str):
        """Send message to all users (example of background messaging)"""
        users = self.db.get_all_users()
        
        for user in users:
            try:
                await self.bot.send_message(
                    chat_id=user["chat_id"],
                    text=message,
                    parse_mode=None
                )
                await asyncio.sleep(0.1)  # Rate limiting
            except Exception as e:
                self.logger.error(f"Failed to send broadcast to {user['chat_id']}: {e}")
    
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
        
        if error_data.get("error"):
            base_message += f"\n\nDetails: {error_data['error']}"
            
        return base_message
    
    async def post_init(self, application: Application) -> None:
        """Post initialization hook"""
        self.logger.info(f"Bot {self.name} initialized successfully")
        # Start background processor
        self.background_task = asyncio.create_task(self.background_processor())
    
    async def shutdown(self, application: Application) -> None:
        """Graceful shutdown hook"""
        self.logger.info(f"Bot {self.name} is shutting down...")
        if self.background_task:
            self.background_task.cancel()
            try:
                await self.background_task
            except asyncio.CancelledError:
                pass
    
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


# Example usage for scheduled/background messaging
async def scheduled_message_example():
    """Example function for sending scheduled messages"""
    # This would be called by a scheduler (cron job, APScheduler, etc.)
    TOKEN = os.getenv('TELEGRAM_BOT_API_KEY')
    bot = Bot(token=TOKEN)
    db = UserDatabase()
    
    users = db.get_all_users()
    message = "🌅 Good morning! This is your daily automated message."
    
    for user in users:
        try:
            await bot.send_message(
                chat_id=user["chat_id"],
                text=message,
                parse_mode=None
            )
            await asyncio.sleep(0.1)  # Rate limiting
        except Exception as e:
            print(f"Failed to send to {user['chat_id']}: {e}")


if __name__ == '__main__':
    try:
        # Validate environment
        validate_environment()
        
        # Configuration
        TOKEN = os.getenv('TELEGRAM_BOT_API_KEY')
        BOT_USERNAME = os.getenv('BOT_USERNAME', 'SuperAgentBot')
        BACKEND_URL = f"http://{os.getenv('SUPERAGENT_HOST', 'localhost')}:{os.getenv('SUPERAGENT_PORT', 5000)}/chat"
        MAX_QUEUE_SIZE = int(os.getenv('MAX_QUEUE_SIZE', 5))
        
        print("🤖 Initializing SuperAgent Telegram Bot...")
        print(f"   Bot Username: {BOT_USERNAME}")
        print(f"   Backend URL: {BACKEND_URL}")
        print(f"   Max Queue Size: {MAX_QUEUE_SIZE}")
        
        # Create and run the bot
        bot = SuperAgentTelegramBot(
            name=BOT_USERNAME,
            token=TOKEN,
            backend_url=BACKEND_URL,
            max_queue_size=MAX_QUEUE_SIZE
        )
        
        bot.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        logging.error(f"❌ Fatal error: {e}")
        raise
