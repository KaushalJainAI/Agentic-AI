#!/usr/bin/env python3
"""
Deployment script to run both Flask server and Telegram bot
"""

import os
import sys
import subprocess
import time
import signal
from multiprocessing import Process

def run_flask_server():
    """Run the Flask server"""
    print("🚀 Starting Flask server...")
    try:
        from host import SuperAgentFlaskApp, SuperAgentConfig
        app = SuperAgentFlaskApp()
        app.run(
            host=SuperAgentConfig.API_HOST,
            port=SuperAgentConfig.API_PORT,
            debug=SuperAgentConfig.DEBUG
        )
    except Exception as e:
        print(f"❌ Flask server error: {e}")

def run_telegram_bot():
    """Run the Telegram bot"""
    print("🤖 Starting Telegram bot...")
    # Wait a bit for Flask server to start
    time.sleep(3)
    try:
        subprocess.run([sys.executable, "connections.py"])
    except Exception as e:
        print(f"❌ Telegram bot error: {e}")

def signal_handler(sig, frame):
    """Handle shutdown signals"""
    print('\n🛑 Shutting down services...')
    sys.exit(0)

if __name__ == "__main__":
    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🌟 Starting SuperAgent System...")
    
    # Create processes
    flask_process = Process(target=run_flask_server)
    bot_process = Process(target=run_telegram_bot)
    
    try:
        # Start both services
        flask_process.start()
        bot_process.start()
        
        print("✅ All services started successfully!")
        print("📱 Telegram bot is ready to receive messages")
        print("🌐 Flask API is available at http://localhost:5000")
        
        # Wait for processes
        flask_process.join()
        bot_process.join()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        # Clean up processes
        if flask_process.is_alive():
            flask_process.terminate()
        if bot_process.is_alive():
            bot_process.terminate()
