# from 


# class TelegramBot(API_ID, API_HASH, BOT_TOKEN):
#     def __init__(self):
#         self.api_id = API_ID
#         self.api_hash = API_HASH
#         self.bot_token = BOT_TOKEN
#         self.app = None

#     async def start(self):
#         self.app = Client("my_bot", api_id=self.api_id, api_hash=self.api_hash, bot_token=self.bot_token)
#         await self.app.start()
#         print("Bot started!")

#     async def stop(self):
#         if self.app:
#             await self.app.stop()
#             print("Bot stopped!")

#     async def send_message(self, chat_id: int, message: str):
#         if self.app:
#             await self.app.send_message(chat_id, message)
#         else:
#             print("Bot not started. Cannot send message.")

#     async def get_me(self):
#         if self.app:
#             return await self.app.get_me()
#         else:
#             print("Bot not started. Cannot get bot info.")
#             return None