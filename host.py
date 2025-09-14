from flask import Flask, request, jsonify
import os
from agents import Chatbot
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
os.environ["GEMINI_API_KEY"] = api_key

chatbot = Chatbot(api_key=api_key)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/chat", methods=["POST"])  # Changed from GET to POST
def chat():
    data = request.json
    user_message = data.get("message", "")
    user_id = data.get("userId", "")  # Get userId from request
    
    if not user_message:
        return jsonify({"reply": "No message provided."}), 400
    
    try:
        bot_reply = chatbot.chat(user_message)
        return jsonify({"reply": bot_reply})
    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)



# import os
# # from google import genai
# import re
# from openai import OpenAI 
# from dotenv import load_dotenv
# load_dotenv()


# # key = os.getenv("GEMINI_API_KEY")
# # client_gemini = genai.Client(key=key)

# deep_seek_key = os.getenv("DEEPSEEK_API_KEY")

# open_router_key = os.getenv("OPEN_ROUTER_KEY")
# open_router_url = "https://openrouter.ai/api/v1"

# # client = OpenAI(api_key=open_router_key, base_url=open_router_url)



# class RemoteLLM:
#     def __init__(self, model_name = "deepseek/deepseek-chat-v3-0324:free" , key = open_router_key):
#         self.model_name = model_name
#         self.llm = OpenAI(api_key=key, base_url=open_router_url)
#         self.requests = []

#     def prompt(self, query):
#         self.requests.append(query)
#         response = self.llm.chat.completions.create(
#             model = self.model_name,
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant"},
#                 {"role": "user", "content": query},
#                 ],
#             stream=False
#             )
#         print(response.choices[0].message.content)
#         return response.choices[0].message.content