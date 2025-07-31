from flask import Flask, redirect
from flask import request
from flask import jsonify
import os

from main import RemoteLLM

remote_llm = RemoteLLM()

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/callLLM", methods = ["POST", "GET"])
def call_LLM():
    res = remote_llm.prompt("Who is the president of the United States?")
    return f"<p>Calling deepseek! <br> {res} </p>"

@app.route("/automate", methods = ["POST", "GET"])
def automate():
    res = remote_llm.prompt("Write a script that could send an email to the recipient")
    return f"<p>Calling deepseek! <br> {res} </p>"


# @app.route("/automate", methods = ["POST", "GET"])
# def automate():
#     instruction = "Give the code to message my friend on whatsapp. His name is Anurag Sharma."
#     executed = remote_llm.CreateAUTOfile(instruction)
#     if executed:
#         print("Happy Birthday!")
#         return "<h1>We have successfully wished your friend!</h1>"        
    
#     return "<h1>The code Broke!</h1>"


if __name__ == "__main__":
    app.run(debug=True)


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