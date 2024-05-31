from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain_community.chat_models import ChatOllama
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import ollama
from dotenv import load_dotenv
load_dotenv()



app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="Just a simple fukin server"
)

Gemini = GoogleGenerativeAI(model="gemini-1.5-pro-latest")
Ollama = ChatOllama(model='llama3')

prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic}")
prompt2 = ChatPromptTemplate.from_template("Write me a poem about {topic}")

add_routes(
    app,
    prompt1 | Gemini,
    path="/essay"
)

add_routes(
    app,
    prompt2 | Ollama,
    path="/poem"
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)