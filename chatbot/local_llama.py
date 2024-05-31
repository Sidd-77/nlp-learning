from langchain_community.llms import ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

## Prompt Template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant which will answer user quries."),
        ("user","Question:{question}")
    ]
)

## Streamlit UI

st.title("LangChain Ollama Demo")
input_text = st.text_input("Enter your question here:")

## Gemini LLM
llm = ChatOllama(model='llama3')
output_parser = StrOutputParser()
chain = prompt|llm|output_parser


if input_text:
    st.write(chain.invoke({'question':input_text}))

