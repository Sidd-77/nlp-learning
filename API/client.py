import requests 
import streamlit as st

def get_gemini_response(input_txt: str) -> str:
    response = requests.post(
        'http://localhost:8000/essay/invoke',
        json={'input':{'topic':input_txt}},
        )
    return response.json()['output']


def get_ollama_response(input_txt: str) -> str:
    response = requests.post(
        'http://localhost:8000/poem/invoke',
        json={'input':{'topic':input_txt}},
        )
    return response.json()['output']["content"]



## Streamlit UI

st.title("Langchain API Demo with Langserver")

input_text1 = st.text_input("Write essay on ...")
input_text2 = st.text_input("Write poem on ...")

if input_text1:
    st.write(get_gemini_response(input_text1))

if input_text2:
    st.write(get_ollama_response(input_text2))