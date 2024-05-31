import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time


if "vector" not in st.session_state:
    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    st.session_state.loader = WebBaseLoader(web_path='https://python.langchain.com/v0.2/docs/concepts/')
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(st.session_state.docs)
    st.session_state.vectors = db = FAISS.from_documents(st.session_state.documents, st.session_state.embeddings)


st.title("GROQ Chat Demo")

llm = ChatGroq(model='llama3-8b-8192')

prompt = ChatPromptTemplate.from_template(
    """ 
    Answer the following questions based on given context only.
    Provide the most accurate response based on the context and question.
    <context>
    {context}
    </context>
    Question: {input}
"""
)

document_chain = create_stuff_documents_chain(llm, prompt)
retirver = st.session_state.vectors.as_retriever()
retireval_chain = create_retrieval_chain(retirver, document_chain)

input_text = st.text_input("Enter your question here")

if input_text:
    start = time.process_time()
    response = retireval_chain.invoke({"input": input_text})
    st.write("Response time :", time.process_time()-start)

    st.write("Answer:")
    st.write(response['answer'])


    with st.expander("Context"):
        st.write(response['context'])
