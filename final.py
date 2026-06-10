import streamlit as st
import os
import time
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

os.environ['NVIDIA_API_KEY'] = os.getenv('Nvidia_API_key')

llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")


def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFLoader("./budget_speech.pdf")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)


st.title("Nvidia NIM Demo")

prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
</context>
Questions: {input}
""")

prompt1 = st.text_input("Enter Your Question From Documents")

if st.button("Document Embedding"):
    vector_embedding()
    st.write("FAISS Vector store DB is Ready Using NvidiaEmbeddings")

if prompt1:
    if "vectors" not in st.session_state:
        st.warning("Please click 'Document Embedding' first!")
    else:
        retriever = st.session_state.vectors.as_retriever()

        chain = (
            {"context": retriever, "input": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        start = time.process_time()
        response = chain.invoke(prompt1)
        st.write(f"Response Time: {time.process_time() - start:.2f}s")
        st.write(response)