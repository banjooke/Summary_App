import streamlit as st
from langchain_core.prompts import PromptTemplate
import langchain_community
from langchain_community.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from io import StringIO
from docx import Document
import fitz
import tiktoken

#LLM and key loading function
def load_LLM(openai_api_key, max_tokens):
    # Make sure your openai_api_key is set as an environment variable
    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key, max_tokens=max_tokens)
    return llm


#Page title and header
st.set_page_config(page_title="AI Long Text Summarizer")
st.header("AI Long Text Summarizer")


#Intro: instructions
col1, col2 = st.columns(2)

with col1:
    st.markdown("ChatGPT cannot summarize long texts. Now you can do it with this app.")

with col2:
    st.write("Contact with [AI Accelera](https://aiaccelera.com) to build your AI Projects")


#Input OpenAI API Key
st.markdown("## Enter Your OpenAI API Key")

def get_openai_api_key():
    input_text = st.text_input(label="OpenAI API Key ",  placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input", type="password")
    return input_text

openai_api_key = get_openai_api_key()


# Input
st.markdown("## Upload the file you want to summarize")

uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "docx"])

# Dropdown for summary length
length = st.selectbox("Choose summary length:", ["brief", "medium", "detailed"])

# Functions to read content from different file types
def read_txt(file):
    return file.read().decode("utf-8")

def read_docx(file):
    # Load the DOCX file and extract text from paragraphs
    doc = Document(file)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def read_pdf(file):
    # Read PDF content using PyMuPDF
    pdf_text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as pdf_doc:
        for page in pdf_doc:
            pdf_text += page.get_text()
    return pdf_text

if uploaded_file and openai_api_key:
    # Read content based on file type
    if uploaded_file.name.endswith(".txt"):
        uploaded_file = read_txt(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        uploaded_file = read_docx(uploaded_file)
    elif uploaded_file.name.endswith(".pdf"):
        uploaded_file = read_pdf(uploaded_file)

       
# Output
st.markdown("### Here is your Summary:")

if st.button("Summarize"):
    if uploaded_file is not None:

        if len(uploaded_file.split(" ")) > 20000:
            st.write("Please enter a shorter file. The maximum length is 20000 words.")
            st.stop()

        if uploaded_file:
            if not openai_api_key:
                st.warning('Please insert OpenAI API Key. \
                Instructions [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)', 
                icon="⚠️")
                st.stop()

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n"], 
            chunk_size=2000, 
            chunk_overlap=350
            )

        # Set max_tokens based on the selected summary length
        if length == "brief":
            max_tokens = 200  # Short summary
        elif length == "medium":
            max_tokens = 500  # Moderate-length summary
        else:
            max_tokens = 1000  # Detailed summary

        splitted_documents = text_splitter.create_documents([uploaded_file])

        llm = load_LLM(openai_api_key=openai_api_key, max_tokens=max_tokens)

        summarize_chain = load_summarize_chain(
            llm=llm, 
            chain_type="map_reduce"
            )

        summary_output = summarize_chain.run(splitted_documents)

        st.write(summary_output)
        st.write(len(summary_output.split()))