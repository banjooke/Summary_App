import streamlit as st
from langchain.llms import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from docx import Document
import fitz  # PyMuPDF for handling PDF files
import tiktoken  # Token counting library

# Load the LLM with OpenAI API and dynamic max_tokens
def load_llm(api_key, max_tokens):
    return OpenAI(temperature=0.7, openai_api_key=api_key, max_tokens=max_tokens)

# Define the prompt template for summarizing text with a length parameter
summary_prompt = PromptTemplate(
    input_variables=["text", "length"],
    template="Summarize the following text in a {length} manner:\n\n{text}"
)

# Function to split text using RecursiveCharacterTextSplitter
def split_text_recursively(text, chunk_size=1000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)

# Functions to read content from different file types
def read_txt(file):
    return file.read().decode("utf-8")

def read_docx(file):
    doc = Document(file)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def read_pdf(file):
    pdf_text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as pdf_doc:
        for page in pdf_doc:
            pdf_text += page.get_text()
    return pdf_text

# Token calculation function using tiktoken
def calculate_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Adjust the model name as needed
    tokens = encoding.encode(text)
    return len(tokens)

# Streamlit app layout
st.title("File Summarizer with Recursive Text Splitter and Length-based Limits")
st.write("Upload a .txt, .pdf, or .docx file, choose a summary length, and click 'Summarize' to generate the summary.")

# OpenAI API key input
openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")

# File upload section accepting txt, pdf, and docx files
uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "docx"])

# Dropdown for summary length
length = st.selectbox("Choose summary length:", ["brief", "medium", "detailed"])

# Set max_tokens based on the selected summary length
if length == "brief":
    max_tokens = 50  # Short summary
elif length == "medium":
    max_tokens = 150  # Moderate-length summary
else:
    max_tokens = 300  # Detailed summary

# Split document content into manageable chunks when file is uploaded
if uploaded_file and openai_api_key:
    # Read content based on file type
    if uploaded_file.name.endswith(".txt"):
        content = read_txt(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        content = read_docx(uploaded_file)
    elif uploaded_file.name.endswith(".pdf"):
        content = read_pdf(uploaded_file)
    
    # Split the content into chunks and display them
    chunks = split_text_recursively(content, chunk_size=4000, chunk_overlap=100)
    st.write(f"Document split into {len(chunks)} chunks.")
    st.write("Click 'Summarize' to process the chunks.")

    # Summarize when the "Summarize" button is clicked
    if st.button("Summarize"):
        # Load the LLM with dynamic max_tokens based on length
        llm = load_llm(openai_api_key, max_tokens)

        summaries = []
        for chunk in chunks:
            # Calculate the token count of the prompt
            prompt = summary_prompt.format(text=chunk, length=length)
            input_tokens = calculate_tokens(prompt)

            # Check if the total tokens exceed the model limit
            if input_tokens + max_tokens > 4096:  # Adjust based on the model's limit
                st.warning("Chunk is too large, reducing size...")
                # Optionally trim the chunk or modify your logic here
                chunk = chunk[:4000]  # This is just an example; adjust as needed

            summary = llm(prompt)
            summaries.append(summary)

        # Combine chunk summaries into a final summary
        final_summary = " ".join(summaries)

        # Display the final summary
        st.subheader("Final Summary")
        st.write(final_summary)

        # Download option for the final summary
        st.download_button("Download Summary", final_summary, "summary.txt")
