import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from docx import Document
import fitz  # PyMuPDF for handling PDF files

# Load the LLM with OpenAI API
def load_llm(api_key, max_tokens):
    return OpenAI(temperature=0.7, openai_api_key=api_key, max_tokens=max_tokens)

# Function to read content from different file types
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

# Function to limit summary to a specific word count
def limit_word_count(text, max_words):
    words = text.split()
    return ' '.join(words[:max_words])

# Streamlit app layout
st.title("File Summarizer with Adjustable Summary Length")
st.write("Upload a .txt, .pdf, or .docx file, choose a summary length, and click 'Summarize' to generate the summary.")

# OpenAI API key input
openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")

# File upload section accepting txt, pdf, and docx files
uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "docx"])

# Dropdown for summary length
length = st.selectbox("Choose summary length:", ["brief", "medium", "detailed"])

# Split document content into manageable chunks when file is uploaded
if uploaded_file and openai_api_key:
    # Read content based on file type
    if uploaded_file.name.endswith(".txt"):
        content = read_txt(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        content = read_docx(uploaded_file)
    elif uploaded_file.name.endswith(".pdf"):
        content = read_pdf(uploaded_file)
    
    # Split the content into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(content)
    st.write(f"Document split into {len(chunks)} chunks.")
    st.write("Click 'Summarize' to process the chunks.")

    # Set max_tokens based on the selected summary length
    if length == "brief":
        max_tokens = 50  # Short summary
        max_words = 30   # Enforced word count limit
    elif length == "medium":
        max_tokens = 150  # Moderate-length summary
        max_words = 70    # Enforced word count limit
    else:
        max_tokens = 300  # Detailed summary
        max_words = 150   # Enforced word count limit

    # Summarize when the "Summarize" button is clicked
    if st.button("Summarize"):
        # Load the LLM with the specified max_tokens
        llm = load_llm(openai_api_key, max_tokens)

        # Summarized texts storage
        summaries = []

        # Iterate over each chunk to summarize
        for chunk in chunks:
            # Create the prompt
            prompt = f"Summarize the following text:\n{chunk}"
            summary = llm(prompt)  # Get summary from the model
            
            # Limit the summary to the specified word count
            limited_summary = limit_word_count(summary, max_words)
            summaries.append(limited_summary)

        # Combine summaries into a final output
        final_summary = "\n".join(summaries)

        # Display the final summary
        st.subheader("Final Summary")
        st.write(final_summary)

        # Download option for the final summary
        st.download_button("Download Summary", final_summary, "summary.txt")
