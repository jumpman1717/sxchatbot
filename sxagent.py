import os
import openai
import streamlit as st
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI
import nltk

# Define a custom NLTK data path to ensure resources are available
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Download necessary NLTK resources
nltk.download("punkt", download_dir=nltk_data_dir)
nltk.download("punkt_tab", download_dir=nltk_data_dir)

# Add the custom NLTK data path to NLTK's search paths
nltk.data.path.append(nltk_data_dir)

# For sentence splitting
nltk.download('punkt')

from nltk.tokenize import sent_tokenize

def chunk_text_by_sentence(text, max_words=200):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        if current_word_count + word_count > max_words:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = word_count
        else:
            current_chunk.append(sentence)
            current_word_count += word_count
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

import requests

def setup_rag_index(file_url="https://raw.githubusercontent.com/jumpman1717/sxchatbot/main/big.txt", model="gpt-4", temperature=0.1):
    """
    Sets up the RAG index using a file from a URL.
    """
    try:
        # Fetch the file content from the URL
        response = requests.get(file_url)
        response.raise_for_status()  # Raise an error for failed requests
        full_text = response.text  # Read the text content
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to fetch file from {file_url}: {str(e)}")

    # Chunk the text and create documents
    text_chunks = chunk_text_by_sentence(full_text, max_words=200)
    documents = [Document(text=chunk) for chunk in text_chunks]

    # Configure the LLM and create the index
    llm = OpenAI(model=model, temperature=temperature)
    Settings.llm = llm
    index = VectorStoreIndex.from_documents(documents)
    return index

def main():
    st.title("SX AI")
    st.write("This SX AI Agent is powered with preloaded documentation about SX and GPT-4")

    # Use the OpenAI API key from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OpenAI API key not found. Please set it as an environment variable.")
        return

    openai.api_key = openai_api_key

       # URL for the RAG file
    file_url = "https://raw.githubusercontent.com/jumpman1717/sxchatbot/main/big.txt"

    # Load the RAG index using the URL
    try:
        index = setup_rag_index(file_url)
    except Exception as e:
        st.error(f"Failed to load RAG document: {str(e)}")
        return

    # Chat interface
    st.write("### Chat Interface")
    user_input = st.text_area("Ask a question or request content:", height=150)

    if user_input:
        query_engine = index.as_query_engine()
        response = query_engine.query(user_input)
        st.write("### Response")
        st.code(str(response), language="text")

if __name__ == "__main__":
    main()
