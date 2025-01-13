import os
import openai
import streamlit as st
from nltk.tokenize import sent_tokenize
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI
import nltk
nltk.download('punkt')


# For sentence splitting
nltk.download('punkt')

# Helper function for text chunking
def chunk_text_by_sentence(text, max_words=200):
    """
    Splits 'text' into sentence-based chunks. Each chunk will have 
    up to 'max_words' words, so we don't cut off mid-sentence.
    """
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

# Set up RAG index
def setup_rag_index(file, model="gpt-4", temperature=0.1):
    """
    Sets up the RAG index using the uploaded file.
    """
    # Read the content of the UploadedFile object
    full_text = file.getvalue().decode("utf-8")
    
    # Chunk the text and create documents
    text_chunks = chunk_text_by_sentence(full_text, max_words=200)
    documents = [Document(text=chunk) for chunk in text_chunks]

    # Configure the LLM and create the index
    llm = OpenAI(model=model, temperature=temperature)
    Settings.llm = llm
    index = VectorStoreIndex.from_documents(documents)
    return index

# Streamlit app
def main():
    st.title("RAG-Powered Chatbot for Content Writing")
    st.write(
        "This chatbot is designed to help you write content based on your project-specific knowledge."
    )

    # Input field for OpenAI API key
    openai_api_key = st.text_input(
        "Enter your OpenAI API Key:", type="password"
    )

    # File uploader for the project-specific document
    input_file = st.file_uploader(
        "Upload your project-specific document (text file):", type="txt"
    )

    if input_file and openai_api_key:
        # Set OpenAI API key
        openai.api_key = openai_api_key

        # Set up RAG index
        index = setup_rag_index(input_file)

        # Chat interface
        st.write("### Chat Interface")
        user_input = st.text_area(
            "Ask a question or request content:", 
            height=150  # Adjust height as needed
        )

        if user_input:
            query_engine = index.as_query_engine()
            response = query_engine.query(user_input)
            st.write("### Response")
            st.code(str(response), language="text")
    elif not input_file:
        st.warning("Please upload a text file.")
    elif not openai_api_key:
        st.warning("Please enter your OpenAI API key.")

if __name__ == "__main__":
    main()
