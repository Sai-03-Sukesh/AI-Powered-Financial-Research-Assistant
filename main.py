import os
import streamlit as st
from groq import Groq
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.llms.base import LLM
from typing import Any, Dict, List, Optional

# Load environment variables
load_dotenv()

# Initialize Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Custom Groq LLM Wrapper
class GroqLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=1024,
            top_p=1,
            stream=False,
        )
        return completion.choices[0].message.content

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {}

# Initialize Groq LLM
llm = GroqLLM()

# Streamlit app
st.title("AI Article Analyzer(AAA)")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_groq"  # Corrected to directory name
main_placeholder = st.empty()

if process_url_clicked:
    try:
        # Validate URLs
        valid_urls = [url for url in urls if url.startswith(("http://", "https://")) and url]
        if not valid_urls:
            raise ValueError("No valid URLs provided")

        # Load data using WebBaseLoader
        loader = WebBaseLoader(valid_urls)
        main_placeholder.text("Data Loading...Started...✅✅✅")
        data = loader.load()

        # Split data
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        main_placeholder.text("Text Splitter...Started...✅✅✅")
        docs = text_splitter.split_documents(data)

        # Create embeddings and save it to FAISS index (using HuggingFaceEmbeddings)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore_groq = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Embedding Vector Started Building...✅✅✅")

        # Save the FAISS index
        vectorstore_groq.save_local(file_path)
        main_placeholder.success("Processing completed successfully!")

    except Exception as e:
        main_placeholder.error(f"Error: {str(e)}")

# Query handling
query = st.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        vectorstore = FAISS.load_local(file_path, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)

        st.header("Answer")
        st.write(result["answer"])

        # Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  # Split the sources by newline
            for source in sources_list:
                st.write(source)