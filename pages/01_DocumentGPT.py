import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores.faiss import FAISS
from dotenv import dotenv_values

st.set_page_config(
    page_title="FullStackGPT DocumentGPT",
    page_icon="📄",
)

# a function that return an embedded retriever


def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    # Set a path for file storage
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    char_splitter = CharacterTextSplitter(
        # put separator
        separator="\n",
        # set a max number of characters
        chunk_size=600,
        chunk_overlap=100,
        # count length of the text by using len function by default
        length_function=len,
        # LLM does not count token by the length of text.
    )
    loader = UnstructuredFileLoader("./files/chapter_one.txt")
    docs = loader.load_and_split(text_splitter=char_splitter)
    embedder = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embedder, cache_dir
    )
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


st.title("DocumentGPT")

# Ask users to upload documents
st.markdown(
    """
    Welcome!
    
    Use this chatbot to ask questions to your AI about your documents
    """)
# create a file uploader
file = st.file_uploader("Upload a .txt .pdf or .docx file", type=[
                        "pdf", "txt", "docx"])

if file:
    retriever = embed_file(file)
    s =retriever.invoke("Adam")
    s