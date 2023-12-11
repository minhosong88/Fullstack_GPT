import streamlit as st
from langchain.prompts import ChatPromptTemplate
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
# Use 'cache_data' decorator not to run the function again if the file is the same as earlier


@st.cache_data(show_spinner="Embedding file...")
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
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=char_splitter)
    embedder = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embedder, cache_dir
    )
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        # Note that the messages are stored in a dictionary form
        st.session_state["messages"].append(
            {"message": message, "role": role})


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


template = ChatPromptTemplate.from_messages([
    ("system",
     """
     Answer the question using ONLY the following context. If you don't know the answer, just say you don't know. DO NOT MAKE UP anything.
     Context:{context}
     """),
    ("human", "{question}")
])

st.title("DocumentGPT")

# Ask users to upload documents
st.markdown(
    """
    Welcome!
    
    Use this chatbot to ask questions to your AI about your documents
    
    Upload your files on the sidebar.
    """)
# create a file uploader
with st.sidebar:
    file = st.file_uploader("Upload a .txt .pdf or .docx file", type=[
        "pdf", "txt", "docx"])

if file:
    # if file exists, retrieve and start creating messages.
    retriever = embed_file(file)
    send_message("I am ready. Ask away", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        docs = retriever.invoke(message)
        st.write(docs)
        # join the documents into one string
        docs = "\n\n".join(document.page_content for document in docs)
        prompt = template.format_messages(context=docs, question=message)
        st.write(prompt)

else:
    # When there is no file, initialize the session
    st.session_state["messages"] = []
