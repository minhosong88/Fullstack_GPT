import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.embeddings import OllamaEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOllama
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory

st.set_page_config(
    page_title="FullStackGPT PrivateGPT",
    page_icon="🔒",
)


# Define a class for callback functions
class ChatCallbackHandler(BaseCallbackHandler):
    # Initialize a message variable
    message =""
    # When llm starts, an empty boc is created
    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    # When llm ends, save the created message
    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    # Each new token generated, the function is called
    def on_llm_new_token(self, token, *args, **kwargs):
        # append each token to the message variable
        self.message += token
        # each message appened will be shown to the message box
        self.message_box.markdown(self.message)

# Create an LLM
llm = ChatOllama(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ]
)

# Create a memory
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=120,
    memory_key="chat_history",
    return_messages=True,
)

# a function that return an embedded retriever
# Use 'cache_data' decorator not to run the function again if the file is the same as earlier
@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/private_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    # Set a path for file storage
    cache_dir = LocalFileStore(f"./.cache/private_embeddings/{file.name}")
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
    embedder = OllamaEmbeddings(
        model = "mistral:lastest"
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embedder, cache_dir
    )
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

# Save the message and memory to the session_state
def save_message(message, role):
    st.session_state["messages"].append(
        {"message": message, "role": role}
    )
def save_memory(input, output):
    st.session_state["chat_history"].append(
        {"input": input, "output": output}
    )
    
def send_message(message, role, save=True):
    # shows messages in the beginning, and save them
    with st.chat_message(role):
        st.markdown(message)
    if save:
        # Note that the messages are stored in a dictionary form
        save_message(message, role)
        
        
# Displaying messages without saving them: display saved messages
def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

def restore_memory():
    for history in st.session_state["chat_history"]:
        memory.save_context({"input": history["input"]}, {"output": history["output"]})

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

def load_memory(input):
    return memory.load_memory_variables({})["chat_history"]

def invoke_chain(message):
    # invoke the chain
    result = chain.invoke(message)
    # save the interaction in the memory
    save_memory(message, result.content)
    

prompt = ChatPromptTemplate.from_template([
     MessagesPlaceholder(variable_name="chat_history"),
     """
     Answer the question using ONLY the following context. If you don't know the answer, just say you don't know. DO NOT MAKE UP anything.
     Context:{context}
     Question:{question}
     """
])

st.title("DocumentGPT")

# Ask users to upload documents
st.markdown(
    """
    Welcome!
    
    Use this chatbot to ask questions to your AI about your documents
    
    Upload your files on the sidebar.
    """)
# create a file uploader in a side bar
with st.sidebar:
    file = st.file_uploader("Upload a .txt .pdf or .docx file", type=[
        "pdf", "txt", "docx"])

if file:
    # if file exists, retrieve and start creating messages.
    retriever = embed_file(file)
    send_message("I am ready. Ask away", "ai", save=False)
    # Restore memory and paint the history of previous chat
    restore_memory()
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        # Here, search for document(retriever), format the document(RunnableLambda(format_docs), RunnablePassthrough()=message), format the prompt(prompt), send the prompt to llm(llm)
        chain = {
            "context": retriever | RunnableLambda(format_docs),
            # sending the question straight to the prompt
            "chat_history": load_memory,
            "question": RunnablePassthrough(),
        } | prompt | llm
        with st.chat_message("ai"):
            invoke_chain(message)
else:
    # When there is no file(like in the beginning), initialize the session with a blank list
    st.session_state["messages"] = []
    st.session_state["chat_history"] =[]
