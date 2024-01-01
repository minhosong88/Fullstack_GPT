import streamlit as st
import subprocess
from pydub import AudioSegment
import math
import glob
import openai
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
from langchain.storage import LocalFileStore
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

# Define a class for callback functions
class ChatCallbackHandler(BaseCallbackHandler):
    # Initialize a message variable
    message =""
    # When llm starts, an empty box is created
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
llm = ChatOpenAI(
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


# Not to create the same transcript again when there is one.
has_transcript = os.path.exists("./.cache/podcast.txt")

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=800,
                chunk_overlap=100,
            )
@st.cache_data()
def embed_file(file_path):
    # Set a path for file storage
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embedder = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embedder, cache_dir
    )
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

@st.cache_data()
def transcribe_chunks(chunk_folder, destination):
    if has_transcript:
        return
    # Grab the files with glob module
    files = glob.glob(f"{chunk_folder}/*.mp3")
    # Sort the files in an ascending order because glob gets files in a random order
    files.sort()
    # Iterate files, open and transcribe into a text file
    for file in files:
        with open(file, "rb") as audio_file, open(destination, "a") as text_file:
            # Transcribe the text
            transcript = openai.Audio.transcribe(
                "whisper-1",
                audio_file,
            )
            # Write the transcript in the destination path
            text_file.write(transcript["text"])


@st.cache_data()
def extract_audio_from_video(video_path):
    if has_transcript:
        return
    audio_path = video_path.replace("mp4", "mp3")
    command = ['ffmpeg', '-y', '-i', video_path, "-vn", audio_path]
    subprocess.run(command)

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


@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    if has_transcript:
        return
    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size*60*1000
    chunks = math.ceil(len(track)/chunk_len)
    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i+1) * chunk_len

        chunk = track[start_time:end_time]
        chunk.export(f"{chunks_folder}/chunk_{i}.mp3", format="mp3")


st.set_page_config(
    page_title="MeetingGPT",
    page_icon="🧑‍💻"
)
st.title("MeetingGPT")
st.markdown(
    """
    Welcome to MeetingGPT, upload a video and I will give you a transcript, a summary and a chat bot to ask any questions about it.
    
    Get started by uploading a video file in the side bar
    """
)

with st.sidebar:
    video = st.file_uploader("Video", type=["mp4", "avi", "mkv", "mov"])

if video:
    chunks_folder = "./.cache/chunks"
    with st.status("Loading video...") as status:
        video_content = video.read()
        video_path = f"./.cache/{video.name}"
        audio_path = video_path.replace("mp4", "mp3")
        transcript_path = video_path.replace("mp4", "txt")
        # Open the file from the path as writing binary mode
        with open(video_path, "wb") as f:
            f.write(video_content)
        status.update(label="Extracting audio...")
        extract_audio_from_video(video_path)
        status.update(label="Cutting audio segments...")
        cut_audio_in_chunks(audio_path, 10, chunks_folder)
        status.update(label="Transcribing audio...")
        transcribe_chunks(chunks_folder, transcript_path)

    transcript_tab, summary_tab, qa_tab = st.tabs(
        [
            "Transcrript",
            "Summary",
            "Q&A",
        ]
    )

    with transcript_tab:
        with open(transcript_path, "r") as file:
            st.write(file.read())
    with summary_tab:
        generate = st.button("Generate summary")

        if generate:
            loader = TextLoader(transcript_path)
            docs = loader.load_and_split(text_splitter=splitter)
            
            first_summary_prompt = ChatPromptTemplate.from_template(
                """
                Write a concise summary of the following:
                "{text}"
                CONCISE SUMMARY:
                """
            )
            first_summary_chain = first_summary_prompt|llm|StrOutputParser()
            # Create an initial summary
            summary = first_summary_chain.invoke({
                "text":docs[0].page_content
            })

            refine_prompt = ChatPromptTemplate.from_template(
                """
                Your job is to produce a final summary.
                We have provided an existing summary up to a certain point:{existing_summary}
                We have the opportunity to refine the existing summary(Only if needed) with some more context below.
                -------------------
                {context}
                -------------------
                Given the new context, refine the original summary.
                If the context isn't useful, RETURN the original summary
                """
            )
            
            refine_chain =  refine_prompt|llm|StrOutputParser()
            # Iterate the documents based on the initial summary
            with st.status("Summarizing...") as status:
                for idx, doc in enumerate(docs[1:]):
                    status.update(label=f"Processing document # {idx+1} / {len(docs[1:])}")
                    # Keep updating summary based on the previous summary
                    summary = refine_chain.invoke({
                        "existing_summary": summary,
                        "context": doc.page_content,
                    })
            st.write(summary)
            
    with qa_tab:
        retriever = embed_file(transcript_path)
        prompt = ChatPromptTemplate.from_messages([
        ("system",
        """
        Answer the question using ONLY the following context. If you don't know the answer, just say you don't know. DO NOT MAKE UP anything.
        Context:{context}
        """),
        # passing chat history to the template? =>message placeholder
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
        ])
        send_message("Ask anything about the video", "ai", save=False)
        restore_memory()
        paint_history()
        question = st.chat_input("Type your question here.")
        if question:
            send_message(question, "human")
            chain = {
            "context": retriever | RunnableLambda(format_docs),
            # sending the question straight to the prompt
            "chat_history": load_memory,
            "question": RunnablePassthrough(),
            } | prompt | llm
            with st.chat_message("ai"):
                invoke_chain(question)
        else:
            # When there is no file(like in the beginning), initialize the session with a blank list
            st.session_state["messages"] = []
            st.session_state["chat_history"] =[]