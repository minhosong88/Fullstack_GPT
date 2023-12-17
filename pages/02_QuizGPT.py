import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts import ChatPromptTemplate

st.title("QuizGPT")

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler()
    ],
)


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    # Set a path for file storage
    char_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        # put separator
        separator="\n",
        # set a max number of characters
        chunk_size=600,
        chunk_overlap=100,
        # LLM does not count token by the length of text.
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=char_splitter)
    return docs


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


with st.sidebar:
    # Initialize docs variable
    docs = None
    # Create a selectbox
    choice = st.selectbox("Choose what you wnat to use", (
        "File", "Wikipedia Article",
    ),
    )
    # if File is selected, present an uploader
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx, .txt or .pdf file",
            type=["pdf", "txt", "docx"],
        )
        # Once file is selected, split the texts in the file
        if file:
            docs = split_file(file)
    # else, present a search box for Wikipedia articles
    else:
        topic = st.text_input("Search Wikipedia articles:")
        if topic:
            # You can change language by adding "lang=" at retriever
            retriever = WikipediaRetriever(top_k_results=5)
            with st.status("Searching Wikipedia"):
                docs = retriever.get_relevant_documents(topic)

# Initialize a front page
if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
    
    I will make a quiz from the files you upload or Wikipedia articles to test your knowledge and help you study.
    
    Get started by uploading a file or searching on Wikipedia in the sidebar
    """
    )
else:
    prompt = ChatPromptTemplate.from_messages(
        [   # Examples are formed to show AI that the answer could be either first, second, third or fourth. TO give all possibilities
            ("system", """
             You are a helpful assistant that is role playing as a teacher.
             Based ONLY on the following context make 10 questions to test the user's knowledge about the text.
             Each question should have 4 answers, three of them must be incorrect and one should be correct.
             
             Use (o) to signal the correct answer.
             
             Question examples:
             Question: What is the color of the ocean?
             Answers: Red| Yellow| Green| Blue(o)
             
             Question: What is the capital of Georgia?
             Answers: Baku| Tbilisi(o)| Manilla| Beirut
             
             Question: When was Avatar released?
             Answers: 2007| 2008| 2009(o)| 1998
             
             Question: WHo was Julius Caesar?
             Answers: A Roman Emperor(o)| Painter| Actor| Model
             
             Your turn!
             
             Context: {context}
             """),
        ]
    )

    chain = {
        # when invoke method takes docs, docs will be an argument of format_docs function, the result of which will be a string.
        "context": format_docs
    } | prompt | llm

    start = st.button("Generate Quiz")
    if start:
        chain.invoke(docs)
