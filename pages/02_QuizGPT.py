import streamlit as st
import json
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate

st.set_page_config(
    page_title="FullStackGPT QuizGPT",
    page_icon="💯",
)
st.title("QuizGPT")

quiz_schema = {
    "name": "generate_quiz",
    "description": "function that takes a list of questions and answers, then returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"]
                            },
                        },
                    },
                    "required": ["question", "answers"]
                },
            }
        },
    },
    "required": ["questions"],
}

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler()
    ],
).bind(
    function_call={
        "name": "generate_quiz",
    },
    functions=[
        quiz_schema,
    ]
)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


questions_prompt = PromptTemplate.from_template(
    "Based ONLY on the following context make 10 questions to test the user's knowledge about the text. Each question should have 4 answers, three of them must be incorrect and one should be correct. Context: {context}")

questions_chain = {
    # when invoke method takes docs, docs will be an argument of format_docs function, the result of which will be a string.
    "context": format_docs
} | questions_prompt | llm


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


@st.cache_data(show_spinner="Making quiz...")
# adding '_' says not to make a signature of docs. The functoin runs only once  gets the same results without additional parameters
# adding another parameter allows running function again when documents change
def run_quiz_chain(_docs, topic):
    response = questions_chain.invoke(_docs)
    return response


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(input):
    # You can change language by adding "lang=" at retriever
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(input)
    return docs


with st.sidebar:
    # Initialize docs, topic variable
    docs = None
    topic = None
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
            docs = wiki_search(topic)
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

    response = run_quiz_chain(docs, topic if topic else file.name)
    # For python to deal with the AI message chunk, convert it to json objet.
    response = json.loads(
        response.additional_kwargs["function_call"]["arguments"])
    # inside a form, streamlit waits to rerun until submitted
    with st.form("questions_form"):

        # Iterate each question in questions dictionary
        for question in response["questions"]:
            # paint options with st.radio. value is the option that the user chooses
            st.write(question["question"])
            value = st.radio("Select an option.", [answer["answer"]
                                                   for answer in question["answers"]], index=None)
            # checking if selected answer in the answers dictionary
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct")
            elif value is not None:
                st.error("Wrong")
        button = st.form_submit_button()
