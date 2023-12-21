from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.storage import LocalFileStore
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
import streamlit as st
import re

llm = ChatOpenAI(
    temperature=0.1,
)

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context, answer the user's question. If you can't just say you don't know. DO NOT make anything up.
    
    Then, give a score to the answer between 0 and 5, 0 being not helpful and 5 being the most helpful to the user.
    
    Make sure to add scores to the answers
    
    Context: {context}

    Example: 
    Question: How far away is the moon?
    Answer: The moon is 384,000 km away
    Score: 5
    
    Question: How far away is the sum?
    Answer: I don't know
    Score: 0
    
    Your turn!
    QUestion: {question}
    """
)


def get_answers(inputs):
    docs = inputs['docs']
    question = inputs['question']
    answers_chain = answers_prompt | llm
    # answers = []
    # for doc in docs:
    #     result = answers_chain.invoke({
    #         "question": question,
    #         "context": doc.page_content,
    #     })
    #     answers.append(result)
    # Return a dictionary of a question and answers to send to choose_answer function
    return {"question": question,
            "answers": [
                {
                    "answer": answers_chain.invoke(
                        {
                            "question": question,
                            "context": doc.page_content,
                        }
                    ).content,
                    "source": doc.metadata["source"],
                    "date": doc.metadata["lastmod"],
                }
                for doc in docs
            ],
            }


choice_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """
         Use ONLY the following pre-existing answers to answer the user's question.
         
         Use the answers that have the highest score(more helpful) and favor the most recent one.
         
         Provide the sources. Return the source without any modification.
         
         Answer: {answers}
         """,
         ),
        ("human", "{question}")

    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choice_chain = choice_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n" for answer in answers)
    return choice_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


def parse_page(soup):
    # Currently using this to specifically scrape OpenAI site map
    header = soup.find("header")
    footer = soup.find("footer")
    regex = r"Authors(\w+\s\w*View all articles)+"
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    text = str(soup.get_text()).replace(
        "\n", " ").replace("\xa0", " ").replace("CloseSearch Submit Blog", "")
    result = re.sub(regex, "", text)
    return result


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    # Create a splitter to split documents
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800,
        chunk_overlap=200,
    )
    # sitemap loader to filter urls
    loader = SitemapLoader(
        url,
        # filter by regex
        # filter_urls=[
        #     r"^(.*\/blog\/).*",
        # ],
        parsing_function=parse_page
    )
    # Create a cached embeddings
    regex = re.compile(r"(www.)*\w+\.\w{2,3}")
    cache_dir = LocalFileStore(
        f"./.cache/site_embeddings/{re.search(regex, url).group()}")
    embedder = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embedder,
        cache_dir
    )
    loader.requests_per_second = 2
    # Create documents with splitter
    docs = loader.load_and_split(text_splitter=splitter)
    # Create a vector store with docs and embeddings
    vector_store = FAISS.from_documents(docs, cached_embeddings)
    # retriever has invoke() method, which can be used in chain
    return vector_store.as_retriever()


st.set_page_config(
    page_title="FullStackGPT SiteGPT",
    page_icon="💼",
)


st.title("SiteGPT")

st.markdown(
    """
Ask questions about the content of a website.

Start by writing the URL of the website on the sidebar.
""")

with st.sidebar:
    url = st.text_input("Write down a URL", placeholder="https://example.com")

if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write sitemap URL")
    else:
        retriever = load_website(url)
        query = st.text_input("Ask a question to the website:")
        if query:
            chain = ({"docs": retriever, "question": RunnablePassthrough(),
                      } | RunnableLambda(get_answers) | RunnableLambda(choose_answer)
                     )
            result = chain.invoke(query)
            st.write(result.content.replace("$", "\$"))
