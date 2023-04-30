import streamlit as st
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import textwrap
import validators
import os

# loading dotenv
from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv())

# setting up the app -----------------------------------
st.set_page_config(page_title="LangChain Quickstart", page_icon="snake")
# Force responsive layout for columns also on mobile
st.write(
    """<style>
    [data-testid="column"] {
        width: calc(50% - 1rem);
        flex: 1 1 calc(50% - 1rem);
        min-width: calc(50% - 1rem);
    }
    </style>""",
    unsafe_allow_html=True,
)
# render Page
st.title("🦜️🔗 Youtube Transcript Analyzer")
st.markdown(
    "This mini-app generates Transcripts using OpenAI's GPT-3 based [Turbo model](https://beta.openai.com/docs/models/overview). You can find the code on [GitHub](https://github.com/virajsabhaya23/auto_YT) and the author on [LinkedIn](https://www.linkedin.com/in/vsabhaya23/)."
)

# API key variable
openai_api_key = st.text_input("Enter your OpenAI API key", type="password", placeholder="sk-...")
os.environ["OPENAI_API_KEY"] = openai_api_key
if not openai_api_key:
    st.warning("Please enter your OpenAI API key.")
    st.stop()

embeddings = OpenAIEmbeddings(openai_api_key= openai_api_key)


def app():
    # INPUT variables
    video_url = st.text_input(label="Enter the YouTube video URL", placeholder="https://www.youtube.com/watch?v=L7J3aSGP0s4&t=1s&ab_channel=ITLab-UTA")

    if not validators.url(video_url):
        st.text_error = "Invalid URL. Please enter a valid URL."
        return

    query = st.text_input(label="Ask your question about the video", placeholder="Describe video in 1 sentence or 10 words")

    if st.button(label="Get Answer"):
        db = yt_loader(video_url)
        response, docs = reponse_fromQuery(db, query, openai_api_key)

        with st.spinner("Generating response..."):
            st.markdown("""---""")
        # st.text_area(label="Answer", value=response, height=200)
        with st.container():
            with st.expander("ANSWER >", expanded=True):
                st.info(textwrap.fill(response, width=80))

def yt_loader(video_url):
    #import the YT URL to loader variable
    loader = YoutubeLoader.from_youtube_url(video_url)

    # get the transcript
    transcript = loader.load()

    # transcirpt return tokens in large text format.
    # To only use the tokens as input params for the model,
    # we use text splittles, that feeds chunks of data to the model.

    # text splitter divides the text into chunks of 1000 characters
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)
    # print(docs)

    db = FAISS.from_documents(docs, embeddings)
    return db

def reponse_fromQuery(db, query, openai_api_key, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

    template = """
        You are a helpful assistant that that can answer questions about youtube videos
        based on the video's transcript: {docs}

        Only use the factual information from the transcript to answer the question.

        If you feel like you don't have enough information to answer the question, say "I don't know".

        Your answers should be verbose, and detailed.
    """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs

if __name__ == '__main__':
    app()
