import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from PIL import Image
import asyncio


# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


# Cache the PDF loading function to avoid reprocessing the same PDF multiple times
@st.cache_data(show_spinner=True)
def load_pdf(file_path):
    loader = UnstructuredPDFLoader(file_path=file_path)
    return loader.load()

# Use st.cache_resource to handle non-serializable objects like Chroma
@st.cache_resource(show_spinner=True)
def create_vector_db(_chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
    return Chroma.from_documents(documents=_chunks, embedding=embeddings, collection_name="local-rag")

# Cache the text splitting function to avoid re-splitting the same text multiple times
@st.cache_data(show_spinner=True)
def split_text(_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    return text_splitter.split_documents(_data)

# Asynchronous function to get KPI answers
async def get_kpi_answer(vector_db, llm, kpi_question):
    KPI_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""
            You are a highly intelligent assistant. You have been provided with a document titled 
            "Environmental Factors and Pollution in Egypt" by Miral Khodeir, 
            which explores various ecological and environmental issues in Egypt.
            Your task is to extract all the KPIs from the document. \n\n{question}\n\n
            Completely provide them in the format 'KPI: value'. 
            If the value is not provided, use 'Not_Provided' as a placeholder. 
            The user can ask you to change the KPI values or add new KPIs to the list, so be dynamic and helpful.
        """
    )

    retriever2 = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), 
        llm,
        prompt=KPI_PROMPT
    )

    template2 = """
        Answer the question based ONLY on the following context:
        {context}
        Question: {question}
    """

    prompt2 = ChatPromptTemplate.from_template(template2)

    kpi_chain = (
        {"context": retriever2, "question": RunnablePassthrough()}
        | prompt2
        | llm
        | StrOutputParser()
    )

    return kpi_chain.invoke(input=kpi_question)

def main():
    # Set up header with logo
    logo = Image.open("logo.png")  # Add your logo file path here
    st.image(logo, width=100)
    st.title("🧠 ABG system - PDF Question Answering")

    st.markdown(
        """
        <style>
        .user-icon { float: left; width: 50px; margin-right: 10px; }
        .bot-icon { float: left; width: 50px; margin-right: 10px; }
        .message { margin-left: 60px; }
        .clear { clear: both; }
        </style>
        """, unsafe_allow_html=True
    )

    # Upload PDF file
    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("temp_uploaded_file.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_path = "temp_uploaded_file.pdf"

        data = load_pdf(file_path)
        chunks = split_text(data)
        vector_db = create_vector_db(chunks)

        # LLM from Ollama
        local_model = "mistral"
        llm = ChatOllama(model=local_model)

        # Prompt template for query
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""
                You are a highly intelligent assistant. You have been provided with a document titled 
                "Environmental Factors and Pollution in Egypt" by Miral Khodeir, 
                which explores various ecological and environmental issues in Egypt.
                Your task is to generate accurate and concise answers based on the contents of this document.
                Here are some examples to guide you:

                Context: "The expansion of the desert, a process known as desertification, threatens agricultural land in Egypt."
                Question: How does desertification impact Egypt's agriculture?
                Answer: Desertification reduces the amount of usable agricultural land in Egypt, making it harder to grow crops and sustain livestock.

                Context: "Climate change is also disrupting established rainfall patterns in Egypt."
                Question: How is climate change affecting rainfall patterns in Egypt?
                Answer: Climate change is causing irregular rainfall patterns in Egypt, which can lead to droughts or floods, impacting water supply and agriculture.

                Now, I want to know: {question}
            """,
        )

        retriever = MultiQueryRetriever.from_llm(
            vector_db.as_retriever(), 
            llm,
            prompt=QUERY_PROMPT
        )

        # RAG prompt
        template = """
            Answer the question based ONLY on the following context:
            {context}
            and provide your answer in an organized and clear way.
            Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)

        # Chain
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        question = st.text_input("Ask a question about the document")
        if question:
            st.markdown(f"👨‍💻 {question}")
            answer = chain.invoke(input=question)
            st.markdown(f"🤖 {answer}")
            # Append the question and answer to chat history
            st.session_state.chat_history.append((question, answer))

        # Display the chat history
        for q, a in st.session_state.chat_history:
            st.markdown(f"👨‍💻 **You**: {q}")
            st.markdown(f"🤖 **Assistant**: {a}")

        # Sidebar for KPI extraction
        st.sidebar.title("KPIs Dynamic Monitoring")

        kpi_question = st.sidebar.text_input("You can extract, ask, or edit the KPIs")
        if st.sidebar.button("Show Extracted KPIs"):
            kpi_answer = asyncio.run(get_kpi_answer(vector_db, llm, "Extract main KPIs and provide your answer in bullets"))
            st.sidebar.write(kpi_answer)
        else:
            if kpi_question:
                kpi_answer = asyncio.run(get_kpi_answer(vector_db, llm, kpi_question))
                st.sidebar.write(kpi_answer)

if __name__ == "__main__":
    main()
