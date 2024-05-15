import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import tempfile
from dotenv import load_dotenv
import os 

load_dotenv()

st.set_page_config(page_title="Text Summarizer", page_icon=":robot_face:", layout="wide")

st.title("Text Summarizer")

file_type = st.selectbox("Choose file type:", options=["PDF", "Text"])

if file_type == "PDF":
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name

        # Load the PDF file
        loader = UnstructuredPDFLoader(file_path=tmp_file_path)
        data = loader.load()

        # Split and chunk the text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)
else:
    uploaded_file = st.file_uploader("Upload a text file", type="txt")
    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Read the text file
        with open(tmp_file_path, "r", encoding="utf-8") as text_file:
            data = text_file.read()
        chunks = [data]

if uploaded_file is not None:
    # Add the chunks to the vector database
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001", show_progress=True),
        collection_name="text"
    )

    # Set up the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

    # RAG prompt
    template = """give me summary not in points according to the number of words or max length
    {context}
    """

    prompt = PromptTemplate.from_template(template)

    # Customizable Summary Length
    summary_length = st.slider("Choose summary length (in sentences):", min_value=1, max_value=100, value=5)

    # Customizing the style of the summary
    summary_style = st.radio("Choose summary style:", options=["Paragraph", "Bullet Points"], index=0)

    # Fine-tuning the model for summary generation based on word length
    word_length = st.slider("Choose summary word length:", min_value=10, max_value=1000, value=100)

    chain = (
        {"context": vector_db.as_retriever()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Generate summary
    summary = chain.invoke("Summarize the key points from the document")

    # Filter out redundant information
    summary_sentences = summary.split(". ")
    unique_summary = []
    for sentence in summary_sentences:
        if sentence not in unique_summary:
            unique_summary.append(sentence)

    # Combine unique sentences to form the final summary
    final_summary = ". ".join(unique_summary[:summary_length])

    # Trim summary to match word length
    final_summary_words = final_summary.split()[:word_length]
    final_summary = " ".join(final_summary_words)

    # Display Summary
    st.write("Summary:")
    if summary_style == "Paragraph":
        st.write(final_summary)
    else:
        bullet_points = [f"- {sentence}" for sentence in final_summary.split(". ")]
        st.write("\n".join(bullet_points))

    # Remove the temporary file
    os.remove(tmp_file_path)
