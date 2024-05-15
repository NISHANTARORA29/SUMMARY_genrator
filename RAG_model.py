import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from nltk.tokenize import sent_tokenize
from textwrap import wrap
import PyPDF2

# Load pre-trained T5 model and tokenizer
summary_model = T5ForConditionalGeneration.from_pretrained('t5-base')
summary_tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
summary_model = summary_model.to(device)

# Define function to preprocess text
def preprocess_text(text):
    # Remove extra whitespace and newline characters
    text = ' '.join(text.split())
    return text

# Define function to post-process summary
def postprocess_summary(summary):
    # Capitalize first letter of each sentence
    summary = '. '.join(sentence.capitalize() for sentence in summary.split('. '))
    return summary

# Define function for document summarization
def summarize_document(text, model, tokenizer):
    # Preprocess input text
    text = preprocess_text(text)
    
    # Tokenize input text
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True).to(device)
    
    # Generate summary
    summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    # Decode and post-process summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    summary = postprocess_summary(summary)
    
    return summary

# Streamlit app
st.title("Document Summarization Tool")

# Get input text or PDF file
input_type = st.radio("Select input type:", ("Text", "PDF"))

if input_type == "Text":
    input_text = st.text_area("Enter the text to summarize:", height=200)
else:
    uploaded_file = st.file_uploader("Upload a PDF file:", type="pdf")
    if uploaded_file is not None:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        input_text = ""
        for page in pdf_reader.pages:
            input_text += page.extract_text()

# Summarize the input text
if st.button("Summarize"):
    summarized_text = summarize_document(input_text, summary_model, summary_tokenizer)

    st.subheader("Original Text:")
    for line in wrap(input_text, 150):
        st.write(line)

    st.subheader("Summarized Text:")
    for line in wrap(summarized_text, 150):
        st.write(line)
