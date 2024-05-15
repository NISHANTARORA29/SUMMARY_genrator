import streamlit as st
import PyPDF2
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
from nltk.tokenize import sent_tokenize
from textwrap import wrap

# Load pre-trained Pegasus model and tokenizer
summary_model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')
summary_tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
summary_model = summary_model.to(device)

# Define function to capitalize sentences
def postprocesstext(content):
    final = ""
    for sent in sent_tokenize(content):
        sent = sent.capitalize()
        final = final + " " + sent
    return final

# Define summarizer function with Pegasus
def summarizer_with_Pegasus(text, model, tokenizer, max_len=1024, min_len=50, num_beams=4):
    # Prepare input text for summarization
    text = text.strip().replace("\n", " ")

    # Tokenize input text and encode
    inputs = tokenizer([text], max_length=max_len, return_tensors='pt', truncation=True).to(device)

    # Generate summary using Pegasus
    outs = model.generate(**inputs,
                          max_length=150,  # Adjust max_length for desired summary length
                          min_length=50,   # Adjust min_length for desired summary length
                          length_penalty=2.0,  # Adjust length_penalty for controlling length
                          num_beams=num_beams,     # Adjust num_beams for better results
                          early_stopping=True)

    # Decode summary and apply post-processing
    summary = tokenizer.decode(outs[0], skip_special_tokens=True)
    summary = postprocesstext(summary)
    summary = summary.strip()

    return summary

# Define function for filtering redundant information
def filter_redundant_information(text, keywords):
    if keywords:
        return "\n".join([sent for sent in sent_tokenize(text) if not any(keyword.lower() in sent.lower() for keyword in keywords.split(","))])
    else:
        return text

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

# Additional Features
if st.checkbox("Enable additional features"):
    # Redundant Information Filtering
    # Example: Filtering out sentences containing the same keywords
    keywords = st.text_input("Enter keywords to filter redundant information (comma-separated):")
    input_text = filter_redundant_information(input_text, keywords)

    # Customization Options
    summary_length = st.slider("Select summary length:", min_value=50, max_value=500, value=150)
    num_beams = st.slider("Select number of beams for summarization:", min_value=1, max_value=8, value=4)

# Summarize the input text
if st.button("Summarize"):
    summarized_text = summarizer_with_Pegasus(input_text, summary_model, summary_tokenizer, max_len=1024, min_len=50, num_beams=num_beams)

    st.subheader("Original Text:")
    for line in wrap(input_text, 150):
        st.write(line)

    st.subheader("Summarized Text:")
    for line in wrap(summarized_text, 150):
        st.write(line)
