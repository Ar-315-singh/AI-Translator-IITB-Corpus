import streamlit as st
import pandas as pd
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from datasets import load_dataset

# App Title
st.title("🤖 AI Translation Web Application")
st.write("Translate English text into Hindi seamlessly using Transformers and benchmark datasets!")

# 1. Load Translation Model & Tokenizer
@st.cache_resource
def load_model():
    model_name = "facebook/m2m100_418M"
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# 2. Load IIT Bombay Dataset from Hugging Face
@st.cache_data
def load_iitb_dataset():
    # Loading the validation split and first 100 rows for quick demo performance
    dataset = load_dataset("cfilt/iitb-english-hindi", split="validation[:100]")
    
    # Flatten the nested dictionary structure into a clean list of rows
    cleaned_data = []
    for item in dataset:
        cleaned_data.append({
            "English (Source)": item["translation"]["en"],
            "Reference Hindi (Ground Truth)": item["translation"]["hi"]
        })
    return pd.DataFrame(cleaned_data)

try:
    with st.spinner("Loading IIT Bombay Dataset... Please wait..."):
        df_dataset = load_iitb_dataset()
except Exception as e:
    st.error("Failed to load the dataset. Please check your internet connection.")
    df_dataset = pd.DataFrame()

# --- SIDEBAR: Dataset Explorer ---
selected_text = ""
if not df_dataset.empty:
    st.sidebar.header("📊 Dataset Explorer")
    st.sidebar.write("Explore samples from the IIT Bombay English-Hindi Parallel Corpus:")
    
    # Let user select a sample sentence from the dataset
    sample_options = df_dataset["English (Source)"].tolist()
    choice = st.sidebar.selectbox("Select a sample sentence:", ["-- Select Sample --"] + sample_options)
    
    if choice != "-- Select Sample --":
        selected_text = choice
        # Show reference translation in sidebar
        ref_hi = df_dataset[df_dataset["English (Source)"] == choice]["Reference Hindi (Ground Truth)"].values[0]
        st.sidebar.info(f"**Dataset Reference Translation:**\n{ref_hi}")

# --- MAIN PAGE: Translation Interface ---
st.subheader("✍️ Translate Text")

# If user chose a sample from sidebar, it pre-fills this text area
user_input = st.text_area(
    "Enter English text here:", 
    value=selected_text,
    placeholder="Type something or select a sample from the sidebar dataset..."
)

if st.button("Translate"):
    if user_input:
        with st.spinner("Translating text... Please wait..."):
            # Set source language
            tokenizer.src_lang = "en"
            model_inputs = tokenizer(user_input, return_tensors="pt")
            
            # Generate translation tokens for Hindi (hi)
            target_lang_code = tokenizer.get_lang_id("hi")
            generated_tokens = model.generate(**model_inputs, forced_bos_token_id=target_lang_code)
            
            # Decode tokens to text
            result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            
            # Display Result
            st.success(f"**Model AI Translation:** {result}")
    else:
        st.warning("Please enter some text before clicking translate!")

# --- BOTTOM PAGE: View Entire Dataset Table ---
if not df_dataset.empty:
    st.write("---")
    st.subheader("📂 IIT Bombay Corpus Dataset Preview")
    st.write("Below is the structured preview of the loaded dataset rows:")
    st.dataframe(df_dataset, use_container_width=True)