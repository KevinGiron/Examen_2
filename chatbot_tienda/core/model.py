import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

@st.cache_resource
def cargar_modelo():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    model.eval()
    return tokenizer, model
