import streamlit as st

def init_memory():
    if "historial" not in st.session_state:
        st.session_state.historial = []

def add_message(rol, contenido):
    st.session_state.historial.append({
        "rol": rol,
        "contenido": contenido
    })
