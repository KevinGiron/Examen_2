import streamlit as st

def sidebar_config():
    st.sidebar.header("⚙️ Configuración")

    st.session_state.temperature = st.sidebar.slider(
        "Creatividad",
        0.2, 1.2, 0.7, 0.1
    )

    st.session_state.max_tokens = st.sidebar.slider(
        "Longitud máxima",
        30, 200, 80, 10
    )

    if st.sidebar.button("Limpiar conversación"):
        st.session_state.historial = []
        st.rerun()

    st.sidebar.info("Modelo: GPT-2 (Hugging Face)")
