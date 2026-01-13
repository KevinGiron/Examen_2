import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

st.set_page_config(
    page_title="Asistente Virtual de la Tienda",
    layout="centered"
)

st.title("Asistente Virtual de la Tienda")
st.write("Atención automática para consultas de productos, compras y devoluciones.")

@st.cache_resource
def cargar_modelo():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    return tokenizer, model

tokenizer, model = cargar_modelo()

if "historial" not in st.session_state:
    st.session_state.historial = []

st.sidebar.header("Panel de control")

temperature = st.sidebar.slider(
    "Creatividad",
    0.2, 1.0, 0.6, 0.1
)

max_tokens = st.sidebar.slider(
    "Longitud máxima",
    80, 120, 80, 10
)

st.sidebar.markdown("Modelo en uso: GPT-2")

if st.sidebar.button("Limpiar conversación"):
    st.session_state.historial = []
    st.rerun()

def generar_respuesta(mensaje, historial):
    contexto = ""
    for h in historial[-4:]:
        contexto += f"{h['rol']}: {h['contenido']}\n"

    prompt = (
        "Eres un asistente virtual profesional de una tienda online de electrónicos.\n"
        "Respondes SIEMPRE en español.\n"
        "Respondes de forma clara, educada y breve.\n"
        "Atiendes consultas sobre productos, compras y devoluciones.\n\n"
        f"{contexto}"
        f"Cliente: {mensaje}\n"
        "Asistente:"
    )

    inputs = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            inputs,
            max_length=inputs.shape[1] + max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    texto = tokenizer.decode(output[0], skip_special_tokens=True)

    respuesta = texto.split("Asistente:")[-1].strip()

    if (
        respuesta.lower() == mensaje.lower()
        or len(respuesta) < 5
        or "español" in respuesta.lower()
    ):
        respuesta = (
            "Claro, puedo ayudarte con la devolución. "
            "Indícame el motivo, la fecha de compra y el número de pedido."
        )

    return respuesta

st.subheader("Conversación")

for m in st.session_state.historial:
    st.markdown(f"**{m['rol']}:** {m['contenido']}")

with st.form(key="chat_form", clear_on_submit=True):
    mensaje_usuario = st.text_input("Escribe tu mensaje")
    enviar = st.form_submit_button("Enviar")

    if enviar and mensaje_usuario.strip():
        st.session_state.historial.append({
            "rol": "Cliente",
            "contenido": mensaje_usuario
        })

        respuesta = generar_respuesta(
            mensaje_usuario,
            st.session_state.historial
        )

        st.session_state.historial.append({
            "rol": "Asistente",
            "contenido": respuesta
        })

        st.rerun()
