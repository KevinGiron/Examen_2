import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

CLASS_NAMES = [
    'Avión', 'Automóvil', 'Pájaro', 'Gato', 'Ciervo',
    'Perro', 'Rana', 'Caballo', 'Barco', 'Camión'
]

model = load_model("modelo_cnn.h5")

st.title("Sistema de Reconocimiento de Objetos - Aeropuerto")
st.write("Clasificación automática de objetos usando CNN")

uploaded_file = st.file_uploader(
    "Sube una imagen",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen cargada", use_column_width=True)

    image = image.resize((32, 32))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 32, 32, 3)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.subheader("Resultado")
    st.write(f"**Objeto detectado:** {CLASS_NAMES[class_index]}")
    st.write(f"**Confianza:** {confidence:.2f}%")
