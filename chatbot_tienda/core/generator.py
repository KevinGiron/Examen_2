import torch

def generar_respuesta(mensaje, historial, temperature, max_tokens, tokenizer, model):
    contexto = ""
    for msg in historial[-4:]:
        contexto += msg["contenido"] + tokenizer.eos_token

    input_ids = tokenizer.encode(
        contexto + mensaje + tokenizer.eos_token,
        return_tensors="pt"
    )

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    respuesta = tokenizer.decode(
        output_ids[:, input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    if len(respuesta.strip()) < 5:
        respuesta = (
            "Puedo ayudarte con compras, devoluciones o información de productos. "
            "¿Qué necesitas?"
        )

    if respuesta.strip().lower() == mensaje.strip().lower():
        respuesta = (
            "Claro, puedo ayudarte con la devolución. "
            "¿Podrías indicarme el motivo y el número de pedido?"
        )


    return respuesta
