from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from keras.layers import Lambda
from keras.utils import CustomObjectScope
from PIL import Image

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import cv2
import io
import time
import os

# ----------------- MODELO -----------------
model_path = 'Modelos/model_MobileNet_01p100.keras'
#model_path = 'Modelos/model_MobileNet_5p100.keras'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Modelo não encontrado em: {os.path.abspath(model_path)}")

def expand_channels(x):
    return tf.stack([x[..., 0]]*3, axis=-1)

modelo_MobileNet = tf.keras.models.load_model(
    model_path,
    custom_objects={'expand_channels': expand_channels},
    safe_mode=False
)

for layer in modelo_MobileNet.layers:
    layer.trainable = False

# ----------------- FUNÇÕES -----------------
def preprocessar_imagem(imagem):
    img = cv2.resize(imagem, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = img.astype('float32') / 255.0
    return img

def freq_spec(fshift, image, threshold_percent, add_noise, corner):
    threshold = threshold_percent/100

    if add_noise:
        rows, cols = image.shape
        noise_size = int(np.sqrt(threshold * rows * cols))
        sigma = 0.8

        def gaussian_blur(size, sigma):
            ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
            xx, yy = np.meshgrid(ax, ax)
            kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
            return kernel / np.sum(kernel)

        blur_kernel = gaussian_blur(noise_size, sigma)

        def blur_patch(spectrum, r_start, c_start):
            patch = spectrum[r_start:r_start+noise_size, c_start:c_start+noise_size]
            if patch.shape != blur_kernel.shape:
                return
            real_blurred = cv2.filter2D(np.real(patch), -1, blur_kernel)
            imag_blurred = cv2.filter2D(np.imag(patch), -1, blur_kernel)
            spectrum[r_start:r_start+noise_size, c_start:c_start+noise_size] = real_blurred + 1j * imag_blurred

        if corner == 0:
            blur_patch(fshift, 0, 0)
        elif corner == 1:
            blur_patch(fshift, 0, cols - noise_size)
        elif corner == 2:
            blur_patch(fshift, rows - noise_size, 0)
        else:
            blur_patch(fshift, rows - noise_size, cols - noise_size)

    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    magnitude_spectrum_norm = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)

    return fshift, magnitude_spectrum_norm.astype(np.uint8)

def gerar_heatmap(model, sample_image):
    sample_image_resized = cv2.resize(sample_image, (224, 224))
    if len(sample_image_resized.shape) == 2:
        sample_image_resized = sample_image_resized[..., np.newaxis]

    sample_image_resized = sample_image_resized.astype('float32') / 255.0
    sample_image_exp = np.expand_dims(sample_image_resized, axis=0)

    camadas_conv = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    ultima_conv = camadas_conv[-1] if camadas_conv else None

    intermediate_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(ultima_conv).output)
    activations = intermediate_model.predict(sample_image_exp)

    predictions = model.predict(sample_image_exp)

    with tf.GradientTape() as tape:
        iterate = tf.keras.models.Model([model.input], [model.output, model.get_layer(ultima_conv).output])
        model_out, last_conv_layer = iterate(sample_image_exp)
        class_out = model_out[:, np.argmax(model_out[0])]
        tape.watch(last_conv_layer)
        grads = tape.gradient(class_out, last_conv_layer)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    pooled_grads = tf.where(pooled_grads == 0, tf.ones_like(pooled_grads) * 1e-10, pooled_grads)

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer[0]), axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1

    heatmap_resized = cv2.resize(heatmap, (sample_image.shape[1], sample_image.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)

    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
    heatmap_colored = np.uint8(heatmap_colored * 255)

    alpha_channel = np.uint8(heatmap_resized)
    heatmap_colored_with_alpha = np.dstack((heatmap_colored, alpha_channel))

    return heatmap_colored_with_alpha

def criptografar_imagem(fshift, aes_key):
    inicio = time.perf_counter()
    buffer = io.BytesIO()
    np.save(buffer, fshift)
    buffer.seek(0)
    cipher_aes = AES.new(aes_key, AES.MODE_EAX)
    ciphertext, tag = cipher_aes.encrypt_and_digest(buffer.read())

    out_buffer = io.BytesIO()
    out_buffer.write(cipher_aes.nonce)
    out_buffer.write(tag)
    out_buffer.write(ciphertext)
    out_buffer.seek(0)
    fim = time.perf_counter()
    return out_buffer, fim - inicio

def descriptografar_imagem(chave, arquivo):
    inicio = time.perf_counter()
    chave.seek(0)
    arquivo.seek(0)
    aes_key = chave.read()
    enc_bytes = arquivo.read()

    nonce = enc_bytes[:16]
    tag = enc_bytes[16:32]
    ciphertext = enc_bytes[32:]
    cipher_aes = AES.new(aes_key, AES.MODE_EAX, nonce)
    decrypted_data = cipher_aes.decrypt_and_verify(ciphertext, tag)
    buffer = io.BytesIO(decrypted_data)
    fshift_restaurado = np.load(buffer)
    fim = time.perf_counter()
    return fshift_restaurado, fim - inicio


def ifft(fshift):
    imagem_restaurada = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift)))
    imagem_restaurada_uint8 = cv2.normalize(imagem_restaurada, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return imagem_restaurada_uint8

# ----------------- INTERFACE -----------------
st.set_page_config(
    page_title="FakeDICOM",
    # layout="wide",  # opções: "centered" (padrão) ou "wide"
    # initial_sidebar_state="expanded"
)

image = Image.open("banner.jpg")
st.image(image, use_container_width =True)

st.title("Fake-DICOM: Detecção de Anomalias em Imagens Médicas")

# ----- ESTATÍSTICAS -----
st.subheader("Estatísticas da Criptografia")
st.caption("Último teste realizado com o algoritmo AES para geração de estatísticas: 10/05/2025")
df = pd.read_excel('cripto.xlsx')
st.dataframe(df.describe())

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df['slice'], df['tempoExecucaoAES'], color='blue')
ax.set_title('Execução da Criptografia por Slice - 1799 Slices')
ax.set_xticks([])
ax.tick_params(axis='y', labelsize=10)
ax.set_xlabel('Slice', fontsize=10)
ax.set_ylabel('Tempo de Execução (s)', fontsize=10)
ax.legend(['Tempo de Execução'])
st.pyplot(fig)

st.markdown("---")

# ----- UPLOAD DE IMAGEM -----
st.title("Upload de Imagem PNG")
arquivo_imagem = st.file_uploader("Escolha um arquivo PNG para ser analisado", type=["png"])

if arquivo_imagem:
    imagem_upload = Image.open(arquivo_imagem)
    imagem_np = np.array(imagem_upload)
    imagem = cv2.resize(imagem_np, (512, 512), interpolation=cv2.INTER_AREA)
    f = np.fft.fft2(imagem)
    fshift = np.fft.fftshift(f)

    mag_spec = 20 * np.log(np.abs(fshift) + 1)
    mag_spec_norm = cv2.normalize(mag_spec, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    st.success("Imagem carregada com sucesso e Transformada para o domínio da frequência!")
    col1, col2 = st.columns(2)
    with col1:
        st.image(imagem, caption="Imagem carregada", use_container_width=True)
    with col2:
        st.image(mag_spec_norm, caption="Espectro Gerado", use_container_width=True)

    st.markdown("---")

# ----- CRIPTOGRAFIA -----
    st.header("🔐 Criptografia AES!")
    if 'aes_key' not in st.session_state:
        st.session_state.aes_key = get_random_bytes(32)

    st.markdown('<h4>🔑 Sua chave de Criptografia AES (salve):</h4>', unsafe_allow_html=True)
    st.code(st.session_state.aes_key.hex(), language="text")

    st.download_button("🔑 Baixar Chave de Criptografia", st.session_state.aes_key, "aes_key.pem", "application/octet-stream")
    
    st.markdown("---")
    st.markdown('### ▶️ Executar!')
    if st.button("🔒 Executar Criptografia"):
        cripto, tempo = criptografar_imagem(fshift, st.session_state.aes_key)
        st.session_state.cripto = cripto
        st.session_state.tempo_cripto = tempo

    if 'cripto' in st.session_state:
        st.markdown('<h4>🔒 30 primeiros bytes do arquivo criptografado (salve):</h4>', unsafe_allow_html=True)
        st.code(st.session_state.cripto.getvalue()[:30], language="text")

        st.download_button("🗃️ Baixar Arquivo Criptografado (.enc)", st.session_state.cripto, "imagem_criptografada.enc", "application/octet-stream")
        st.success(f"✅ Criptografia concluída em {st.session_state.tempo_cripto:.4f} segundos.")

    st.markdown("---")

# ----- DESCRIPTOGRAFIA -----
    st.header("🔓 Descriptografia AES!")

    col1, col2 = st.columns(2)    
    with col1:
        st.markdown('<h6>🔑 Insira a Chave para Descriptografia</h6>', unsafe_allow_html=True)
        chave_descript = st.file_uploader("", type=["pem"], key="chave")
    
    with col2:
        st.markdown('<h6>🗃️ Insira o Arquivo Criptografado</h6>', unsafe_allow_html=True)
        enc_file = st.file_uploader("", type=["enc"], key="arquivo")


    if chave_descript and enc_file:
        try:
            fshift_restaurado, tempo_decript = descriptografar_imagem(chave_descript, enc_file)
            st.success(f"✅ Descriptografia concluída em {tempo_decript:.4f} segundos.")
        except Exception as e:
            st.error(f"❌ Erro na descriptografia: {e}")

    st.markdown("---")
    
# ----- INICIAR CNN -----
    if st.button("Iniciar CNN"):
        st.session_state.cnn_ativa = True

    try:
        if st.session_state.cnn_ativa:
            st.subheader("🔍 Análise de Segurança com MobileNet")

            col1, col2 = st.columns(2)
            with col1:
                st.image(imagem, caption="Imagem Original", use_container_width=True)
            with col2:
                st.image(mag_spec_norm, caption="Espectro de Frequência", use_container_width=True)

            st.markdown("### 🎯 Selecionar Região para Análise")
            opcao_ruido = st.radio("Deseja aplicar ruído na imagem?", ("Não", "Sim"))

            canto = st.selectbox("Selecione o canto para análise:", ["Superior Esquerdo", "Superior Direito", "Inferior Esquerdo", "Inferior Direito"])

            corners_dict = {
                "Superior Esquerdo": 0,
                "Superior Direito": 1,
                "Inferior Esquerdo": 2,
                "Inferior Direito": 3
            }

            aplicar_ruido = opcao_ruido == "Sim"

            modified_fshift, mag_spec_modificado = freq_spec(
                fshift.copy(), imagem, threshold_percent=0.1, add_noise=aplicar_ruido, corner=corners_dict[canto]
            )

            # Imagem restaurada (alterada ou não) para exibição
            img_restaurada = ifft(modified_fshift)
            img_processada = preprocessar_imagem(img_restaurada)

            # Predição com CNN
            predicao = modelo_MobileNet.predict(img_processada[np.newaxis, ...])
            classe = np.argmax(predicao)
            confianca = predicao[0][classe]

            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(img_restaurada, caption="Imagem Analisada")
            with col2:
                st.image(mag_spec_modificado, caption="Espectro Analisado")

            # Se houver ruído, gera o heatmap
            if aplicar_ruido:
                heatmap = gerar_heatmap(modelo_MobileNet, mag_spec_modificado)

                rotation_code = {
                    0: None,
                    1: cv2.ROTATE_90_CLOCKWISE,
                    2: cv2.ROTATE_90_COUNTERCLOCKWISE,
                    3: cv2.ROTATE_180
                }[corners_dict[canto]]

                if rotation_code is not None:
                    rotated_heatmap = cv2.rotate(heatmap, rotation_code)
                else:
                    rotated_heatmap = heatmap

                mag_spec_rgb = cv2.cvtColor(mag_spec_modificado, cv2.COLOR_GRAY2RGB)
                mag_spec_rgba = cv2.cvtColor(mag_spec_rgb, cv2.COLOR_RGB2RGBA)

                overlay_pil = Image.alpha_composite(
                    Image.fromarray(mag_spec_rgba),
                    Image.fromarray(rotated_heatmap)
                )

                with col3:
                    st.image(overlay_pil.convert('RGB'), caption="Mapa de Ativação sobre Espectro")
            else:
                with col3:
                    st.image(np.zeros_like(imagem), caption="(Sem ruído detectado)")

            # Diagnóstico final
            st.markdown(f"**Diagnóstico:** {'🚨 Com Ruído' if opcao_ruido == 'Sim' else '✅ Normal'}") 

            if opcao_ruido == "Sim":
                st.warning("**⚠️ Invasão detectada! Imagem bloqueada por segurança!**")
            else:
                st.success("**✅ Nenhuma invasão detectada!**")
                
                # Exibir imagem restaurada com ifft caso imagem esteja normal
                if not aplicar_ruido:
                    st.markdown("---")
                    st.subheader("↩️ Restauração da Imagem Original")
                    col_central = st.columns([1, 2, 1])[1]
                    with col_central:
                        st.image(img_restaurada, width=300)

            predicao = modelo_MobileNet.predict(img_processada[np.newaxis, ...])
            classe = np.argmax(predicao)
            confianca = predicao[0][classe]

    except Exception as e:
        st.error(f"Erro: {e}")
st.markdown("""<hr style="border:1px solid gray">""", unsafe_allow_html=True)
st.caption("TCC - Ciência da Computação - FEI")
st.caption("Projeto desenvolvido por Gabriel N. Missima, Carlos M. H. Chinen, Vinicius A. Pedro")
st.caption("Orientador: Prof. Dr. Paulo Sérgio Rodrigues")
