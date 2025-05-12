from PIL import Image, ImageOps
from keras.models import load_model

import matplotlib
import numpy as np
import streamlit as st
import tensorflow as tf
import cv2

def load_image(img):
    im = Image.open(img)
    im = im.resize([224,224])
    return im

st.title("Fake-DICOM: Uso de redes neurais profundas para avaliar se a combinação de algoritmos de criptografia com transformadas temporais aumenta a segurança de imagens DICOM contra tentativas intrusão maliciosas")
st.header("Envie um exame médico qualquer .jpg ou .png", divider="gray")

Model_CNN = tf.keras.models.load_model('model_fold_1.keras')

for layer in Model_CNN.keras.layers:
    layer.trainable = False

uploadFile = st.file_uploader(label="Upload image", type=['jpg', 'png'])

def generate_heatmap(model, sample_image):
    sample_image_exp = np.expand_dims(sample_image, axis=0)
    
    intermediate_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer('last_conv').output)
    activations = intermediate_model.predict(sample_image_exp)
    activations = tf.convert_to_tensor(activations)

    predictions = model.predict(sample_image_exp)

    with tf.GradientTape() as tape:
        iterate = tf.keras.models.Model([model.input], [model.output, model.get_layer('last_conv').output])
        model_out, last_conv_layer = iterate(sample_image_exp)
        class_out = model_out[:, np.argmax(model_out[0])]
        tape.watch(last_conv_layer)
        grads = tape.gradient(class_out, last_conv_layer)

    if grads is None:
        raise ValueError('Gradients could not be computed. Check the model and layer names.')

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    pooled_grads = tf.where(pooled_grads == 0, tf.ones_like(pooled_grads) * 1e-10, pooled_grads)
    
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer[0]), axis=-1)

    min_value = np.min(heatmap)
    max_value = np.max(heatmap)

    heatmap = (heatmap - min_value) / (max_value - min_value)
    heatmap = np.asarray(heatmap)
    heatmap = (heatmap - 1) * (-1)

    heatmap_resized = cv2.resize(heatmap, (sample_image.shape[1], sample_image.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    
    heatmap_colored = matplotlib.cm.jet(heatmap_resized)[:, :, :3]
    heatmap_colored = np.uint8(heatmap_colored * 255)
    
    alpha_channel = np.uint8(heatmap_resized)
    heatmap_colored_with_alpha = np.dstack((heatmap_colored, alpha_channel))
    
    sample_image_uint8 = np.uint8(255 * np.squeeze(sample_image))
    image_rgb = cv2.cvtColor(sample_image_uint8, cv2.COLOR_GRAY2RGB)
    image_rgba = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2RGBA)
    
    alpha_factor = alpha_channel / 255.0
    for c in range(0, 3):
        image_rgba[..., c] = image_rgba[..., c] * (1 - alpha_factor) + heatmap_colored[..., c] * alpha_factor
    
    return image_rgba

def CNN_predict(input_img, Model_CNN):
    dense_CNN = Model_CNN.get_layer('dense').output
    dense_CNN_extractor = tf.keras.models.Model(inputs=Model_CNN.input, outputs=dense_CNN)

    dense_features = dense_CNN_extractor.predict(input_img)
    dense_features = dense_features.reshape(dense_features.shape[0], -1)

    pred_confidence_CNN = Model_CNN.predict(dense_features)
    pred_class_labels_CNN = np.argmax(pred_confidence_CNN, axis=1)

    return pred_confidence_CNN, pred_class_labels_CNN

if uploadFile is not None:
    img = load_image(uploadFile)
    st.image(img)
    hide_img_fs = '''
    <style>
    button[title="View fullscreen"]{
        visibility: hidden;}
    </style>
    '''

    if st.button('Diagnosis'):
        X = Image.open(uploadFile)
        X = ImageOps.grayscale(X)

        img_array = np.array(X)
        
        st.markdown(hide_img_fs, unsafe_allow_html=True)
        st.write("Image Uploaded Successfully")
        
        input_shape = Model_CNN.input_shape[1:-1]
        h, w = input_shape
        
        image_resized = cv2.resize(img_array, (w, h))
        image_normalized = image_resized.astype('float32') / 255.0
        
        image_normalized = np.expand_dims(image_normalized, axis=-1)
        sample_image_exp = np.expand_dims(image_normalized, axis=0)

# Predição CNN
        
        prediction, y_pred = CNN_predict(sample_image_exp, Model_CNN)

        print(prediction.max())
        print(y_pred[0])
        
        if(y_pred[0] == 0):
            st.subheader("Normal")
            st.write("Esta imagem tem " + str("{:.2f}".format(prediction[0].max()*100)+"% de ser segura para ser aberta."))
        else:
            st.subheader("Hackeada")
            st.write("Esta imagem tem " + str("{:.2f}".format(prediction[0].max()*100)+"% de conter informações maliciosas."))

# Predições Conv4
        
        pred_CNN = Model_CNN.predict(sample_image_exp)
        
        pred_CNN_class = np.argmax(pred_CNN[0])
        CNN_confidence = pred_CNN[0][pred_CNN_class]

# Heatmap

        heatmap_image = generate_heatmap(Model_CNN, image_normalized)

        st.image(heatmap_image)
else:
    st.write("Por gentileza, envie um arquvo em .jpg ou .png.")
