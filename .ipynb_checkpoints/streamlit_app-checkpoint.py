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

st.title("Kidney Stone Detection from Coronal CT Images")
st.header("Upload a coronal CT image to be diagnosed", divider="gray")

Conv4_A = tf.keras.models.load_model('model_Conv4_A.keras')
Conv4_B = tf.keras.models.load_model('model_Conv4_B.keras')
DEMLP = tf.keras.models.load_model('model_DEMLP.keras')

for layer in Conv4_A.layers:
    layer.trainable = False
for layer in Conv4_B.layers:
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

def DEMLP_predict(input_img, model_A, model_B, model_DEMLP):
    dense_output_A = model_A.get_layer('dense').output
    model_A_extractor = tf.keras.models.Model(inputs=model_A.input, outputs=dense_output_A)
    dense_output_B = model_B.get_layer('dense').output
    model_B_extractor = tf.keras.models.Model(inputs=model_B.input, outputs=dense_output_B)

    img_features_A = model_A_extractor.predict(input_img)
    img_features_B = model_B_extractor.predict(input_img)

    img_features = np.column_stack((img_features_A, img_features_B))
    img_features = img_features.reshape(img_features.shape[0], -1)

    pred_confidence_DEMLP = model_DEMLP.predict(img_features)
    pred_class_labels_DEMLP = np.argmax(pred_confidence_DEMLP, axis=1)

    return pred_confidence_DEMLP, pred_class_labels_DEMLP

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
        
        input_shape = Conv4_A.input_shape[1:-1]
        h, w = input_shape
        
        image_resized = cv2.resize(img_array, (w, h))
        image_normalized = image_resized.astype('float32') / 255.0
        
        image_normalized = np.expand_dims(image_normalized, axis=-1)
        sample_image_exp = np.expand_dims(image_normalized, axis=0)

# Predição DE-MLP
        
        prediction, y_pred = DEMLP_predict(sample_image_exp, Conv4_A, Conv4_B, DEMLP)

        print(prediction.max())
        print(y_pred[0])
        
        if(y_pred[0] == 0):
            st.subheader("Kidney Stone")
            st.write("This image has a " + str("{:.2f}".format(prediction[0].max()*100)+"% probability of containing a kidney stone."))
        else:
            st.subheader("Healthy")
            st.write("This image has a " + str("{:.2f}".format(prediction[0].max()*100)+"% probability of being healthy."))

# Predições Conv4
        
        pred_Conv4_A = Conv4_A.predict(sample_image_exp)
        pred_Conv4_B = Conv4_B.predict(sample_image_exp)
        
        pred_Conv4_A_class = np.argmax(pred_Conv4_A[0])
        confidence_A = pred_Conv4_A[0][pred_Conv4_A_class]
        
        pred_Conv4_B_class = np.argmax(pred_Conv4_B[0])
        confidence_B = pred_Conv4_B[0][pred_Conv4_B_class]

# Heatmap

        if confidence_A >= confidence_B:
            heatmap_image = generate_heatmap(Conv4_A, image_normalized)
        else:
            heatmap_image = generate_heatmap(Conv4_A, image_normalized)

        st.image(heatmap_image)
else:
    st.write("Make sure you image is in JPG/PNG Format.")
