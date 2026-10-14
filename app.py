
import os
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    PROJECT_PATH, "models", "densenet121_best.keras"
)

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🩻",
    layout="wide"
)

st.title("Pneumonia Detection from Chest X-rays")
st.caption("Deep Learning-Based Detection Using Explainable AI")

st.warning(
    "Research Prototype — Not for Clinical Diagnosis. "
    "This application is for educational and research purposes only. "
    "Do not use its predictions to make medical decisions."
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH, compile=False
    )

def make_gradcam(model, image_array):
    # Locate the DenseNet121 feature extractor
    base_index = None
    base_model = None

    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.Model):
            try:
                layer.get_layer("conv5_block16_2_conv")
                base_index = i
                base_model = layer
                break
            except (ValueError, AttributeError):
                continue

    if base_model is None:
        raise ValueError(
            "Could not locate DenseNet121 or its final convolutional layer."
        )

    last_conv = base_model.get_layer("conv5_block16_2_conv")

    feature_model = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=[last_conv.output, base_model.output]
    )

    # Apply the saved model's classification head to the feature maps
    head_layers = model.layers[base_index + 1:]

    input_tensor = tf.convert_to_tensor(
        image_array, dtype=tf.float32
    )

    with tf.GradientTape() as tape:
        conv_outputs, features = feature_model(
            input_tensor, training=False
        )
        tape.watch(conv_outputs)

        x = features
        for layer in head_layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                x = layer(x, training=False)
            else:
                x = layer(x)

        # Binary sigmoid output: explain the predicted class
        probability = x[:, 0]
        class_score = (
            probability
            if float(probability[0].numpy()) >= 0.5
            else 1.0 - probability
        )

    gradients = tape.gradient(class_score, conv_outputs)

    if gradients is None:
        raise ValueError("Grad-CAM gradients could not be calculated.")

    weights = tf.reduce_mean(gradients, axis=(1, 2))
    heatmap = tf.reduce_sum(
        conv_outputs[0] * weights[0], axis=-1
    )
    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)
    heatmap = heatmap / (max_value + 1e-8)

    return heatmap.numpy(), float(probability[0].numpy())

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

st.success("DenseNet121 model loaded successfully.")

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        resized_image = image.resize((224, 224))

        image_array = np.asarray(
            resized_image, dtype=np.float32
        ) / 255.0

        input_array = np.expand_dims(image_array, axis=0)

        st.subheader("Uploaded X-ray")
        st.image(image, caption="Original chest X-ray", width=350)

        if st.button("Analyze X-ray", type="primary"):
            with st.spinner("Analyzing image..."):
                probability = float(
                    model.predict(input_array, verbose=0)[0][0]
                )

                prediction = (
                    "PNEUMONIA" if probability >= 0.5
                    else "NORMAL"
                )

                st.subheader("Model Prediction")

                if prediction == "PNEUMONIA":
                    st.error("Predicted class: PNEUMONIA")
                else:
                    st.info("Predicted class: NORMAL")

                st.metric(
                    "Model-estimated pneumonia probability",
                    f"{probability * 100:.2f}%"
                )

                st.caption(
                    "This probability is a model output, not a "
                    "validated clinical risk estimate."
                )

                try:
                    heatmap, checked_probability = make_gradcam(
                        model, input_array
                    )

                    heatmap_image = Image.fromarray(
                        np.uint8(heatmap * 255)
                    ).resize((224, 224))

                    heatmap_array = np.asarray(heatmap_image)

                    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

                    axes[0].imshow(resized_image)
                    axes[0].set_title("Resized X-ray")
                    axes[0].axis("off")

                    axes[1].imshow(resized_image)
                    axes[1].imshow(
                        heatmap_array,
                        cmap="jet",
                        alpha=0.4,
                        vmin=0,
                        vmax=255
                    )
                    axes[1].set_title("DenseNet121 Grad-CAM")
                    axes[1].axis("off")

                    plt.tight_layout()
                    st.subheader("Explainable AI — Grad-CAM")
                    st.pyplot(fig)
                    plt.close(fig)

                    st.caption(
                        "Grad-CAM highlights image regions influencing "
                        "the selected model output. It does not confirm "
                        "the presence or location of disease."
                    )

                except Exception as error:
                    st.warning(
                        "Prediction completed, but Grad-CAM could not "
                        f"be generated: {error}"
                    )

    except Exception as error:
        st.error(f"Could not process the uploaded image: {error}")
