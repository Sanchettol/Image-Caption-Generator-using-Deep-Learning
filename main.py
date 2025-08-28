# main.py
import streamlit as st
import numpy as np
import io
import pandas as pd
import pickle
from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from typing import List, Tuple

# ---------- Utils & cached loaders ----------
@st.cache_resource(show_spinner=False)
def load_models(model_path: str, feature_path: str, tokenizer_path: str):
    """Load caption model, feature extractor and tokenizer once and cache them."""
    caption_model = load_model(model_path)
    feature_extractor = load_model(feature_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    return caption_model, feature_extractor, tokenizer

def preprocess_image_pil(pil_img: Image.Image, img_size: int = 224):
    """Resize and normalize PIL image, return array suitable for extractor."""
    pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize((img_size, img_size))
    arr = img_to_array(pil_img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def sample_from_probs(probs: np.ndarray, temperature: float = 1.0, top_k: int = 0):
    """Sample an index from probability vector with temperature and optional top-k."""
    preds = np.asarray(probs).astype("float64")
    # temperature scaling
    if temperature != 1.0:
        preds = np.log(preds + 1e-9) / float(temperature)
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
    # top-k filtering
    if top_k and top_k > 0:
        top_indices = preds.argsort()[-top_k:]
        top_probs = preds[top_indices]
        top_probs = top_probs / top_probs.sum()
        chosen = np.random.choice(len(top_indices), p=top_probs)
        return int(top_indices[chosen])
    # otherwise sample full distribution
    return int(np.random.choice(len(preds), p=preds))

def keyword_to_emoji(caption: str) -> str:
    """Small heuristic to return an emoji based on caption keywords."""
    c = caption.lower()
    if any(k in c for k in ["dog", "cat", "horse", "puppy", "kitten"]):
        return "🐶"
    if any(k in c for k in ["beach", "sea", "ocean", "waves"]):
        return "🌊"
    if any(k in c for k in ["car", "road", "drive", "vehicle"]):
        return "🚗"
    if any(k in c for k in ["food", "pizza", "plate", "meal"]):
        return "🍽️"
    if any(k in c for k in ["person", "man", "woman", "people"]):
        return "🧑"
    return "🖼️"

# ---------- Captioning logic ----------
def generate_caption_from_pil(
    pil_img: Image.Image,
    caption_model,
    feature_extractor,
    tokenizer,
    max_length: int,
    img_size: int = 224,
    sampling: str = "argmax",    # 'argmax' | 'temperature' | 'top_k'
    temperature: float = 0.8,
    top_k: int = 5
) -> str:
    img_array = preprocess_image_pil(pil_img, img_size)
    features = feature_extractor.predict(img_array, verbose=0)

    in_text = "startseq"
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length)
        preds = caption_model.predict([features, seq], verbose=0)[0]

        if sampling == "argmax":
            idx = int(np.argmax(preds))
        elif sampling == "top_k":
            idx = sample_from_probs(preds, temperature=1.0, top_k=top_k)
        else:  # temperature sampling
            idx = sample_from_probs(preds, temperature=temperature, top_k=0)

        word = tokenizer.index_word.get(idx, None)
        if word is None:
            break
        if word == "endseq":
            break
        in_text += " " + word

    caption = in_text.replace("startseq", "").replace("endseq", "").strip()
    # nice capitalization
    if caption:
        caption = caption[0].upper() + caption[1:]
    return caption

# ---------- Streamlit UI ----------
def main():
    st.set_page_config(page_title="✨ Smart Captioner", layout="wide")
    st.markdown(
        """
        <style>
        .page-title { font-size:32px; font-weight:700; }
        .card { border-radius:12px; padding:12px; box-shadow: 0 4px 10px rgba(0,0,0,0.08); }
        .caption-box { background:#f0f8ff; padding:8px 10px; border-radius:8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header
    col1, col2 = st.columns([3,1])
    with col1:
        st.markdown('<div class="page-title">📸 Smart Image Caption Generator</div>', unsafe_allow_html=True)
        st.write("Upload photos and get natural captions. Try multiple images — responsive grid, sampling controls, and download captions.")
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/3784/3784184.png", width=80)

    st.markdown("---")

    # Sidebar (settings)
    with st.sidebar:
        st.header("Model & Settings")
        st.write("Load your trained model files (paths below are defaults).")
        model_path = st.text_input("Caption model path", value= "models\\model.keras")
        feature_path = st.text_input("Feature extractor path", value= "models\\feature_extractor.keras")
        tokenizer_path = st.text_input("Tokenizer path", value= "models\\tokenizer.pkl")
        st.markdown("---")

        st.subheader("Generation settings")
        sampling_mode = st.radio("Sampling method", ("argmax (stable)", "temperature (diverse)", "top-k (restricted)"))
        temperature = st.slider("Temperature (for sampling)", 0.3, 1.5, 0.8, 0.1)
        top_k = st.slider("Top-K (for top-k sampling)", 1, 50, 5, 1)
        st.markdown("---")
        st.subheader("Display")
        cols_per_row = st.selectbox("Images per row", [1,2,3,4], index=2)
        img_size = st.slider("Image test size (square)", 128, 512, 224, step=16)
        st.markdown("---")
        st.caption("Tip: increase temperature for more creative captions. Use argmax for consistent results.")

    # File uploader
    uploaded_images = st.file_uploader("Upload one or more images", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if not uploaded_images:
        st.info("Upload images to generate captions. Try a few scenic or object photos for best results.")
        return

    # Load models (cached)
    try:
        caption_model, feature_extractor, tokenizer = load_models(model_path, feature_path, tokenizer_path)
    except Exception as e:
        st.error(f"Failed loading models/tokenizer: {e}")
        return

    # determine max_length dynamically from model input shape (second input)
    try:
        seq_input_shape = caption_model.inputs[1].shape  # TensorShape
        max_length = int(seq_input_shape[1])
    except Exception:
        max_length = st.sidebar.slider("Max caption length (fallback)", 10, 60, 34)

    # choose sampling flags
    sampling_flag = "argmax"
    if sampling_mode.startswith("temperature"):
        sampling_flag = "temperature"
    elif sampling_mode.startswith("top-k"):
        sampling_flag = "top_k"

    # Process images and generate captions
    results: List[Tuple[str,str]] = []  # (filename, caption)
    total = len(uploaded_images)
    progress_text = st.empty()
    progress_bar = st.progress(0)

    # Responsive grid
    cols = st.columns(cols_per_row)
    for idx, uploaded_file in enumerate(uploaded_images):
        progress_text.text(f"Processing {idx+1}/{total}: {uploaded_file.name}")
        try:
            # Read PIL image
            pil_img = Image.open(io.BytesIO(uploaded_file.read()))
            # generate
            caption = generate_caption_from_pil(
                pil_img=pil_img,
                caption_model=caption_model,
                feature_extractor=feature_extractor,
                tokenizer=tokenizer,
                max_length=max_length,
                img_size=img_size,
                sampling=sampling_flag,
                temperature=temperature,
                top_k=top_k
            )
        except Exception as e:
            caption = f"[Error generating caption: {e}]"

        # Save result and display in grid
        results.append((uploaded_file.name, caption))
        col = cols[idx % cols_per_row]

        with col:
            # Card style
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(pil_img, use_container_width=True)
            # caption box with emoji
            emoji = keyword_to_emoji(caption)
            caption_html = f'<div class="caption-box"><strong>{emoji} {caption}</strong></div>'
            st.markdown(caption_html, unsafe_allow_html=True)
            # Buttons: copy, download single
            c1, c2 = st.columns([3,1])
            with c1:
                # copy to clipboard (works in browser via JS is tricky), so provide a text area and a copy button
                if st.button("Copy caption", key=f"copy_{idx}"):
                    st.write("Caption copied to clipboard: (press Ctrl+C)")  # hint only
                    st.text_area("Caption", value=caption, key=f"txt_{idx}", height=50)
            with c2:
                st.download_button(
                    label="Download",
                    data=caption.encode("utf-8"),
                    file_name=f"{uploaded_file.name.rsplit('.',1)[0]}_caption.txt",
                    mime="text/plain"
                )
            st.markdown('</div>', unsafe_allow_html=True)

        progress_bar.progress((idx+1)/total)
    progress_text.text("Done ✅")
    progress_bar.empty()

   
        # Show a table of generated captions and allow download CSV
    if results:
        df = pd.DataFrame(results, columns=["filename", "caption"])
        st.markdown("### Captions summary")
        st.dataframe(df)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download all captions (CSV)", data=csv_bytes, file_name="captions.csv", mime="text/csv")

# --- Footer (always visible, outside the if loop) ---
st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f0f2f6;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        color: #333;
    }
    </style>
    <div class="footer">
        🚀 Developed by <b>Sanchet Khemani</b>
    </div>
    """, unsafe_allow_html=True)



if __name__ == "__main__":
    main()