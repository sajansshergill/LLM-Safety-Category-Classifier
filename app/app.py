import streamlit as st
from openai import OpenAI
import pickle
import numpy as np

# Load classifier
MODEL_PATH = "safety_model.pkl"

with open(MODEL_PATH, "rb") as f:
    clf = pickle.load(f)

# Initialize OpenAI client (requires OPENAI_API_KEY in env)
client = OpenAI()

# UI settings
st.set_page_config(page_title="Mini Safety Classifier", page_icon="🛡", layout="centered")
st.title("🛡 Mini LLM Safety Category Classifier")
st.write("Paste any text below and the model will classify it into safety categories.")


def get_embedding(text):
    """
    Generate OpenAI embedding for a given text.
    """
    try:
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return np.array(response.data[0].embedding)
    except Exception as e:
        st.error(f"Embedding Error: {e}")
        return None


# User Input
user_text = st.text_area("✏️ Enter text to classify:", height=150)

if st.button("🔍 Classify Text"):
    if not user_text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):
            emb = get_embedding(user_text)

            if emb is not None:
                pred = clf.predict([emb])[0]
                st.success(f"### 🧩 Predicted Category: **{pred}**")

                # Also show confidence
                proba = clf.predict_proba([emb])[0]
                labels = clf.classes_
                conf_dict = {labels[i]: float(proba[i]) for i in range(len(labels))}

                st.write("### 📊 Confidence Scores:")
                st.json(conf_dict)
