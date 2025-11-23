📘 LLM Safety Category Classifier (Mini Safety Model)
---

A lightweight, fast, and fully reproducible project that classifies text into safety categories such as hate, violence, fraud, sexual content, self-harm, and benign.
This project uses OpenAI embeddings + a simple ML classifier to build a small but effective safety filter.

---

🚀 1. Problem Statement

Modern AI applications must detect unsafe or harmful content before generating responses.
Large safety models are powerful but often expensive and slow to experiment with.

Goal:
Build a small, fast, efficient safety classifier that categorizes text into:
- hate
- violence
- fraud
- sexual_content
- self_harm
- benign


This model is ideal for:
- Prototyping safety filters
- Research & learning
- Demonstrating end-to-end ML pipeline skills
- Resume/GitHub portfolio projects
- Fast on-device or API-side content moderation

---

🧩 2. Key Features
✔ Synthetic dataset created using GPT
✔ Embeddings generated using OpenAI text-embedding-3-large
✔ Simple classifier (Logistic Regression / SVM / XGBoost)
✔ Clear evaluation: accuracy, F1-score, confusion matrix
✔ Optional: Streamlit mini dashboard
✔ Minimal dependencies, no GPUs required
✔ End-to-end training notebook

---

🏗 3. Project Structure
📦 mini-safety-classifier
│

├── data/

│   ├── synthetic_data.jsonl        # generated dataset
│   ├── safety_embeddings.pkl       # precomputed embeddings

│
├── notebooks/
│   ├── 01_generate_data.ipynb      # synthetic dataset creation
│   ├── 02_train_classifier.ipynb   # embedding + training + evaluation
│
├── app/
│   ├── app.py                      # optional Streamlit mini UI
│
├── models/
│   ├── safety_model.pkl            # trained classifier
│
├── README.md                       # project documentation
├── requirements.txt                # Python dependencies
└── .env.example                    # example for OpenAI API key
