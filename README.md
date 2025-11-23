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

<img width="1182" height="784" alt="image" src="https://github.com/user-attachments/assets/7a2b957a-8021-4ff0-9c84-b77e28fc8cb6" />

---

🛠 4. Tech Stack
Core
- Python 3.10+
- OpenAI API (embeddings)
- scikit-learn (ML models)
- XGBoost (optional)
- NumPy / Pandas

Visualization
- Matplotlib
- Seaborn (optional)

Optional
- Streamlit (UI demo)
- Jupyter / VSCode Notebooks
