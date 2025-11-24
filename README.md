#📘 LLM Safety Category Classifier (Mini Safety Model)
---

A lightweight, fast, and fully reproducible project that classifies text into safety categories such as hate, violence, fraud, sexual content, self-harm, and benign.
This project uses OpenAI embeddings + a simple ML classifier to build a small but effective safety filter.

---

## 🚀 1. Problem Statement

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

## 🧩 2. Key Features
✔ Synthetic dataset created using GPT
✔ Embeddings generated using OpenAI text-embedding-3-large
✔ Simple classifier (Logistic Regression / SVM / XGBoost)
✔ Clear evaluation: accuracy, F1-score, confusion matrix
✔ Optional: Streamlit mini dashboard
✔ Minimal dependencies, no GPUs required
✔ End-to-end training notebook

---

<img width="3644" height="2190" alt="image" src="https://github.com/user-attachments/assets/ec9cdc70-7f21-4e4d-9ab3-10416d424e5d" />


---

## Demo Video
https://drive.google.com/file/d/1E_38w35ycMWfuCqqye3L1WPVcGkyvvgj/view?usp=share_link

---

## 🏗 3. Project Structure

<img width="718" height="418" alt="image" src="https://github.com/user-attachments/assets/2cb1ebbf-d8e0-49bf-9d0f-8cb80794585e" />

---

## 🛠 4. Tech Stack
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

--- 

## 📦 5. Installation
Create a virtual environment (recommended):
python3 -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

Install dependencies:
pip install -r requirements.txt

Add your OpenAI API key:
OPENAI_API_KEY=your_key_here

---

## 🧪 6. How the Model Works
1. You generate a synthetic dataset with 300–1000 labeled examples.
2. Each text is converted into a high-dimensional vector using OpenAI embeddings.
3. A lightweight classifier (Logistic Regression, SVM, or XGBoost) learns to map embeddings → label.
4. Evaluate accuracy, precision, recall, F1-score.
5. (Optional) Deploy a small web UI for demos.

---

## 📖 7. Usage
Run the notebook
1. Generate dataset
2. Embed text
3. Train and evaluate classifier

Run the dashboard
streamlit run app/app.py
You’ll get a small UI where you can paste a message and see its predicted category.

---

## 📊 8. Expected Results
1. 85–95% accuracy achievable with 600–800 synthetic samples
2. Very high precision/recall for clear categories
3. Slight confusion between borderline cases (natural limitation)

---

## 🧱 9. Requirements File (requirements.txt)
openai
pandas
numpy
scikit-learn
xgboost
matplotlib
streamlit
python-dotenv

---

## 🧠 10. Future Enhancements
- Add soft probability thresholds
- Add SHAP interpretation of embeddings
- Add contrastive hard-negative training
- Fine-tune small models (e.g., LoRA)
- Build a FastAPI inference server
- Deploy UI on HuggingFace Spaces

---

## ⭐ 11. Why This Project Is Valuable
This project demonstrates complete ML workflow skills, including:
- Synthetic dataset generation
- Embedding-based ML
- Multiclass classification
- Evaluation metrics
- Simple deployment

