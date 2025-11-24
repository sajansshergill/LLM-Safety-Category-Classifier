import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

EMBED_FILE = "safety_embeddings.pkl"
MODEL_FILE = "safety_model.pkl"


def load_embeddings():
    """
    Load the stored embeddings and convert them to numpy arrays.
    """
    df = pd.read_pickle(EMBED_FILE)
    df["embedding"] = df["embedding"].apply(np.array)
    return df


def train_model(df):
    """
    Train a Logistic Regression classifier on embedding vectors.
    """
    X = np.vstack(df["embedding"].values)
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🔵 Training Logistic Regression classifier...")
    clf = LogisticRegression(max_iter=4000)
    clf.fit(X_train, y_train)

    # Evaluation
    y_pred = clf.predict(X_test)

    print("\n=== MODEL PERFORMANCE ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return clf


def save_model(clf):
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(clf, f)
    print(f"\n✅ Model saved → {MODEL_FILE}")


def main():
    print("📦 Loading embeddings...")
    df = load_embeddings()

    clf = train_model(df)
    save_model(clf)


if __name__ == "__main__":
    main()
