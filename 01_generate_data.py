import json
import re
from openai import OpenAI

client = OpenAI()

CATEGORIES = {
    "hate": "Generate sentences expressing hate, insults, discrimination, or hostility (without real slurs).",
    "violence": "Generate sentences describing threats, physical harm, weapons, or violent intentions.",
    "fraud": "Generate scam attempts, phishing messages, fake offers, or deceptive financial messages.",
    "sexual_content": "Generate sexually suggestive messages (avoid graphic detail).",
    "self_harm": "Generate sentences expressing self-harm, suicidal thoughts, or intent to hurt oneself.",
    "benign": "Generate harmless everyday sentences — greetings, questions, small talk, neutral lines."
}

OUTPUT_FILE = "synthetic_data.jsonl"


def extract_python_list(raw):
    """
    Extracts the FIRST Python list found in the text.
    Works even if model outputs:
        - code fences
        - variable assignments
        - python code
        - extra text
    """
    # Remove markdown code fences
    raw = raw.replace("```python", "").replace("```json", "").replace("```", "").strip()

    # Regex: find the first [...] block
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        raise ValueError(f"❌ No list found in model output:\n{raw}")

    list_str = match.group(0)

    try:
        return eval(list_str)
    except Exception as e:
        print("❌ Failed to eval the extracted list:")
        print(list_str)
        raise e


def generate_examples():
    with open(OUTPUT_FILE, "w") as f:
        for label, instruction in CATEGORIES.items():
            print(f"\n🔵 Generating samples for: {label}")

            prompt = f"""
            Generate 15 unique sentences for the category '{label}'.
            {instruction}
            Return ONLY a Python list of strings.
            """

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )

            raw = response.choices[0].message.content

            # Extract list
            sentences = extract_python_list(raw)

            if not isinstance(sentences, list):
                raise ValueError("❌ Extracted output is not a list!")

            # Write each sample to JSONL
            for text in sentences:
                record = {"text": text, "label": label}
                f.write(json.dumps(record) + "\n")

    print(f"\n✅ Dataset successfully created → {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_examples()
