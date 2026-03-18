
import joblib
import re
import os

# Load trained pipeline
base_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir,"model","product_classifier.pkl")
model = joblib.load(model_path)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

while True:
    user_input = input("\nEnter a product title (or type 'exit' to quit): ")

    if user_input.lower() == "exit":
        print("Exiting...")
        break

    cleaned_text = clean_text(user_input)

    # Pipeline does TF-IDF + prediction internally
    prediction = model.predict([cleaned_text])

    print(f"Predicted category: {prediction[0]}")
