from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pickle
import os

app = Flask(__name__)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "best_bert_model.pt"
model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=6)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Predict genre from summary
def predict_genre(summary):
    inputs = tokenizer(summary, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()
    
    genre = label_encoder.inverse_transform([predicted_label])[0]
    return genre

# @app.route("/", methods=["GET", "POST"])
# def index():
#     prediction = None
#     if request.method == "POST":
#         summary = request.form["summary"]
#         if summary.strip():
#             prediction = predict_genre(summary)
#     return render_template("index.html", prediction=prediction)
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    summary = ""
    short_summary = ""

    if request.method == "POST":
        summary = request.form["summary"]
        if summary.strip():
            prediction = predict_genre(summary)
            # Limit to first 100 characters
            short_summary = summary[:100]

    return render_template(
        "index.html",
        showresult=prediction is not None,
        Prediction=prediction,
        text=short_summary
    )


if __name__ == "__main__":
    app.run(debug=False)
