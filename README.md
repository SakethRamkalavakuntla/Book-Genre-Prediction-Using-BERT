# 📚 Book Genre Prediction Flask App using BERT with AWS Deployment

A state-of-the-art book genre prediction web application that uses **BERT** for natural language understanding to classify the genre of a book from its summary.  
Built with **Python**, **Hugging Face Transformers**, and a clean web interface using **Flask** and **Tailwind CSS**, this app is deployed on **AWS EC2** for public access.


---

## 🧠 How It Works

1. **Input:** User submits a book summary via the web interface.  
2. **Text Processing:**  
   - Text cleaning, tokenization, and conversion to BERT-compatible input.  
3. **Model:**  
   - Fine-tuned `bert-base-cased` model for sequence classification.  
   - Loads pretrained weights from a hosted model file (`best_bert_model.pt`).  
4. **Output:**  
   - Predicts one of the book genres:  
     - Fantasy  
     - Science Fiction  
     - Crime Fiction  
     - Historical Novel  
     - Horror  
     - Thriller

---
## 🖼️ Demo Screenshots

### 🔹 Home Page (User Input Form)  
[Index Page](images/index_page.png)  

### 🔹 Prediction Result Page  
[Result Page](images/result_page.png)

---

## 🚀 Features

- 📖 Predicts book genres based on summaries using a fine-tuned BERT model.  
- 🤖 Uses Hugging Face’s `BertForSequenceClassification` for advanced text understanding.  
- 🖥️ Simple and responsive user interface styled with **Tailwind CSS**.  
- 🌐 Web server built with **Flask**.  
- ☁️ Deployed on **AWS EC2** with Gunicorn and Nginx for production-grade hosting.

---

## ⚙️ Tech Stack

- **Frontend:** HTML5, Tailwind CSS  
- **Backend:** Flask (Python)  
- **ML Libraries:** Hugging Face Transformers, PyTorch, pandas  
- **Deployment:** AWS EC2 (Ubuntu, Gunicorn, Nginx)

---

## 🤖 About the BERT Model

This project uses **BERT (Bidirectional Encoder Representations from Transformers)**, a powerful pre-trained language model developed by Google AI for natural language understanding. Unlike traditional models, BERT reads text bidirectionally, capturing deeper context and meaning from both left and right of each word.

### Key Points about the BERT model in this app:

- **Pre-trained Model Used:** `bert-base-cased` — a version of BERT trained on a large corpus with case-sensitive tokens.  
- **Fine-tuning:** The model is fine-tuned on a labeled book summary dataset for the specific task of **genre classification**. This adapts BERT’s general language knowledge to the specific nuances of book summaries.  
- **Architecture:** Uses `BertForSequenceClassification` from Hugging Face Transformers, adding a classification head on top of the base BERT encoder to output genre predictions.
- **Training:** Fine-tuned using PyTorch with early stopping for best performance.
- **Model File:** The trained weights are saved as `best_bert_model.pt` and loaded dynamically when the Flask app starts.  
- **Advantages:**  
  - Handles complex language patterns and context better than traditional models like Naive Bayes.  
  - Achieves higher accuracy and robustness in classifying book genres from text summaries.

This approach allows the app to understand subtle semantic cues in the book summaries, resulting in more precise and reliable genre predictions.

---


## 🏗️ Deployment on AWS EC2

- Uses **Gunicorn** as the WSGI HTTP server.  
- Configured **Nginx** as a reverse proxy to forward requests to Gunicorn.  
- Runs Gunicorn inside a Python virtual environment for dependency isolation.  
- Set up a **systemd** service for automatic start, restart, and management of the app.  
- AWS security group configured to allow HTTP (port 80) and the app port (e.g., 5000 or 8000) as needed.  

### 🔧 Notes  
- The model file `best_bert_model.pt` is hosted on AWS S3 for easy loading during app startup.  
- Tailwind CSS is used for styling instead of Bootstrap (used in the previous ML project).  
- The app is designed to be scalable and easily maintainable for further model upgrades.

---

## 🧪 Run Locally

1. **Clone the repo**
   ```bash
   git clone https://github.com/SakethRamkalavakuntla/Book-Genre-Prediction-Using-BERT.git
   cd Book-Genre-Prediction-Using-BERT
