# AI ASSISTANCE CHATBOT

# 🧠 AI Chatbot for UK Visa Assistance  

This project is an **AI-powered conversational assistant** designed to help users navigate the complex UK visa application process.  
It uses **Retrieval-Augmented Generation (RAG)** and **FAISS vector databases** to provide fast, accurate, and trustworthy responses from official UK government documentation.

---

## 🚀 Key Features  
- Indexes 105 pages (~26,570 words) of GOV.UK visa documentation.  
- Uses **FAISS vector search** for relevant document retrieval.  
- Combines with **Google Generative AI** for natural, context-aware answers.  
- Achieves **95% accuracy** and **sub-second response time**.  
- Built with **Flask** and **Python**, easily extendable for other policy domains.  

---

## 🧰 Tech Stack  
| Category | Tools / Libraries |
|-----------|------------------|
| Programming | Python 3.10+ |
| Framework | Flask |
| AI / NLP | Google Generative AI API |
| Vector Database | FAISS |
| Evaluation | scikit-learn |
| Front-End | HTML / CSS / Bootstrap |
| Data Handling | Pandas, NumPy |

---

## 📁 Project Structure
```
chatbot/
│
├── app.py                       # Flask app entry point
├── requirements.txt              # Dependencies
├── vectorDB_performance_test.py  # Vector index test scripts
├── performance_test.py           # Benchmark testing
├── performance_evidence.csv      # Evaluation metrics
├── static/                       # Front-end assets
├── templates/                    # HTML templates
├── README.md                     # Documentation
└── .gitignore                    # Ignore sensitive files like .env
```

---

## ⚙️ Installation & Setup  

1. **Clone the repository**
   ```bash
   git clone https://github.com/ahmedaqeel3122-sys/chatbot.git
   cd chatbot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate     # For Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create `.env` file**
   Add your API key securely:
   ```
   GOOGLE_API_KEY="your_api_key_here"
   ```

5. **Run the app**
   ```bash
   python app.py
   ```
   Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) to test locally.

---

## 🧪 Benchmark Results
| Metric | Result |
|--------|---------|
| Documents processed | 105 (26,570 words) |
| Average response time | < 1 second |
| Ranked accuracy | 95% |
| Functional tests | 15 successful query tests |

---

## 🧩 How It Works  
1. All visa policy text is embedded using FAISS vector representations.  
2. When a user asks a question, the system retrieves the most relevant document chunks.  
3. These chunks are passed to a **Generative AI model** for natural language synthesis.  
4. The chatbot returns a context-grounded, accurate, and human-like answer.

---
 

## 🧑‍💻 Author  
**Aqeel Ahmed**  
MSc Big Data Technology – Glasgow Caledonian University London  
📧 [ahmedaqeel3122@gmail.com](mailto:ahmedaqeel3122@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/aqeel-ahmed-5049a8312)  
✍️ [Medium](https://medium.com/@ahmedaqeel3122)

---