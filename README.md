# Ready Tensor Publication Explorer

Conversational assistant to explore Ready Tensor publications using natural language.

## Features
- Summarize publications
- List models/tools used
- Show limitations/assumptions
- Retrieval-Augmented Generation (RAG) with semantic search

## Setup (Windows)

### 1. Clone the repo
```powershell
git clone https://github.com/<your-username>/ready-tensor-pub-exp.git
cd ready-tensor-pub-exp
```

### 2. Setup the env
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
copy configs\settings.example.env .env
```

### 3. Edit env
```powershell
# Edit .env to add your OpenAI API key
```

### 4. Execute commands
```powershell
# Execute ingest_database.py once in order to populate database
python src\ingest_database.py

# run chatbot.py to chat against the content of PDF file
python src\chatbot.py

# Questions that could be asked from PDF file
1. In 2 words: what was built in last 5 months?
2. How Can I Add a Memory to My RAG or AI Agent?
3. What is the advantage of using Proper environment and dependency management?

```