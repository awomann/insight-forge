# InsightForge 2.0

Explainable AI-Powered Business Intelligence

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 4. Run the app
streamlit run app.py
```

## Project Structure

```
Insight Forge/
├── app.py                  # Streamlit application
├── requirements.txt
├── .env.example
├── data/
│   └── sales.csv           # Sample dataset (2,000 orders)
├── src/
│   ├── data_loader.py      # Data ingestion & metrics
│   ├── retriever.py        # ChromaDB vector store & RAG
│   ├── llm_engine.py       # LLM chains & memory
│   ├── explainability.py   # Evidence chains & confidence scoring
│   ├── fairness.py         # Bias detection & fairness metrics
│   └── governance.py       # Report generation & audit trail
└── notebooks/              # Exploration notebooks
```
