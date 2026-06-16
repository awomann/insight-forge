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

## Future State

### UX Redesign: Dashboard View
The current multi-page navigation separates insight, evidence, fairness, 
and governance into distinct panels. A future redesign should consolidate 
these into a unified dashboard view:

- **Query input** at the top
- **Insight summary** with inline confidence score and fairness status
- **Collapsible detail panels** for evidence, assumptions, and full governance report
- **Drill-down navigation** preserved for deep analysis

This mirrors an admin/command center pattern — decision makers get a full 
picture at a glance, with the option to investigate further without leaving 
the view.


## Known Limitations

### Retrieval Coverage
The knowledge base chunks data by month/region, product category, region summary, 
and customer segment. Questions requiring time-series trend analysis (e.g. 
"how did sales change year over year?") cannot be answered accurately because 
no time trend summary chunks exist. This is a known gap for a future iteration.

### Confidence Scoring
The current confidence scoring system measures retrieval coverage (how many 
chunks were retrieved) and data volume, not answer quality. A response that 
says "I don't have enough data to answer" will still score High confidence if 
5 chunks were retrieved. A future improvement would detect hedging language 
in the LLM response and downgrade confidence accordingly.

### Data Freshness
The knowledge base is built once and persisted to disk. If the underlying 
dataset changes, the knowledge base must be manually rebuilt. A production 
system would automate this process.