# IMPORTS

import pandas as pd
import numpy as np


# EVIDENCE

# Evidence trail: question + retrieved chunks → dictionary of data used by the LLM
def build_evidence_chain(question: str, retrieved_docs: list) -> dict:
    return {
        "question": question,
        "chunks_used": len(retrieved_docs),
        "retrieved_data": [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in retrieved_docs
        ]
    }


# CONFIDENCE

# Estimates how trustworthy an insight is based on how much data supports it
def score_confidence(retrieved_docs: list, df: pd.DataFrame) -> dict:
    chunk_score = min(len(retrieved_docs) / 5, 1.0)    # How many relevant chunks were retrieved (5 is full coverage)
    volume_score = min(len(df) / 10000, 1.0)    # how much data supports the insight; produces a High/Medium/Low confidence label
    overall = round((chunk_score * 0.6 + volume_score * 0.4), 2)
    label = "High" if overall >= 0.75 else "Medium" if overall >= 0.5 else "Low"
    
    return {
        "label": label,
        "overall_score": overall,
        "components": {
            "retrieval_coverage": round(chunk_score, 2),
            "data_volume": round(volume_score, 2)
        }
    }


# ASSUMPTIONS

# Documents what the system assumed when generating the answer
def document_assumptions(df: pd.DataFrame) -> list:
    return [
        f"Analysis covers {df['Order Date'].min().date()} to {df['Order Date'].max().date()}",
        f"Dataset contains {len(df):,} transactions across {df['Region'].nunique()} regions",
        "Sales figures are in USD",
        "Profit is calculated as revenue minus cost of goods sold",
        "Historical data may not reflect current business conditions"
    ]