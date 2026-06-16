"""
test_rag.py
-----------
Evaluation framework for InsightForge RAG pipeline.
Runs a standard set of questions and checks response quality.
"""

# IMPORTS

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import load_sales_data
from src.retriever import build_knowledge_base, load_knowledge_base
from src.llm_engine import build_insight_chain, get_memory
from src.explainability import score_confidence


# STANDARD EVALUATION QUESTIONS & SUMMARY

EVAL_QUESTIONS = [
    "Which region had the highest total profit?",
    "What is the best performing product category by sales?",
    "Which region had the lowest profit margin?",
    "How did sales trend over time?",
    "Which customer segment generates the most revenue?",
]

def run_evaluation():
    print("Loading data...")
    df = load_sales_data()

    print("Loading knowledge base...")
    vectorstore = load_knowledge_base()
    memory = get_memory()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    chain = build_insight_chain(retriever, memory)

    print(f"\nRunning {len(EVAL_QUESTIONS)} evaluation queries...\n")
    print("=" * 60)

    results = []
    for i, question in enumerate(EVAL_QUESTIONS, 1):
        print(f"\nQ{i}: {question}")
        response = chain.invoke(question)
        docs = retriever.invoke(question)
        confidence = score_confidence(docs, df)
        print(f"Confidence: {confidence['label']} ({confidence['overall_score']})")
        print(f"Response preview: {response[:200]}...")
        results.append({
            "question": question,
            "confidence": confidence["label"],
            "score": confidence["overall_score"],
            "response": response,
        })

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    high = sum(1 for r in results if r["confidence"] == "High")
    medium = sum(1 for r in results if r["confidence"] == "Medium")
    low = sum(1 for r in results if r["confidence"] == "Low")
    print(f"High confidence: {high}/{len(results)}")
    print(f"Medium confidence: {medium}/{len(results)}")
    print(f"Low confidence: {low}/{len(results)}")
    avg = sum(r["score"] for r in results) / len(results)
    print(f"Average confidence score: {avg:.2f}")

if __name__ == "__main__":
    run_evaluation()