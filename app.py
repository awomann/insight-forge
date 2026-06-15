"""
app.py
------
InsightForge 2.0 — Main Streamlit Application
Explainable AI-Powered Business Intelligence
"""


# IMPORTS & CONFIG

import streamlit as st
import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Tells Streamlit how to set up the browser tab and layout before anything else renders.
st.set_page_config(
    page_title="InsightForge 2.0",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# SESSION STATE

# Persist dataframe, vectorstore, memory, query history, and last insight/governance report across interactions
if "df" not in st.session_state:
    st.session_state.df = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "memory" not in st.session_state:
    st.session_state.memory = None
if "history" not in st.session_state:
    st.session_state.history = []
if "last_insight" not in st.session_state:
    st.session_state.last_insight = None
if "last_governance" not in st.session_state:
    st.session_state.last_governance = None


# SIDEBAR

# Navigation radio buttons, auto-loads sales.csv on startup, 
# and provides a button to build the ChromaDB knowledge base
with st.sidebar:
    st.title("InsightForge 2.0")
    st.markdown("---")
    
    # Navigation — radio buttons that control which page is displayed
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔍 Query & Insights", "📋 Evidence", "⚖️ Fairness", "📄 Governance", "📊 Monitoring"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    
    # Data — auto-loads sales.csv on startup and shows record count
    st.markdown("### Data")
    default_path = Path(__file__).parent / "data" / "sales.csv"
    if default_path.exists() and st.session_state.df is None:
        from src.data_loader import load_sales_data
        st.session_state.df = load_sales_data()
    if st.session_state.df is not None:
        st.caption(f"{len(st.session_state.df):,} records loaded")
    
    # Knowledge Base — button that chunks the data, creates embeddings, and saves to ChromaDB
    st.markdown("### Knowledge Base")
    if st.button("Build Knowledge Base", disabled=st.session_state.df is None):
        with st.spinner("Indexing data..."):
            from src.retriever import build_knowledge_base
            from src.llm_engine import get_memory
            st.session_state.vectorstore = build_knowledge_base(st.session_state.df)
            st.session_state.memory = get_memory()
            st.success("Knowledge base ready!")


# PAGES

# Home —
if page == "🏠 Home":
    st.title("InsightForge 2.0")
    st.subheader("Explainable AI-Powered Business Intelligence")
    st.markdown("""
    InsightForge generates data-driven business insights with full transparency —
    every recommendation comes with evidence, confidence scores, fairness checks, and a governance report.

    **Getting started:**
    1. Click **Build Knowledge Base** in the sidebar
    2. Head to **Query & Insights** to start asking questions
    """)

# Query & Insights
elif page == "🔍 Query & Insights":
    st.title("Query & Insights")
    st.markdown("Ask a question about your data.")

    # Guard — knowledge base must be built before querying
    if st.session_state.vectorstore is None:
        st.info("Build the knowledge base first using the sidebar.")
    else:
        # Input — text box for the user's question and a button to trigger the chain
        question = st.text_input("Your question", placeholder="e.g. Which region had the highest profit?")
        if st.button("Generate Insight") and question:
            with st.spinner("Generating insight..."):
                from src.retriever import load_knowledge_base
                from src.llm_engine import build_insight_chain
                from src.explainability import build_evidence_chain, score_confidence, document_assumptions

                # Run the RAG chain and retrieve supporting docs
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
                chain = build_insight_chain(retriever, st.session_state.memory)
                response = chain.invoke(question)
                docs = retriever.invoke(question)

                # Store everything in session state for other panels to use
                st.session_state.last_insight = {
                    "question": question,
                    "response": response,
                    "evidence": build_evidence_chain(question, docs),
                    "confidence": score_confidence(docs, st.session_state.df),
                    "assumptions": document_assumptions(st.session_state.df),
                }
                st.session_state.history.append({"question": question, "response": response})

        # Display — show the last generated insight
        if st.session_state.last_insight:
            st.markdown("### Insight")
            st.success(st.session_state.last_insight["response"])