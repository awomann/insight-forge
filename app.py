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

# Home — landing page with app description and getting started instructions
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

# Query & Insights — text input that triggers the RAG chain and displays the generated insight
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

# Evidence — shows which data chunks were retrieved and the confidence breakdown
elif page == "📋 Evidence":
    st.title("Evidence Panel")
    st.markdown("See exactly which data was retrieved to generate the insight.")

    # Guard — a query must have been run first
    if st.session_state.last_insight is None:
        st.info("Run a query first to see evidence.")
    else:
        ev = st.session_state.last_insight["evidence"]
        
        # Retrieved chunks
        st.markdown(f"**Question:** {ev['question']}")
        st.markdown(f"**Chunks retrieved:** {ev['chunks_used']}")
        st.markdown("### Retrieved Data")
        for i, chunk in enumerate(ev["retrieved_data"]):
            with st.expander(f"Chunk {i+1} — {chunk['metadata'].get('type', 'unknown')}"):
                st.text(chunk["content"])
                st.json(chunk["metadata"])

        # Confidence breakdown
        st.markdown("### Confidence")
        conf = st.session_state.last_insight["confidence"]
        st.markdown(f"**Level:** {conf['label']} ({conf['overall_score']})")
        for k, v in conf["components"].items():
            st.progress(v, text=f"{k.replace('_', ' ').title()}: {v:.0%}")

        # Assumptions
        st.markdown("### Assumptions")
        for a in st.session_state.last_insight["assumptions"]:
            st.markdown(f"- {a}")

# Fairness — runs disparity checks across regions and customer segments
elif page == "⚖️ Fairness":
    st.title("Fairness Assessment")
    st.markdown("Checks whether any region or customer segment deviates significantly from the average.")

    # Guard — data must be loaded before running checks
    if st.session_state.df is None:
        st.info("Load a dataset to run fairness checks.")
    else:
        from src.fairness import run_full_fairness_check

        # Run all checks and display overall status
        results = run_full_fairness_check(st.session_state.df)
        if results["overall_fair"]:
            st.success("✅ No significant fairness issues detected.")
        else:
            st.warning(f"⚠️ {results['total_flags']} disparity flag(s) detected.")
        st.markdown(results["summary"])

        # Regional check — one expander per check type
        st.markdown("### Regional Fairness")
        regional = results["checks"]["regional"]
        st.markdown(f"**Metric:** {regional['metric']} | **Overall Mean:** {regional['overall_mean']}")
        st.dataframe(
            pd.DataFrame(list(regional["segment_values"].items()), columns=["Region", "Value"]),
            use_container_width=True
        )
        if regional["disparities"]:
            for d in regional["disparities"]:
                st.warning(f"**{d['segment']}**: {d['deviation_pct']}% from mean {d['flag']}")

        # Segment check — Consumer, Corporate, Home Office
        st.markdown("### Segment Fairness")
        segment = results["checks"]["segment"]
        st.markdown(f"**Metric:** {segment['metric']} | **Overall Mean:** {segment['overall_mean']}")
        st.dataframe(
            pd.DataFrame(list(segment["segment_values"].items()), columns=["Segment", "Value"]),
            use_container_width=True
        )
        if segment["disparities"]:
            for d in segment["disparities"]:
                st.warning(f"**{d['segment']}**: {d['deviation_pct']}% from mean {d['flag']}")

# Governance — packages the last insight into a structured compliance report
elif page == "📄 Governance":
    st.title("Governance Report")
    st.markdown("Generate a compliance-ready report for any AI-generated insight.")

    # Guard — a query must have been run first
    if st.session_state.last_insight is None:
        st.info("Run a query first to generate a governance report.")
    else:
        # Button triggers report generation — pulls from session state
        if st.button("Generate Governance Report"):
            with st.spinner("Generating report..."):
                from src.fairness import run_full_fairness_check
                from src.governance import generate_governance_report

                ins = st.session_state.last_insight
                fairness = run_full_fairness_check(st.session_state.df)

                report = generate_governance_report(
                    question=ins["question"],
                    insight=ins["response"],
                    evidence=ins["evidence"],
                    confidence=ins["confidence"],
                    fairness=fairness,
                    assumptions=ins["assumptions"],
                )
                st.session_state.last_governance = report

        # Display — show the report once it exists in session state
        if st.session_state.last_governance:
            from src.governance import format_report_markdown

            report = st.session_state.last_governance
            st.markdown(f"### Status: {report['status']}")

            # Confidence and fairness summary
            col1, col2 = st.columns(2)
            col1.metric("Confidence", report["confidence"]["label"])
            col2.metric("Fairness Flags", report["fairness"]["total_flags"])

            with st.expander("Full Report"):
                st.markdown(format_report_markdown(report))
            st.download_button(
                "Download Report",
                data=format_report_markdown(report),
                file_name="governance_report.md",
                mime="text/markdown",
            )