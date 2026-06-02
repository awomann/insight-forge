"""
app.py
------
InsightForge 2.0 — Main Streamlit Application
Explainable AI-Powered Business Intelligence
"""

import streamlit as st
import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InsightForge 2.0",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session State Init ────────────────────────────────────────────────────────
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

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/200x60?text=InsightForge+2.0", width=200)
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio(
        "",
        ["🏠 Home", "🔍 Query & Insights", "📋 Evidence", "⚖️ Fairness", "📄 Governance", "📊 Monitoring"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    # Data loader
    st.markdown("### Data")
    data_source = st.selectbox("Dataset", ["Default (sales.csv)", "Upload CSV"])
    if data_source == "Upload CSV":
        uploaded = st.file_uploader("Upload your CSV", type=["csv"])
        if uploaded:
            st.session_state.df = pd.read_csv(uploaded, parse_dates=["order_date"])
            st.success(f"Loaded {len(st.session_state.df):,} rows")
    else:
        default_path = Path(__file__).parent / "data" / "sales.csv"
        if default_path.exists() and st.session_state.df is None:
            from src.data_loader import load_sales_data
            st.session_state.df = load_sales_data()
            st.success(f"Loaded {len(st.session_state.df):,} rows")

    if st.session_state.df is not None:
        st.caption(f"{len(st.session_state.df):,} records loaded")

    # KB builder
    st.markdown("### Knowledge Base")
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-..."):
        st.warning("⚠️ Add your OpenAI API key to .env to enable AI features")
    else:
        if st.button("Build Knowledge Base", disabled=st.session_state.df is None):
            with st.spinner("Indexing data..."):
                from src.retriever import build_knowledge_base
                from src.llm_engine import get_memory
                st.session_state.vectorstore = build_knowledge_base(st.session_state.df)
                st.session_state.memory = get_memory()
                st.success("Knowledge base ready!")

# ── Pages ─────────────────────────────────────────────────────────────────────

if page == "🏠 Home":
    st.title("InsightForge 2.0")
    st.subheader("Explainable AI-Powered Business Intelligence")
    st.markdown("""
    InsightForge generates data-driven business insights with full transparency —
    every recommendation comes with evidence, confidence scores, fairness checks, and a governance report.

    **Getting started:**
    1. Load your dataset using the sidebar (or use the default sales dataset)
    2. Add your OpenAI API key to the `.env` file
    3. Click **Build Knowledge Base** in the sidebar
    4. Head to **Query & Insights** to start asking questions
    """)

    if st.session_state.df is not None:
        from src.data_loader import get_summary_stats
        stats = get_summary_stats(st.session_state.df)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Sales", f"${stats['total_sales']:,.0f}")
        col2.metric("Total Profit", f"${stats['total_profit']:,.0f}")
        col3.metric("Profit Margin", f"{stats['profit_margin']}%")
        col4.metric("Total Orders", f"{stats['total_orders']:,}")
        st.markdown(f"**Date range:** {stats['date_range']['start']} → {stats['date_range']['end']}")
        st.markdown(f"**Regions:** {', '.join(stats['regions'])} | **Categories:** {', '.join(stats['categories'])}")


elif page == "🔍 Query & Insights":
    st.title("Query & Insights")
    st.markdown("Ask a question about your data. InsightForge will retrieve relevant evidence and generate a grounded insight.")

    if st.session_state.vectorstore is None:
        st.info("Build the knowledge base first using the sidebar.")
    else:
        question = st.text_input("Your question", placeholder="e.g. Which region had the highest profit margin last quarter?")
        if st.button("Generate Insight") and question:
            with st.spinner("Retrieving data and generating insight..."):
                from src.llm_engine import build_insight_chain, parse_insight_response
                from src.explainability import build_evidence_chain, score_confidence, document_assumptions

                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
                chain = build_insight_chain(retriever, st.session_state.memory)
                response = chain.invoke(question)
                parsed = parse_insight_response(response)

                # Store retrieved docs for evidence panel
                docs = retriever.invoke(question)
                evidence = build_evidence_chain(question, docs, parsed["insight"])
                confidence = score_confidence(docs, st.session_state.df)
                assumptions = document_assumptions(question, st.session_state.df)

                st.session_state.last_insight = {
                    "question": question,
                    "parsed": parsed,
                    "evidence": evidence,
                    "confidence": confidence,
                    "assumptions": assumptions,
                    "raw": response,
                }
                st.session_state.history.append({"question": question, "insight": parsed["insight"]})

        if st.session_state.last_insight:
            ins = st.session_state.last_insight
            st.markdown("### Insight")
            st.success(ins["parsed"]["insight"])
            col1, col2 = st.columns(2)
            col1.markdown(f"**Confidence:** {ins['confidence']['label']} ({ins['confidence']['overall_score']})")
            col2.markdown(f"**Caveats:** {ins['parsed']['caveats']}")
            with st.expander("Full response"):
                st.text(ins["raw"])


elif page == "📋 Evidence":
    st.title("Evidence Panel")
    st.markdown("See exactly which data was retrieved and how it supports the insight.")

    if st.session_state.last_insight is None:
        st.info("Run a query first to see evidence.")
    else:
        ev = st.session_state.last_insight["evidence"]
        st.markdown(f"**Question:** {ev['question']}")
        st.markdown(f"**Chunks retrieved:** {ev['chunk_count']}")
        st.markdown("### Retrieved Data Chunks")
        for i, chunk in enumerate(ev["retrieved_chunks"]):
            with st.expander(f"Chunk {i+1} — {chunk['metadata'].get('type', 'unknown')}"):
                st.text(chunk["content"])
                st.caption(f"Relevance: {chunk['relevance_note']}")
                st.json(chunk["metadata"])

        st.markdown("### Confidence Breakdown")
        conf = st.session_state.last_insight["confidence"]
        for k, v in conf["components"].items():
            st.progress(v, text=f"{k.replace('_', ' ').title()}: {v:.0%}")

        st.markdown("### Assumptions")
        for a in st.session_state.last_insight["assumptions"]:
            st.markdown(f"- {a}")


elif page == "⚖️ Fairness":
    st.title("Fairness Assessment")
    st.markdown("Check whether insights and data distributions treat all segments equitably.")

    if st.session_state.df is None:
        st.info("Load a dataset to run fairness checks.")
    else:
        from src.fairness import run_full_fairness_check, document_known_biases
        results = run_full_fairness_check(st.session_state.df)

        status = "✅ No fairness issues detected" if results["overall_fair"] else f"⚠️ {results['total_flags']} flag(s) detected"
        st.markdown(f"### Overall Status: {status}")
        st.info(results["summary"])

        for check_name, check in results["checks"].items():
            with st.expander(check_name.replace("_", " ").title()):
                st.markdown(f"**Metric:** {check['metric']} | **Overall Mean:** {check['overall_mean']}")
                st.dataframe(pd.DataFrame(list(check["segment_values"].items()), columns=["Segment", "Value"]))
                if check["disparities"]:
                    st.warning("Disparities detected:")
                    for d in check["disparities"]:
                        st.markdown(f"- **{d['segment']}**: {d['deviation_pct']}% from mean — {d['flag']}")

        st.markdown("### Known Data Biases")
        for bias in document_known_biases(st.session_state.df):
            st.markdown(f"- {bias}")


elif page == "📄 Governance":
    st.title("Governance Report")
    st.markdown("Generate and review compliance-ready governance reports for AI-generated insights.")

    if st.session_state.last_insight is None:
        st.info("Run a query first to generate a governance report.")
    else:
        ins = st.session_state.last_insight
        if st.button("Generate Governance Report"):
            with st.spinner("Generating governance report..."):
                from src.fairness import run_full_fairness_check
                from src.governance import generate_governance_report, format_report_markdown

                fairness = run_full_fairness_check(st.session_state.df)
                report = generate_governance_report(
                    question=ins["question"],
                    insight=ins["parsed"]["insight"],
                    evidence_chain=ins["evidence"],
                    confidence=ins["confidence"],
                    fairness_results=fairness,
                    assumptions=ins["assumptions"],
                )
                st.session_state.last_governance = report

        if st.session_state.last_governance:
            from src.governance import format_report_markdown
            report = st.session_state.last_governance
            st.markdown(f"### Status: {report['recommendation_status']}")
            st.markdown("#### Executive Summary")
            st.info(report["executive_summary"])
            with st.expander("Full Report (Compliance)"):
                st.markdown(format_report_markdown(report))
            st.download_button(
                "Download Report (Markdown)",
                format_report_markdown(report),
                file_name="governance_report.md",
                mime="text/markdown",
            )


elif page == "📊 Monitoring":
    st.title("Monitoring Dashboard")
    st.markdown("Track system health, recommendation accuracy, and fairness drift over time.")
    st.info("Monitoring dashboard will populate as you run more queries.")

    if st.session_state.history:
        st.markdown("### Query History")
        for i, h in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Query {len(st.session_state.history) - i}: {h['question'][:60]}..."):
                st.markdown(h["insight"])
    else:
        st.markdown("No queries yet — run some insights to start building history.")
