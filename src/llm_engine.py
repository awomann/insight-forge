# IMPORTS
# LangChain, OpenAI, dotenv, and the model constant

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI   # LLM
from langchain_core.prompts import ChatPromptTemplate   # Structures the prompt
from langchain_core.output_parsers import StrOutputParser   # Converts the LLM's response to a plain string
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# Constant for the model and a function to initialize it
MODEL = "gpt-4o"

def get_llm(temperature=0.2):
    return ChatOpenAI(model=MODEL, temperature=temperature)


# PROMPT
# The template that tells the LLM how to behave and what structure to respond in

# The template that gets sent to the LLM every time a user asks a question
INSIGHT_PROMPT = ChatPromptTemplate.from_template("""
You are InsightForge, an expert business intelligence analyst.
Use ONLY the retrieved data below to answer the question.
Do not invent numbers or trends not present in the context.

Retrieved Data:
{context}

Question: {question}

Provide your response in the following structure:
1. INSIGHT: A clear, direct answer to the question
2. EVIDENCE: The specific data points that support this insight
3. CONFIDENCE: High / Medium / Low — and why
4. CAVEATS: Any limitations or assumptions in this analysis
""")


# CHAIN
# Connects the retriever → prompt → LLM → output parser into one callable pipeline

# Builds the full RAG pipeline: retrieves relevant chunks → fills prompt → sends to LLM → returns response
def build_insight_chain(retriever):
    llm = get_llm()
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | INSIGHT_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain



