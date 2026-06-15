# Giving LLM access to Superstore data so we prevent guessing and hallucinating.

# IMPORTS
import os
import pandas as pd
from langchain_openai import OpenAIEmbeddings    # Converts text into vectors
from langchain_chroma import Chroma  # Vector database that stores vectors
from langchain_core.documents import Document   # LangChain's wrapper for a chunk of text
from dotenv import load_dotenv

load_dotenv()

# These tell the retriever where to save the vector database and which embedding model to use.
CHROMA_DIR = "./data/chroma"
EMBEDDING_MODEL = "text-embedding-3-small"


# CHUNKING
# Takes a DataFrame as input and returns a list of Document objects.
def dataframe_to_documents(df: pd.DataFrame) -> list:
    docs = []

    # Chunk: monthly summary by region
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['month'] = df['order_date'].dt.to_period('M').astype(str)

    for (month, region), group in df.groupby(['month', 'region']):
        text = (
            f"Month: {month} | Region: {region}\n"
            f"Total Sales: ${group['sales'].sum():,.2f} | "
            f"Total Profit: ${group['profit'].sum():,.2f} | "
            f"Orders: {len(group)}"
        )
        docs.append(Document(page_content=text, metadata={"type": "monthly_region", "month": month, "region": region}))

    # Chunk: category summary
    for category, group in df.groupby('product_category'):
        text = (
            f"Category: {category}\n"
            f"Total Sales: ${group['sales'].sum():,.2f} | "
            f"Total Profit: ${group['profit'].sum():,.2f} | "
            f"Profit Margin: {group['profit'].sum()/group['sales'].sum()*100:.1f}% | "
            f"Orders: {len(group)}"
        )
        docs.append(Document(page_content=text, metadata={"type": "category_summary", "category": category}))

    return docs
# 4 regions × 48 months = 192 monthly/region chunks, plus 3 category chunks = 195 total


# KNOWLEDGE BASE
# Takes the DataFrame, converts it to chunks, embeds them and saves to disk
def build_knowledge_base(df: pd.DataFrame):
    docs = dataframe_to_documents(df)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print(f"Knowledge base built: {len(docs)} chunks indexed")
    return vectorstore

# Loads what's already saved so we don't have to rebuild every time
def load_knowledge_base():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)