from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

PROMPT_TEMPLATE = """
You are a helpful assistant that answers questions about a document.
Use the context below to answer the question as clearly and completely as possible.
If the context does not contain enough information, say what you do know and note the limitation.

Context:
{context}

Question: {question}

Answer:
"""

def build_rag_chain(vectorstore):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return chain