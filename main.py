from src.ingest import load_and_split
from src.retriever import build_vectorstore
from src.rag_chain import build_rag_chain

PDF_PATH = "data/MFML.pdf"

chunks = load_and_split(PDF_PATH)
vectorstore = build_vectorstore(chunks)
chain = build_rag_chain(vectorstore)

print("\n=== RAG Demo — ask questions about your document ===")
print("Type 'quit' to exit\n")

while True:
    question = input("Your question: ").strip()
    if question.lower() == "quit":
        break
    if not question:
        continue

    result = chain.invoke({"query": question})
    print(f"\nAnswer: {result['result']}")
    print(f"Sources: {[doc.metadata for doc in result['source_documents']]}\n")