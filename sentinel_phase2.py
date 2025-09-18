

# Import packages
import os
import random
import time
import requests
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA

# Set your Google API key
os.environ["GOOGLE_API_KEY"] = "YOUR GOOGLE API KEY"

# Card Validation Function
def validate_card(card_number: str) -> str:
    """
    Validate a card number using BIN lookup and return scheme, type, brand, and bank name.
    """
    bin_number = card_number[:8] if len(card_number) >= 8 else card_number[:6]
    url = f"https://lookup.binlist.net/{bin_number}"
    try:
        time.sleep(1)
        response = requests.get(url)
        if response.status_code == 429:
            return "Rate limit hit. Please try again."
        if response.status_code != 200:
            return f"Failed for BIN {bin_number}. Status: {response.status_code}"
        data = response.json()
        return (
            f"Scheme: {data.get('scheme','N/A')} | "
            f"Type: {data.get('type','N/A')} | "
            f"Brand: {data.get('brand','N/A')} | "
            f"Bank: {data.get('bank',{}).get('name','N/A')}"
        )
    except Exception as e:
        return f"Error: {e}"


# Transaction Simulation Function
def simulate_transaction(amount, memo):
    try:
        response = requests.post(
            "https://jsonplaceholder.typicode.com/posts",
            json={"amount": amount, "memo": memo}
        )
        if response.status_code in (200, 201):
            data = response.json()
            txn_id = data.get("id", random.randint(1000, 9999))
            return f"Transaction simulated ✅ ID: {txn_id} | Amount: {amount} | Memo: {memo}"
        return f"Failed to simulate transaction. Status {response.status_code}"
    except Exception as e:
        return f"Error: {e}"


# Fraud Risk Function
def get_fraud_risk(transaction_id: str) -> str:
    score = round(random.uniform(0, 1), 2)
    risk = "High" if score > 0.7 else "Medium" if score > 0.4 else "Low"
    return f"Transaction {transaction_id} → Risk: {risk} | Score: {score}"


# PDF Q&A Function (RAG Agent)
def pdf_qa(pdf_file, question):
    if pdf_file is None or question.strip() == "":
        return "Please upload a PDF and enter a question."

    try:
        # Handle Gradio file input safely
        file_path = pdf_file.name if hasattr(pdf_file, "name") else pdf_file
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Split text
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(documents)

        # Embeddings + Vector store
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.environ["GOOGLE_API_KEY"]
        )
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        # LLM
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-1.5-pro-002",
            google_api_key=os.environ["GOOGLE_API_KEY"],
        )

        # Retrieval QA
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        result = rag_chain.invoke({"query": question})
        answer = result["result"]

        # Collect sources
        sources = []
        for i, doc in enumerate(result["source_documents"], start=1):
            snippet = doc.page_content[:150].replace("\n", " ")
            sources.append(f"{i}. {snippet}...")

        sources_text = "\n".join(sources) if sources else "No sources found."

        return f"📌 Answer: {answer}\n\n📖 Sources:\n{sources_text}"

    except Exception as e:
        return f"Error processing PDF: {e}"


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🔐 Sentinel Secure AI (Multi-Tool Interface)")

    # Card Validation
    with gr.Tab("Card Validation"):
        card_input = gr.Textbox(label="Enter Card Number")
        card_output = gr.Textbox(label="Card Validation Output", interactive=False)
        card_btn = gr.Button("Validate Card")
        card_btn.click(fn=validate_card, inputs=card_input, outputs=card_output)

    # Transaction Simulation
    with gr.Tab("Transaction Simulation"):
        amt_input = gr.Number(label="Amount", value=0.01, minimum=0.01)
        memo_input = gr.Textbox(label="Memo")
        txn_output = gr.Textbox(label="Transaction Simulation Output", interactive=False)
        txn_btn = gr.Button("Simulate Transaction")
        txn_btn.click(fn=simulate_transaction, inputs=[amt_input, memo_input], outputs=txn_output)

    # Fraud Risk
    with gr.Tab("Fraud Risk"):
        txn_id_input = gr.Textbox(label="Transaction ID")
        risk_output = gr.Textbox(label="Fraud Risk Result", interactive=False)
        risk_btn = gr.Button("Check Risk")
        risk_btn.click(fn=get_fraud_risk, inputs=txn_id_input, outputs=risk_output)

    # PDF Q&A
    with gr.Tab("PDF Q&A"):
        pdf_input = gr.File(label="Upload PDF")
        question_input = gr.Textbox(label="Ask a question in English")
        answer_output = gr.Textbox(label="Answer", interactive=False, lines=10)
        ask_btn = gr.Button("Ask Question")
        ask_btn.click(fn=pdf_qa, inputs=[pdf_input, question_input], outputs=answer_output)

# Launch App
demo.launch(share=True)
