with open("app.py", "w") as f:
    f.write('''
import streamlit as st
import random
import requests
import os
import time
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from pyngrok import ngrok

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")

# Initialize Gemini Pro LLM
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-pro-002",
    google_api_key=GOOGLE_API_KEY,
)

# Tools for the agent
@tool
def validate_card(card_number: str) -> str:
    """
    Validate card number using BIN lookup API.
    Returns scheme, type, brand, and issuing bank.
    """
    bin_number = card_number[:8] if len(card_number) >= 8 else card_number[:6]
    url = f"https://lookup.binlist.net/{bin_number}"
    try:
        time.sleep(5)
        response = requests.get(url)
        if response.status_code == 429:
            return "Rate limit hit. Please wait and try again."
        if response.status_code != 200:
            return f"Failed to retrieve info for BIN {bin_number}. Status code: {response.status_code}"
        data = response.json()
        return (
            f"Scheme: {data.get('scheme', 'N/A')}, "
            f"Type: {data.get('type', 'N/A')}, "
            f"Brand: {data.get('brand', 'N/A')}, "
            f"Bank: {data.get('bank', {}).get('name', 'N/A')}"
        )
    except Exception as e:
        return f"Error: {e}"

@tool
def simulate_transaction(input_str: str) -> str:
    """
    Simulate a transaction given an amount and memo (e.g., '250 groceries').
    """
    try:
        parts = input_str.split(" ", 1)
        amount = float(parts[0])
        memo = parts[1] if len(parts) > 1 else "No memo"
        response = requests.post("https://jsonplaceholder.typicode.com/posts",
                                 json={"amount": amount, "memo": memo})
        if response.status_code in (200, 201):
            data = response.json()
            return f"Transaction simulated! ID: {data.get('id')} | Amount: {amount} | Memo: {memo}"
        return "Failed to simulate transaction."
    except Exception as e:
        return f"Error: {e}"

@tool
def get_fraud_risk(transaction_id: str) -> str:
    """
    Simulate fraud risk score for a transaction ID.
    """
    score = round(random.uniform(0, 1), 2)
    risk = "High" if score > 0.7 else "Medium" if score > 0.4 else "Low"
    return f"Transaction {transaction_id} | Risk: {risk} | Score: {score}"

# Agent Initialization
agent = initialize_agent(
    tools=[validate_card, simulate_transaction, get_fraud_risk],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Streamlit UI
st.set_page_config(page_title="Sentinel Secure AI", page_icon="🔐")
st.title("🔐 AI-Powered Sentinel Secure")

# Ngrok Public URL (optional)
if NGROK_AUTH_TOKEN:
    os.system("ngrok config add-authtoken " + NGROK_AUTH_TOKEN)
    try:
        public_url = ngrok.connect(8501)
        st.info(f"🔗 Public URL: {public_url}")
    except:
        st.warning("Ngrok failed. Only local access is available.")

# App Tabs
with st.tabs(["Card Validation", "Transaction Simulation", "Fraud Risk", "Smart Agent"]) as (tab1, tab2, tab3, tab4):
    with tab1:
        st.header("Card Validator")
        card = st.text_input("Card Number")
        if st.button("Validate"):
            st.success(agent.run(f"validate_card {card}"))

    with tab2:
        st.header("Transaction Simulation")
        amt = st.number_input("Amount", min_value=0.01)
        memo = st.text_input("Memo")
        if st.button("Simulate"):
            st.success(agent.run(f"simulate_transaction {amt} {memo}"))

    with tab3:
        st.header("Fraud Risk Detector")
        txn_id = st.text_input("Transaction ID")
        if st.button("Check Risk"):
            st.success(agent.run(f"get_fraud_risk {txn_id}"))

    with tab4:
        st.header("Smart Agent Q&A")
        query = st.text_input("Ask Anything")
        if st.button("Ask"):
            st.success(agent.run(query))
''')

