import streamlit as st
from langchain_ollama import OllamaLLM
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os
import time

model = OllamaLLM(model="Cryptobot")  
BINANCE_API_KEY = "JPZ1SziQpLB3Jm91N6c3wpcozKdub1P8KtsSnMutgpbfqqG3Tx6xfVUnw4rCeBqe"  


def fetch_website_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        st.error(f"Error fetching {url}: {e}")
        return ""

def parse_content(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
  
    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()

    text = ' '.join(soup.stripped_strings)
    return text

def build_vector_store(documents):
    
    embeddings = HuggingFaceEmbeddings() 
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

def create_documents(texts):
    documents = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for text in texts:
        chunks = splitter.split_text(text)
        for chunk in chunks:
            documents.append(Document(page_content=chunk))
    return documents

def get_crypto_data():
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error fetching crypto data: {e}")
        return None

def filter_crypto_data(raw_data):
    return {item["symbol"]: {"price": float(item["lastPrice"]), "change": float(item["priceChangePercent"])} for item in raw_data}

def get_response(user_input):
    try:
        result = model.invoke(input=user_input)
        return result
    except Exception as e:
        return f"An error occurred: {e}"

def render_crypto_data():
    raw_data = get_crypto_data()
    if raw_data:
        crypto_data = filter_crypto_data(raw_data)
        st.sidebar.markdown("### Live Prices ⛏️💰")
        if "visible_crypto" not in st.session_state:
            st.session_state.visible_crypto = 10

        symbols = list(crypto_data.keys())[:st.session_state.visible_crypto]
        for symbol in symbols:
            data = crypto_data[symbol]
            color = "green" if data['change'] >= 0 else "red"
            arrow = "\u25B2" if data['change'] >= 0 else "\u25BC"
            st.sidebar.markdown(
                f"{symbol}: ${data['price']:.2f} "
                f"<span style='color:{color};'>{arrow} {data['change']:.2f}%</span>",
                unsafe_allow_html=True
            )

        if st.sidebar.button("Show More"):
            st.session_state.visible_crypto += 10

        return crypto_data
    return {}

def main():
    st.title("🚀🌕Crypto Expert AI Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        # URLs to fetch content from
        urls = [
            "https://www.galaxy.com/insights/research/crypto-predictions-2025/",
            "https://www.security.org/digital-security/cryptocurrency-annual-consumer-report/",
            "https://yellow.com/news/top-4-emerging-crypto-ownership-trends-of-2025",
            "https://www.talupa.com/news/cryptocurrency/understanding-the-2025-cryptocurrency-adoption-and-consumer-sentiment-report-security-org/",
            "https://www.security.org/digital-security/cryptocurrency-annual-consumer-report/2024/",
            "https://www.security.org/digital-security/crypto/",
            "https://www.security.org/digital-security/nft-market-analysis/",
            "https://www.security.org/vpn/best/crypto/",
            "https://www.oscprofessionals.com/e-commerce/top-trends-in-cryptocurrency-adoption-for-e-commerce-in-2025/"
        ]
        texts = []
        for url in urls:
            html_content = fetch_website_content(url)
            if html_content:
                text = parse_content(html_content)
                texts.append(text)
        documents = create_documents(texts)
        st.session_state.vector_store = build_vector_store(documents)

    crypto_data = render_crypto_data()

    user_input = st.chat_input("Ask me anything about crypto:")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

      
        docs = st.session_state.vector_store.similarity_search(user_input, k=3)
        context = "\n".join([doc.page_content for doc in docs])

        full_input = f" below are the data! existing data, based on the data reply the user! the question is given at the end\n\nContext:\n{context}\n\nLive Prices: {crypto_data}\n\n Please reply as simple as possible.your reply should not exceed more than 20 words.   reply only as Crypto expert.  User Question: {user_input} "

        response_text = get_response(full_input)
        st.session_state.messages.append({"role": "assistant", "content": response_text})

    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(
                    f"{message['content']}",
                    unsafe_allow_html=True
                )
        else:
            with st.chat_message("assistant"):
                st.markdown(
                    f"{message['content']}",
                    unsafe_allow_html=True
                )

if __name__ == "__main__":
    main()


