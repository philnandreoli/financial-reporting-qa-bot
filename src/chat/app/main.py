import streamlit as st
import requests 
from decouple import config

QNA_API_ENDPOINT = config("QNA_API_ENDPOINT")

with st.sidebar:
    "[Coming Soon](https://github.com/philnandreoli/financial-reporting-qa-bot)"

st.title("Financial Reporting - Chat with Search")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant", "content": "Hi, I'm a chatbot who can answer questions about Publicly Traded Companies Financial Reporting."
        }
    ]

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="How many outstanding shares of stock did Microsoft have in their most recent report?"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    inputs = {"input": prompt}
    response = requests.post(QNA_API_ENDPOINT, json=inputs).json()

    with st.chat_message("assistant"):    
        
        st.session_state.messages.append({"role": "assistant", "content": response["output"]})
        st.write(response["output"])
