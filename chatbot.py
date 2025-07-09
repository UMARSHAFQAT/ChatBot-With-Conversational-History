import streamlit as st
from dotenv import load_dotenv
import os
import time

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage, AIMessage

# Load .env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# LangSmith (optional)
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "GROQ Chatbot with Memory"

# Streamlit config
st.set_page_config(page_title="GROQ Chatbot", page_icon="🤖", layout="wide")

# Custom styling
st.markdown("""
    <style>
        .reportview-container { background-color: #1e1e1e; color: #ffffff; }
        .sidebar .sidebar-content { background-color: #2c2c2c; }
        h1, h2, h3, p, label { color: #ffffff !important; }
        .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center; color: #00ffcc;'>🧠 UMAR'S Chatbot with Conversational History Powered By GROQ API </h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Ask me anything — I'll remember our conversation!</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Model Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    max_tokens = st.slider("Max Tokens", 50, 1024, 300)
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.experimental_rerun()

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Create prompt with memory placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Respond conversationally and remember context."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# Build the chain
llm = ChatGroq(
    model="llama3-8b-8192",
    api_key=api_key,
    temperature=temperature,
    max_tokens=max_tokens
)

chain: Runnable = prompt | llm | StrOutputParser()

# User input
user_input = st.chat_input("Type your message...")

if user_input:
    # Append user message
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Invoke chain with full chat history
    with st.spinner("Thinking..."):
        response = chain.invoke({
            "question": user_input,
            "chat_history": st.session_state.chat_history
        })

    # Append AI response
    st.session_state.chat_history.append(AIMessage(content=response))

# Display chat history
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)
