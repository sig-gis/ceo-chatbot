import streamlit as st
from ceo_chatbot.rag.pipeline import RagService

# load CEO Chatbot RAG service
@st.cache_resource
def get_rag_service(config_path: str = "conf/base/rag_config.yml") -> RagService:
    """
    Create and cache a single RagService instance.

    Caching ensures:
      - FAISS index is loaded once
      - Hugging Face model checkpoints are loaded once
    """
    return RagService(config_path=config_path)


# chat UI
st.set_page_config(page_title="Demo RAG Chatbot", page_icon="💬", layout="wide")
st.title("Demo RAG Chatbot")

# Initialize RAG service
with st.spinner("Loading RAG pipeline (docs + model)..."):
    rag = get_rag_service()

# Session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# chat input
if prompt := st.chat_input("Ask a question about the docs..."):
    # 1. Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Get RAG answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer_stream,_ = rag.stream_answer(
                question=prompt,
                num_retrieved_docs=30,
                num_docs_final=5,
            )

            # st.write_stream will iterate the generator and render chunks as they arrive
            answer_text = st.write_stream(answer_stream)

    # store full answer in history
    st.session_state.messages.append({"role": "assistant", "content": answer_text})
