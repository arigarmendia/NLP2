import streamlit as st

from rag.retriever import get_rag_chain
from memory import wrap_with_memory


def main():
    st.title("CV RAG Chatbot (Groq + Pinecone)")

    st.markdown(
        """
        Ask questions about your resume.

        This chatbot uses:
        - Groq LLM for generation
        - Pinecone for vector search over your resume
        - LangChain for memory and RAG orchestration
        """
    )

    st.sidebar.title("Chatbot Settings")
    st.sidebar.markdown("---")

    st.sidebar.subheader("Assistant persona")
    system_prompt = st.sidebar.text_area(
        "System message:",
        value=(
            "You are an assistant that answers questions strictly based on the user's resume. "
            "Be concise, factual, and refer explicitly to relevant experience, skills, and education. "
            "If the resume does not contain the answer, say you are not sure."
        ),
        height=130,
    )

    st.sidebar.subheader("Groq model")
    model = st.sidebar.selectbox(
        "Choose a model:",
        [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
        ],
    )

    conversational_memory_length = st.sidebar.slider(
        "Conversational memory length (informational):",
        min_value=1,
        max_value=10,
        value=5,
        help="Approximate number of previous exchanges remembered. "
             "Memory is handled automatically by LangChain.",
    )

    if "session_id" not in st.session_state:
        st.session_state.session_id = "default"

    if "history_store" not in st.session_state:
        st.session_state.history_store = {}

    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []
        st.sidebar.success("New conversation started")
    else:
        st.sidebar.info(f"Conversation with {len(st.session_state.chat_log)} messages")

    if st.sidebar.button("Clear conversation"):
        st.session_state.chat_log = []
        sid = st.session_state.session_id
        if sid in st.session_state.history_store:
            del st.session_state.history_store[sid]
        st.sidebar.success("Conversation cleared")
        st.rerun()

    st.markdown("### Ask a question about your resume:")
    user_question = st.text_input(
        "Type your message:",
        placeholder="For example: What is my experience with machine learning?",
        label_visibility="collapsed",
    )

    try:
        base_rag_chain = get_rag_chain(system_prompt=system_prompt, model_name=model)
        st.sidebar.success("RAG pipeline ready")
    except Exception as e:
        st.sidebar.error(f"Error initializing RAG pipeline: {e}")
        st.stop()

    session_id = st.session_state.session_id
    history_store = st.session_state.history_store
    chain_with_memory = wrap_with_memory(
        base_rag_chain,
        history_store=history_store,
        session_id=session_id,
        input_key="question",
    )

    if user_question and user_question.strip():
        with st.spinner("The chatbot is thinking..."):
            try:
                result = chain_with_memory.invoke(
                    {"question": user_question},
                    config={"configurable": {"session_id": session_id}},
                )
                response = getattr(result, "content", result)

                message = {"human": user_question, "ai": response}
                st.session_state.chat_log.append(message)

                st.markdown("### Answer:")
                st.markdown(
                    f"""
                    <div style="background-color: #f0f8ff; padding: 15px; border-radius: 10px; border-left: 4px solid #1f77b4;">
                        {response}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption(f"Model: {model}")

            except Exception as e:
                st.error(f"Error processing the question: {e}")
                st.info("Check your internet connection and API configuration.")

    with st.expander("Technical info"):
        st.markdown(
            """
            - Uses Pinecone serverless index 'cv-rag-index' with integrated embeddings.
            - Uses a LangChain RAG chain to combine retrieved chunks with your question.
            - Groq models provide fast, low-latency generation.
            """
        )


if __name__ == "__main__":
    main()
