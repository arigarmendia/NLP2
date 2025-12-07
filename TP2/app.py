import logging
import streamlit as st
from rag.retriever import get_rag_chain
from memory import wrap_with_memory, clear_session_history, get_session_message_count

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    st.set_page_config(
        page_title="CV RAG Chatbot",
        page_icon="💼",
        layout="wide",
    )
    
    st.title("💼 CV RAG Chatbot (Groq + Pinecone)")

    st.markdown(
        """
        Ask questions about your resume.

        This chatbot uses:
        - **Groq LLM** for fast generation
        - **Pinecone** for vector search over your resume
        - **LangChain** for RAG orchestration and conversational memory
        """
    )

    # Sidebar settings
    st.sidebar.title("⚙️ Chatbot Settings")
    st.sidebar.markdown("---")

    st.sidebar.subheader("Assistant Persona")
    system_prompt = st.sidebar.text_area(
        "System message:",
        value=(
            "You are an assistant that answers questions strictly based on the user's resume. "
            "Be concise, factual, and refer explicitly to relevant experience, skills, and education. "
            "If the resume does not contain the answer, say you are not sure."
        ),
        height=130,
    )

    st.sidebar.subheader("Groq Model")
    model = st.sidebar.selectbox(
        "Choose a model:",
        [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
        ],
        help="Llama 3.1 8B is faster, Llama 3.3 70B is more capable"
    )

    # This is now informational only - memory is handled automatically
    conversational_memory_length = st.sidebar.slider(
        "Conversational memory display:",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of previous messages to display (informational only). "
             "Full memory is handled automatically by LangChain.",
    )

    # Initialize session state
    if "session_id" not in st.session_state:
        st.session_state.session_id = "default"

    if "history_store" not in st.session_state:
        st.session_state.history_store = {}

    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []
        st.sidebar.success("✓ New conversation started")
    else:
        msg_count = get_session_message_count(
            st.session_state.history_store,
            st.session_state.session_id
        )
        st.sidebar.info(f"💬 {len(st.session_state.chat_log)} messages in conversation")

    # Clear conversation button
    if st.sidebar.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.chat_log = []
        clear_session_history(
            st.session_state.history_store,
            st.session_state.session_id
        )
        st.sidebar.success("✓ Conversation cleared")
        st.rerun()

    st.sidebar.markdown("---")
    
    # Initialize RAG chain
    try:
        base_rag_chain = get_rag_chain(system_prompt=system_prompt, model_name=model)
        st.sidebar.success("✓ RAG pipeline ready")
    except Exception as e:
        st.sidebar.error(f"❌ Error initializing RAG pipeline")
        st.error(f"Failed to initialize RAG pipeline: {e}")
        logger.error(f"RAG initialization error: {e}")
        st.info("Please check your API keys and Pinecone index configuration.")
        st.stop()

    # Wrap with memory
    session_id = st.session_state.session_id
    history_store = st.session_state.history_store
    
    try:
        chain_with_memory = wrap_with_memory(
            base_rag_chain,
            history_store=history_store,
            session_id=session_id,
            input_key="question",
        )
    except Exception as e:
        st.error(f"Failed to initialize conversational memory: {e}")
        logger.error(f"Memory initialization error: {e}")
        st.stop()

    # Main chat interface
    st.markdown("### 💬 Ask a question about your resume:")
    
    user_question = st.text_input(
        "Type your message:",
        placeholder="For example: What is my experience with machine learning?",
        label_visibility="collapsed",
        key="user_input"
    )

    # Process user question
    if user_question and user_question.strip():
        with st.spinner("🤔 The chatbot is thinking..."):
            try:
                # Invoke chain with memory
                result = chain_with_memory.invoke(
                    {"question": user_question},
                    config={"configurable": {"session_id": session_id}},
                )
                
                # Extract response
                response = getattr(result, "content", str(result))

                # Store in chat log
                message = {"human": user_question, "ai": response}
                st.session_state.chat_log.append(message)

                # Display answer
                st.markdown("### ✨ Answer:")
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
                st.error(f"❌ Error processing your question: {e}")
                logger.error(f"Query processing error: {e}")
                st.info(
                    "Please check:\n"
                    "- Your internet connection\n"
                    "- API keys are valid\n"
                    "- Pinecone index contains data"
                )

    # Display conversation history
    if st.session_state.chat_log:
        st.markdown("---")
        st.markdown("### 📝 Conversation History")
        
        # Show last N messages based on slider
        display_count = min(conversational_memory_length * 2, len(st.session_state.chat_log))
        recent_messages = st.session_state.chat_log[-display_count:]
        
        for i, msg in enumerate(recent_messages):
            col1, col2 = st.columns([1, 20])
            with col1:
                st.markdown("👤")
            with col2:
                st.markdown(f"**You:** {msg['human']}")
            
            col1, col2 = st.columns([1, 20])
            with col1:
                st.markdown("🤖")
            with col2:
                st.markdown(f"**Assistant:** {msg['ai']}")
            
            if i < len(recent_messages) - 1:
                st.markdown("---")

    # Technical info expander
    with st.expander("ℹ️ Technical Information"):
        st.markdown(
            """
            **Architecture:**
            - **Vector Database:** Pinecone serverless index with multilingual-e5-large embeddings
            - **LLM:** Groq (fast, low-latency generation)
            - **Framework:** LangChain for RAG orchestration
            - **Memory:** In-memory conversation history with automatic context management
            
            **Process:**
            1. Your question is embedded using Pinecone's embedding model
            2. Top-k most relevant resume chunks are retrieved
            3. Chunks + conversation history are sent to Groq LLM
            4. LLM generates a contextually-aware answer
            
            **Data Privacy:**
            - Resume data stays in Pinecone
            - Conversation history is stored locally in browser session
            - No data is persisted after session ends
            """
        )


if __name__ == "__main__":
    main()
