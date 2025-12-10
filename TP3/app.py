import logging
import streamlit as st
from rag.retriever import retrieve_with_filter
from rag.multi_agent_orchestrator import MultiAgentOrchestrator
from config import PERSONS
from langchain_core.messages import HumanMessage, AIMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# def main():
#     st.set_page_config(
#         page_title="Multi-Agent CV RAG Chatbot",
#         page_icon="img/app_logo.png",
#         layout="wide",
#     )
    
#     col1, col2 = st.columns([2, 3])

#     with col1:
#         st.title("Multi-Agent Resume RAG")
#         st.write("Ask questions about multiple candidates' resumes.")
#         st.write("""
#         This chatbot uses:
#         - **Groq LLM** for fast generation  
#         - **Pinecone** for vector search with metadata filtering
#         - **Multi-Agent System** with routing & synthesis
#         """)

#     with col2:
#         st.image("img/multi-agent-system.png", width=550)


def convert_chat_log_to_langchain_messages(chat_log):
    """
    Convert Streamlit chat log to LangChain message format.
    
    Args:
        chat_log: List of dicts with 'human' and 'ai' keys
    
    Returns:
        List of LangChain message objects
    """
    messages = []
    for entry in chat_log:
        messages.append(HumanMessage(content=entry["human"]))
        messages.append(AIMessage(content=entry["ai"]))
    return messages



def main():
    st.set_page_config(
        page_title="Multi-Agent CV RAG Chatbot",
        page_icon="img/app_logo.png",
        layout="wide",
    )

    st.title("Multi-Agent resume RAG")

    # Image right under the title
    st.image("img/multi-agent-system.png", width=600)

    # Description text and bullets below the image
    st.write("Ask questions about multiple candidates' resumes.")
    st.write("""
    This chatbot uses:
    - **Groq LLM** for fast generation  
    - **Pinecone** for vector search with metadata filtering  
    - **Multi-Agent System** with routing & synthesis  
    """)


    # Sidebar settings
    st.sidebar.title("⚙️ Chatbot Settings")
    st.sidebar.markdown("---")

    st.sidebar.subheader("Assistant Persona")
    system_prompt = st.sidebar.text_area(
        "System message:",
        value=(
            "You are an assistant that answers questions about candidates' resumes. "
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

    # Initialize session state
    if "session_id" not in st.session_state:
        st.session_state.session_id = "default"

    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []
        st.sidebar.success("✓ New conversation started")
    else:
        st.sidebar.info(f"💬 {len(st.session_state.chat_log)} messages in conversation")

    # Clear conversation button
    if st.sidebar.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.chat_log = []
        st.sidebar.success("✓ Conversation cleared")
        st.rerun()

    st.sidebar.markdown("---")
    
    # Initialize Multi-Agent Orchestrator
    try:
        orchestrator = MultiAgentOrchestrator(
            retriever_func=retrieve_with_filter,
            model_name=model
        )
        st.sidebar.success("✓ Multi-agent system ready")
        
        # Show available persons
        st.sidebar.markdown("**Available Candidates:**")
        for person_id, info in PERSONS.items():
            st.sidebar.markdown(f"• {info['name']}")
        
    except Exception as e:
        st.sidebar.error("✗ Error initializing agents")
        st.error(f"Failed to initialize: {e}")
        logger.error(f"Agent initialization error: {e}")
        st.info("Please check your API keys and Pinecone index configuration.")
        st.stop()

    # Main chat interface
    st.markdown("### 💬 Ask about the candidates:")
    
    user_question = st.text_input(
        "Type your message:",
        placeholder="e.g., What is John's Python experience? or Compare Jane and Alice's skills",
        label_visibility="collapsed",
        key="user_input"
    )

        # Process user question
    if user_question and user_question.strip():
        with st.spinner("🤔 The multi-agent system is working..."):
            try:
                # Convert chat log to LangChain message format
                chat_history = convert_chat_log_to_langchain_messages(st.session_state.chat_log)
                
                # Use orchestrator with chat history
                response = orchestrator.query(
                    user_question,
                    chat_history=chat_history,
                    verbose=True
                )

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
                st.error(f"✗ Error processing your question: {e}")
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
        st.markdown("### 📜 Conversation History")
        
        # Show last 5 messages
        display_count = min(5, len(st.session_state.chat_log))
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
            - **Vector Database:** Pinecone with person_id metadata filtering
            - **LLM:** Groq (fast, low-latency generation)
            - **Framework:** LangChain for RAG orchestration
            - **Multi-Agent System:** Router agent + specialized person agents
            - **Memory:** Conversation history for context resolution
            
            **Process:**
            1. Router agent determines which candidate(s) the query is about (with memory context)
            2. Specialized agents retrieve relevant info using metadata filters
            3. For multi-person queries, responses are synthesized
            4. LLM generates contextually-aware answer
            
            **Query Types:**
            - **Single:** "What is John's Python experience?"
            - **Comparison:** "Compare John and Jane's skills"
            - **Cross-person:** "Who has ML experience?"
            - **Follow-up:** "What about her education?" (uses memory to resolve "her")
            
            **Data Privacy:**
            - Resume data stays in Pinecone
            - Conversation history is stored locally in browser session
            - No data is persisted after session ends
            """
        )


if __name__ == "__main__":
    main()