import logging
from typing import List, Dict
from rag.person_agent import PersonAgent
from rag.router_agent import RouterAgent
from llm_client import get_chat_model
from config import PERSONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiAgentOrchestrator:
    """Orchestrates multiple person agents for complex queries."""
    
    def __init__(self, retriever_func, model_name: str = "llama-3.1-8b-instant"):
        self.retriever_func = retriever_func
        self.model_name = model_name
        self.router = RouterAgent()
        
        # Create person agents
        self.person_agents = {}
        for person_id in PERSONS.keys():
            self.person_agents[person_id] = PersonAgent(
                person_id=person_id,
                retriever_func=retriever_func,
                model_name=model_name
            )
        
        # LLM for synthesis
        self.synthesis_llm = get_chat_model(model_name="llama-3.3-70b-versatile", max_tokens=400)
    
    def query(self, question: str, chat_history=None, verbose: bool = True) -> str:
        """
        Process query through multi-agent system with conversation memory.
        
        Args:
            question: User's question
            chat_history: List of previous messages for context
            verbose: Whether to log routing info
        """
        
        # Route query with chat history for context
        routing_info = self.router.route_query(question, chat_history=chat_history)
        
        if verbose:
            logger.info(f"Router: {routing_info['type']} | Persons: {routing_info['person_ids']}")
        
        query_type = routing_info["type"]
        person_ids = routing_info["person_ids"]
        
        if query_type == "single" and len(person_ids) == 1:
            return self._handle_single(question, person_ids[0], chat_history)
        else:
            return self._handle_multi(question, person_ids, query_type, chat_history)
    
    def _handle_single(self, question: str, person_id: str, chat_history=None) -> str:
        """Handle single person query with memory."""
        agent = self.person_agents[person_id]
        return agent.query(question, chat_history=chat_history)
    
    def _handle_multi(self, question: str, person_ids: List[str], query_type: str, chat_history=None) -> str:
        """Handle multi-person query with memory."""
        responses = {}
        
        for person_id in person_ids:
            agent = self.person_agents[person_id]
            person_name = PERSONS[person_id]["name"]
            logger.info(f"Querying {person_name}...")
            responses[person_name] = agent.query(question, chat_history=chat_history)
        
        return self._synthesize(question, responses, query_type, chat_history)
    
    def _synthesize(self, question: str, responses: Dict[str, str], query_type: str, chat_history=None) -> str:
        """Synthesize multiple agent responses with conversation context."""
        context_parts = [
            f"=== {name} ===\n{response}\n"
            for name, response in responses.items()
        ]
        context = "\n".join(context_parts)
        
        if query_type == "comparison":
            instruction = "Compare the candidates briefly. Highlight key differences in 2-3 sentences."
        else:
            instruction = "Summarize which candidates match. Be concise (2-3 sentences)."
        
        # Build synthesis prompt with chat history awareness
        synthesis_prompt = f"""Question: {question}

Candidate Info:
{context}

Task: {instruction}

Keep response BRIEF and focused."""
        
        response = self.synthesis_llm.invoke(synthesis_prompt)
        return response.content
