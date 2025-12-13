import logging
from llm_client import get_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from config import PERSONS
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RouterAgent:
    """Routes queries to appropriate person agents."""
    
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        # Limit max_tokens for routing (only needs JSON response)
        self.llm = get_chat_model(model_name=model_name, temperature=0.0, max_tokens=150)
        self.persons = PERSONS
    
    def route_query(self, query: str, chat_history=None) -> dict:
        """
        Determine which person(s) the query is about.
        
        Args:
            query: User's current question
            chat_history: Previous conversation messages for context
        
        Returns:
            {
                "type": "single" | "multiple" | "comparison" | "all",
                "person_ids": List[str],
                "intent": str
            }
        """
        person_list = "\n".join([
            f"- {pid}: {info['name']}"
            for pid, info in self.persons.items()
        ])
        
        system_message = """You are a routing assistant analyzing user queries about resumes.

Available persons:
""" + person_list + """

Analyze the query and determine:
1. Which person(s) they're asking about
2. The query type:
   - "single": One specific person
   - "multiple": Multiple specific persons
   - "comparison": Comparing persons
   - "all": Search across all persons (e.g., "who has Python?")

IMPORTANT: Use conversation history to resolve pronouns like "he", "she", "her", "his", "their", "them".

Return ONLY valid JSON (no extra text):
{{
    "type": "single|multiple|comparison|all",
    "person_ids": ["person_id1", ...],
    "intent": "brief description"
}}"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{query}")
        ])
        
        # Build input with chat history if available
        input_dict = {"query": query}
        if chat_history:
            input_dict["chat_history"] = chat_history
        
        chain = prompt | self.llm
        response = chain.invoke(input_dict)
        
        try:
            # Extract JSON from response
            content = response.content.strip()
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0]
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content)
            
            # Validate person_ids
            valid_ids = [
                pid for pid in result.get("person_ids", [])
                if pid in self.persons
            ]
            
            # Defaults to specific person if no names mentioned
            if not valid_ids and not chat_history:
                # No person identified and no conversation context
                from config import DEFAULT_PERSON  
                valid_ids = [DEFAULT_PERSON]
                result["type"] = "single"
            elif not valid_ids:
                valid_ids = list(self.persons.keys())
            # if not valid_ids:
            #     valid_ids = list(self.persons.keys())
            
            return {
                "type": result.get("type", "all"),
                "person_ids": valid_ids,
                "intent": result.get("intent", "General query")
            }
        except Exception as e:
            logger.error(f"Routing error: {e}, defaulting to all persons")
            return {
                "type": "all",
                "person_ids": list(self.persons.keys()),
                "intent": "General query (parsing failed)"
            }