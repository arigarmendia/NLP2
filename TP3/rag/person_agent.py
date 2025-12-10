import logging
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from llm_client import get_chat_model
from config import PERSONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PersonAgent:
    """Agent specialized in answering questions about a specific person's resume."""
    
    def __init__(self, person_id: str, retriever_func, model_name: str = "llama-3.1-8b-instant"):
        self.person_id = person_id
        self.person_info = PERSONS[person_id]
        self.retriever_func = retriever_func
        self.llm = get_chat_model(model_name=model_name, max_tokens=300)
        self.agent = self._create_agent()
    
    def _search_resume(self, query: str) -> str:
        """Search this person's resume."""
        try:
            # Use retriever with person_id filter
            results = self.retriever_func(query, person_id=self.person_id)
            
            if not results:
                return f"No relevant information found in {self.person_info['name']}'s resume."
            
            context = "\n\n".join([doc.page_content for doc in results])
            return f"Information from {self.person_info['name']}'s resume:\n\n{context}"
        except Exception as e:
            logger.error(f"Error searching {self.person_id}: {e}")
            return f"Error retrieving information: {e}"
    
    def _create_agent(self) -> AgentExecutor:
        """Create LangChain agent."""
        search_tool = Tool(
            name=f"search_{self.person_id}_resume",
            func=self._search_resume,
            description=(
                f"Search {self.person_info['name']}'s resume for experience, "
                "skills, education, projects, or any other details."
            )
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are an AI assistant answering questions about {self.person_info['name']}'s resume.

Your job:
1. Use the search tool to find relevant information
2. Provide BRIEF, CONCISE answers (2-3 sentences max)
3. Focus only on what's asked - no extra details
4. If information is not in the resume, clearly state that
5. Never make up or infer information
5. Use conversation history to understand context (e.g., if user says "her" or "their", refer to {self.person_info['name']})

Be professional and helpful."""),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=[search_tool],
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=[search_tool],
            verbose=True,
            handle_parsing_errors=True
        )
    
    def query(self, question: str, chat_history=None) -> str:
        """
        Query this person's resume with optional chat history.
        
        Args:
            question: User's question
            chat_history: List of previous messages for context
        """
        input_dict = {"input": question}
        
        if chat_history:
            input_dict["chat_history"] = chat_history
        
        response = self.agent.invoke(input_dict)
        return response["output"]