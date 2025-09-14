# Super Agent System - Ultimate AI Assistant Orchestrator
import json
import logging
import os
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
import sqlite3
import pickle

from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

# Import your existing agents (assuming they're in the same directory)
from agents import (
    Chatbot, WebScrapingAgent, DatabaseQueryOrchestrator, 
    VectorKnowledgeAgent, DatabaseDiscoveryAgent, SQLQueryAgent, FlatFileQueryAgent
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentType(Enum):
    CHATBOT = "chatbot"
    WEB_SCRAPING = "web_scraping"
    DATABASE_QUERY = "database_query"
    VECTOR_KNOWLEDGE = "vector_knowledge"
    SQL_QUERY = "sql_query"
    FLATFILE_QUERY = "flatfile_query"

class QueryIntent(BaseModel):
    """Structured output for query classification"""
    primary_agent: AgentType = Field(description="The primary agent best suited for this query")
    confidence: float = Field(description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Explanation for why this agent was chosen")
    requires_multiple_agents: bool = Field(description="Whether this query needs multiple agents")
    secondary_agents: List[AgentType] = Field(default=[], description="Additional agents that might be needed")
    complexity_level: str = Field(description="simple, moderate, or complex")

class IntelligentRouterAgent:
    """
    Smart routing agent that uses LLM to classify queries and route to appropriate agents
    """
    
    def __init__(self, llm, agents: Dict[AgentType, Any]):
        self.llm = llm
        self.agents = agents
        self.routing_prompt = self._create_routing_prompt()
        self.classifier = self.routing_prompt | self.llm.with_structured_output(QueryIntent)
        
    def _create_routing_prompt(self) -> ChatPromptTemplate:
        """Create the intelligent routing prompt"""
        return ChatPromptTemplate.from_messages([
            ("system", """
You are an intelligent routing agent that analyzes user queries and determines which specialized AI agent should handle the request.

Available Agents and their capabilities:

🤖 CHATBOT:
- General conversation and casual chat
- Simple Q&A and explanations  
- Greetings, thanks, goodbyes
- Basic help and information
- When user needs friendly conversation
Examples: "Hi there", "How are you?", "Thanks!", "What can you do?", "Tell me a joke"

🌐 WEB_SCRAPING:
- Research information from websites
- Search for current/real-time information
- Gather data from online sources
- News, prices, reviews, articles
- When user needs fresh web data
Examples: "Search for latest news about AI", "Find reviews for iPhone 15", "What's the current price of Tesla stock?"

🗄️ DATABASE_QUERY:
- Query SQL databases
- Access structured data from tables
- Generate reports from databases
- Business intelligence queries
- When user needs data from company databases
Examples: "Show sales data for Q3", "Find customer records", "Generate monthly report", "Query employee database"

📚 VECTOR_KNOWLEDGE:
- Search through knowledge base
- Retrieve relevant documents
- Answer questions from stored knowledge
- Company policies, manuals, documentation
- When user needs information from knowledge base
Examples: "What's our refund policy?", "How do I configure the system?", "Find documentation about feature X"

📄 FLATFILE_QUERY:
- Process CSV, JSON, XML files
- Analyze spreadsheet data
- Extract information from files
- Data transformation and analysis
- When user needs to work with files
Examples: "Analyze this CSV file", "Process the uploaded data", "Extract information from JSON"

Classification Guidelines:
1. Choose the SINGLE best agent for the primary task
2. Set confidence based on how clear the intent is (0.0-1.0)
3. Mark requires_multiple_agents=True only if genuinely needs 2+ agents
4. Classify complexity: simple (basic chat/info), moderate (single agent task), complex (multi-step/multi-agent)
5. When in doubt between agents, prefer the simpler option
6. For ambiguous queries, choose CHATBOT and let it ask for clarification

Remember: Route intelligently based on INTENT and CONTEXT, not just keywords.
            """),
            ("user", """
Analyze this user query and determine the best routing:

User Query: "{query}"

Consider:
- What is the user actually trying to accomplish?
- What type of information or action do they need?
- Which agent has the right capabilities for this task?
- How confident are you in this classification?

Provide your routing decision with clear reasoning.
            """)
        ])
    
    def route_query(self, query: str, conversation_context: Optional[str] = None) -> QueryIntent:
        """
        Intelligently route a query to the appropriate agent(s)
        """
        try:
            # Add conversation context if available
            enhanced_query = query
            if conversation_context:
                enhanced_query = f"Context: {conversation_context}\n\nCurrent Query: {query}"
            
            # Get routing decision from LLM
            routing_result = self.classifier.invoke({"query": enhanced_query})
            
            logger.info(f"Query routed to {routing_result.primary_agent.value} with confidence {routing_result.confidence}")
            logger.info(f"Reasoning: {routing_result.reasoning}")
            
            return routing_result
            
        except Exception as e:
            logger.error(f"Error in query routing: {e}")
            # Fallback to chatbot for safety
            return QueryIntent(
                primary_agent=AgentType.CHATBOT,
                confidence=0.5,
                reasoning=f"Fallback to chatbot due to routing error: {str(e)}",
                requires_multiple_agents=False,
                secondary_agents=[],
                complexity_level="simple"
            )
    
    def execute_routing(self, query: str, conversation_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Route query and execute with the appropriate agent(s)
        """
        # Get routing decision
        intent = self.route_query(query, conversation_context)
        
        try:
            # Execute with primary agent
            primary_agent = self.agents.get(intent.primary_agent)
            if not primary_agent:
                raise ValueError(f"Agent {intent.primary_agent} not available")
            
            # Execute based on agent type
            primary_result = self._execute_with_agent(primary_agent, intent.primary_agent, query)
            
            result = {
                "routing_decision": {
                    "primary_agent": intent.primary_agent.value,
                    "confidence": intent.confidence,
                    "reasoning": intent.reasoning,
                    "complexity": intent.complexity_level
                },
                "primary_response": primary_result,
                "requires_multiple_agents": intent.requires_multiple_agents
            }
            
            # If multiple agents needed, execute secondary ones
            if intent.requires_multiple_agents and intent.secondary_agents:
                result["secondary_responses"] = {}
                for agent_type in intent.secondary_agents:
                    secondary_agent = self.agents.get(agent_type)
                    if secondary_agent:
                        secondary_result = self._execute_with_agent(secondary_agent, agent_type, query)
                        result["secondary_responses"][agent_type.value] = secondary_result
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing routing: {e}")
            # Fallback to chatbot
            chatbot = self.agents.get(AgentType.CHATBOT)
            if chatbot:
                fallback_response = self._execute_with_agent(chatbot, AgentType.CHATBOT, query)
                return {
                    "routing_decision": {
                        "primary_agent": "chatbot",
                        "confidence": 0.5,
                        "reasoning": f"Fallback due to execution error: {str(e)}",
                        "complexity": "simple"
                    },
                    "primary_response": fallback_response,
                    "requires_multiple_agents": False
                }
            else:
                return {
                    "routing_decision": {
                        "primary_agent": "error",
                        "confidence": 0.0,
                        "reasoning": f"System error: {str(e)}",
                        "complexity": "error"
                    },
                    "primary_response": f"I'm sorry, I encountered an error: {str(e)}",
                    "requires_multiple_agents": False
                }
    
    def _execute_with_agent(self, agent: Any, agent_type: AgentType, query: str) -> str:
        """Execute query with specific agent based on its type"""
        try:
            if agent_type == AgentType.CHATBOT:
                return agent.chat(query)
            
            elif agent_type == AgentType.WEB_SCRAPING:
                result = agent.run(query)
                return result if isinstance(result, str) else str(result)
            
            elif agent_type == AgentType.DATABASE_QUERY:
                result = agent.query(query)
                return result if isinstance(result, str) else str(result)
            
            elif agent_type == AgentType.VECTOR_KNOWLEDGE:
                result = agent.query_knowledge(query)
                if hasattr(result, 'retrieved_knowledge'):
                    return result.retrieved_knowledge
                return str(result)
            
            elif agent_type in [AgentType.SQL_QUERY, AgentType.FLATFILE_QUERY]:
                result = agent.query(query)
                return result if isinstance(result, str) else str(result)
            
            else:
                return f"Agent {agent_type} execution not implemented"
                
        except Exception as e:
            logger.error(f"Error executing {agent_type}: {e}")
            return f"Error executing {agent_type.value}: {str(e)}"

class WorkflowStep(BaseModel):
    """Individual step in a workflow"""
    step_id: str
    agent_type: AgentType
    action: str
    parameters: Dict[str, Any]
    requires_human_approval: bool = False
    depends_on: List[str] = []
    
class WorkflowPlan(BaseModel):
    """Complete workflow execution plan"""
    plan_id: str
    user_query: str
    steps: List[WorkflowStep]
    estimated_duration: str
    requires_human_approval: bool
    reasoning: str

@dataclass
class ConversationMemory:
    """Conversation memory structure"""
    user_id: str
    messages: List[Dict[str, Any]]
    context: Dict[str, Any]
    last_workflow: Optional[str]
    created_at: datetime
    updated_at: datetime

class SuperAgentState(TypedDict):
    """State for SuperAgent workflow"""
    user_id: str
    user_query: str
    conversation_memory: Optional[ConversationMemory]
    vector_context: str
    workflow_plan: Optional[WorkflowPlan]
    current_step: int
    step_results: Dict[str, Any]
    requires_human_approval: bool
    human_approval_received: bool
    final_response: str
    error_message: str

# Updated SuperAgent class with intelligent routing
class SuperAgent:
    """
    Ultimate AI Assistant with intelligent query routing
    """
    
    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        model_provider: str = "google_genai", 
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        tavily_api_key: Optional[str] = None,
        database_directory: str = "./Databases",
        knowledge_base_path: str = "./knowledge_base",
        memory_db_path: str = "./conversation_memory.db"
    ):
        self.model = model
        self.model_provider = model_provider
        self.temperature = temperature
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.memory_db_path = memory_db_path
        
        if not self.api_key:
            raise ValueError("API key is required.")
        
        # Initialize LLM
        self.llm = init_chat_model(
            model=self.model,
            model_provider=self.model_provider,
            temperature=self.temperature,
            api_key=self.api_key
        )
        
        # Initialize all specialized agents
        self.agents = self._initialize_agents(tavily_api_key, database_directory, knowledge_base_path)
        
        # Initialize intelligent router
        self.router = IntelligentRouterAgent(self.llm, self.agents)
        
        # Initialize conversation memory database
        self._init_memory_db()
        
        logger.info("SuperAgent initialized with intelligent routing")
    
    def _initialize_agents(self, tavily_api_key, database_directory, knowledge_base_path):
        """Initialize all specialized agents"""
        agents = {}
        
        try:
            # Basic Chatbot
            agents[AgentType.CHATBOT] = Chatbot(
                model=self.model,
                temperature=self.temperature,
                api_key=self.api_key
            )
            
            # Web Scraping Agent
            agents[AgentType.WEB_SCRAPING] = WebScrapingAgent(
                model=self.model,
                temperature=self.temperature,
                api_key=self.api_key,
                tavily_api_key=tavily_api_key
            )
            
            # Database Query Orchestrator
            agents[AgentType.DATABASE_QUERY] = DatabaseQueryOrchestrator(
                model=self.model,
                temperature=self.temperature,
                api_key=self.api_key,
                database_directory=database_directory
            )
            
            # Vector Knowledge Agent
            agents[AgentType.VECTOR_KNOWLEDGE] = VectorKnowledgeAgent(
                model=self.model,
                temperature=self.temperature,
                api_key=self.api_key,
                index_path=knowledge_base_path
            )
            
            logger.info("All agents initialized successfully")
            return agents
            
        except Exception as e:
            logger.error(f"Error initializing agents: {e}")
            raise
    
    def _init_memory_db(self):
        """Initialize SQLite database for conversation memory"""
        try:
            conn = sqlite3.connect(self.memory_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    user_id TEXT PRIMARY KEY,
                    messages TEXT,
                    context TEXT,
                    last_workflow TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Memory database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing memory database: {e}")
    
    def _load_conversation_memory(self, user_id: str) -> Optional[ConversationMemory]:
        """Load conversation memory for a user"""
        try:
            conn = sqlite3.connect(self.memory_db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM conversations WHERE user_id = ?", 
                (user_id,)
            )
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return ConversationMemory(
                    user_id=row[0],
                    messages=json.loads(row[1]),
                    context=json.loads(row[2]),
                    last_workflow=row[3],
                    created_at=datetime.fromisoformat(row[4]),
                    updated_at=datetime.fromisoformat(row[5])
                )
            return None
            
        except Exception as e:
            logger.error(f"Error loading conversation memory: {e}")
            return None
    
    def _save_conversation_memory(self, memory: ConversationMemory):
        """Save conversation memory for a user"""
        try:
            conn = sqlite3.connect(self.memory_db_path)
            cursor = conn.cursor()
            
            memory.updated_at = datetime.now()
            
            cursor.execute('''
                INSERT OR REPLACE INTO conversations 
                (user_id, messages, context, last_workflow, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                memory.user_id,
                json.dumps(memory.messages),
                json.dumps(memory.context),
                memory.last_workflow,
                memory.created_at.isoformat(),
                memory.updated_at.isoformat()
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Memory saved for user {memory.user_id}")
            
        except Exception as e:
            logger.error(f"Error saving conversation memory: {e}")
    
    def _build_workflow(self):
        """Build the SuperAgent LangGraph workflow"""
        workflow = StateGraph(SuperAgentState)
        
        # Add workflow nodes
        workflow.add_node("load_memory", self._load_memory_node)
        workflow.add_node("analyze_query", self._analyze_query_node)
        workflow.add_node("create_plan", self._create_plan_node)
        workflow.add_node("human_approval", self._human_approval_node)
        workflow.add_node("execute_workflow", self._execute_workflow_node)
        workflow.add_node("finalize_response", self._finalize_response_node)
        workflow.add_node("save_memory", self._save_memory_node)
        
        # Define workflow edges
        workflow.set_entry_point("load_memory")
        workflow.add_edge("load_memory", "analyze_query")
        workflow.add_edge("analyze_query", "create_plan")
        
        # Conditional edge for human approval
        workflow.add_conditional_edges(
            "create_plan",
            lambda state: "human_approval" if state.get("requires_human_approval") else "execute_workflow"
        )
        
        workflow.add_edge("human_approval", "execute_workflow")
        workflow.add_edge("execute_workflow", "finalize_response")
        workflow.add_edge("finalize_response", "save_memory")
        workflow.add_edge("save_memory", END)
        
        return workflow.compile()
    
    def _load_memory_node(self, state: SuperAgentState) -> Dict[str, Any]:
        """Load conversation memory for the user"""
        logger.info("Loading conversation memory")
        
        memory = self._load_conversation_memory(state["user_id"])
        
        return {
            "conversation_memory": memory,
            "current_step": 0,
            "step_results": {},
            "requires_human_approval": False,
            "human_approval_received": False
        }
    
    def _analyze_query_node(self, state: SuperAgentState) -> Dict[str, Any]:
        """Analyze user query and retrieve relevant context"""
        logger.info("Analyzing user query")
        
        user_query = state["user_query"]
        
        # Get context from vector knowledge base
        vector_context = ""
        try:
            vector_agent = self.agents[AgentType.VECTOR_KNOWLEDGE]
            context_result = vector_agent.query_knowledge(user_query, k=3)
            vector_context = context_result.retrieved_knowledge
        except Exception as e:
            logger.error(f"Error getting vector context: {e}")
        
        return {
            "vector_context": vector_context
        }
    
    def _create_plan_node(self, state: SuperAgentState) -> Dict[str, Any]:
        """Create dynamic workflow plan based on query analysis"""
        logger.info("Creating workflow plan")
        
        user_query = state["user_query"]
        vector_context = state.get("vector_context", "")
        conversation_memory = state.get("conversation_memory")
        
        # Build context for planning
        context = f"User Query: {user_query}\n"
        if vector_context:
            context += f"Relevant Knowledge: {vector_context}\n"
        if conversation_memory:
            recent_messages = conversation_memory.messages[-3:] if conversation_memory.messages else []
            context += f"Recent Conversation: {json.dumps(recent_messages)}\n"
        
        # Use LLM to create workflow plan
        planning_prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are a workflow planning expert. Based on the user query and available context, create a dynamic workflow plan.

Available Agents:
- CHATBOT: General conversation and simple Q&A
- WEB_SCRAPING: Research information from websites
- DATABASE_QUERY: Query SQL databases or flat files
- VECTOR_KNOWLEDGE: Query vector knowledge base

Create a workflow plan with specific steps. For complex or potentially harmful operations, require human approval.
            """),
            ("user", "Context: {context}\n\nCreate a workflow plan.")
        ])
        
        class PlanGenerator(BaseModel):
            steps: List[Dict[str, Any]] = Field(description="List of workflow steps")
            requires_human_approval: bool = Field(description="Whether human approval is needed")
            reasoning: str = Field(description="Reasoning for the plan")
            estimated_duration: str = Field(description="Estimated completion time")
        
        planner = planning_prompt | self.llm.with_structured_output(PlanGenerator)
        
        try:
            plan_result = planner.invoke({"context": context})
            
            # Convert to WorkflowPlan
            steps = []
            for i, step_data in enumerate(plan_result.steps):
                step = WorkflowStep(
                    step_id=f"step_{i}",
                    agent_type=AgentType(step_data.get("agent_type", "chatbot")),
                    action=step_data.get("action", "process"),
                    parameters=step_data.get("parameters", {}),
                    requires_human_approval=step_data.get("requires_human_approval", False)
                )
                steps.append(step)
            
            workflow_plan = WorkflowPlan(
                plan_id=f"plan_{datetime.now().timestamp()}",
                user_query=user_query,
                steps=steps,
                estimated_duration=plan_result.estimated_duration,
                requires_human_approval=plan_result.requires_human_approval,
                reasoning=plan_result.reasoning
            )
            
            return {
                "workflow_plan": workflow_plan,
                "requires_human_approval": plan_result.requires_human_approval
            }
            
        except Exception as e:
            logger.error(f"Error creating workflow plan: {e}")
            # Fallback to simple chatbot
            fallback_plan = WorkflowPlan(
                plan_id=f"fallback_{datetime.now().timestamp()}",
                user_query=user_query,
                steps=[
                    WorkflowStep(
                        step_id="fallback_step",
                        agent_type=AgentType.CHATBOT,
                        action="chat",
                        parameters={"message": user_query}
                    )
                ],
                estimated_duration="< 1 minute",
                requires_human_approval=False,
                reasoning="Fallback to simple chatbot due to planning error"
            )
            
            return {
                "workflow_plan": fallback_plan,
                "requires_human_approval": False
            }
    
    def _human_approval_node(self, state: SuperAgentState) -> Dict[str, Any]:
        """Handle human-in-the-loop approval"""
        logger.info("Requesting human approval")
        
        # In a real implementation, this would integrate with your Telegram bot
        # to send an approval request and wait for response
        
        workflow_plan = state.get("workflow_plan")
        if workflow_plan:
            approval_message = f"""
🤖 **Workflow Approval Required**

**Query:** {workflow_plan.user_query}
**Plan:** {workflow_plan.reasoning}
**Steps:** {len(workflow_plan.steps)}
**Estimated Duration:** {workflow_plan.estimated_duration}

Reply with 'APPROVE' to proceed or 'DENY' to cancel.
            """
            
            # For now, we'll auto-approve (in production, implement actual approval mechanism)
            logger.info("Auto-approving workflow (implement actual approval in production)")
            
            return {
                "human_approval_received": True
            }
        
        return {
            "human_approval_received": False,
            "error_message": "No workflow plan available for approval"
        }
    
    def _execute_workflow_node(self, state: SuperAgentState) -> Dict[str, Any]:
        """Execute the planned workflow steps"""
        logger.info("Executing workflow steps")
        
        workflow_plan = state.get("workflow_plan")
        if not workflow_plan:
            return {
                "error_message": "No workflow plan available",
                "final_response": "Sorry, I couldn't create a plan to handle your request."
            }
        
        step_results = {}
        
        try:
            for step in workflow_plan.steps:
                logger.info(f"Executing step {step.step_id}: {step.action}")
                
                agent = self.agents.get(step.agent_type)
                if not agent:
                    step_results[step.step_id] = {
                        "error": f"Agent {step.agent_type} not available"
                    }
                    continue
                
                # Execute step based on agent type
                if step.agent_type == AgentType.CHATBOT:
                    result = agent.chat(step.parameters.get("message", ""))
                    step_results[step.step_id] = {"response": result}
                
                elif step.agent_type == AgentType.WEB_SCRAPING:
                    query = step.parameters.get("query", workflow_plan.user_query)
                    result = agent.run(query)
                    step_results[step.step_id] = result
                
                elif step.agent_type == AgentType.DATABASE_QUERY:
                    query = step.parameters.get("query", workflow_plan.user_query)
                    result = agent.query(query)
                    step_results[step.step_id] = result
                
                elif step.agent_type == AgentType.VECTOR_KNOWLEDGE:
                    query = step.parameters.get("query", workflow_plan.user_query)
                    result = agent.query_knowledge(query)
                    step_results[step.step_id] = {
                        "contexts": result.similar_contexts,
                        "knowledge": result.retrieved_knowledge
                    }
                
                logger.info(f"Step {step.step_id} completed successfully")
        
        except Exception as e:
            logger.error(f"Error executing workflow: {e}")
            return {
                "error_message": f"Workflow execution failed: {str(e)}",
                "step_results": step_results
            }
        
        return {
            "step_results": step_results
        }
    
    def _finalize_response_node(self, state: SuperAgentState) -> Dict[str, Any]:
        """Combine workflow results into final response"""
        logger.info("Finalizing response")
        
        user_query = state["user_query"]
        step_results = state.get("step_results", {})
        error_message = state.get("error_message", "")
        
        if error_message:
            return {
                "final_response": f"I encountered an error: {error_message}"
            }
        
        # Use LLM to combine results into coherent response
        response_prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are an AI assistant. Combine the workflow results into a coherent, helpful response to the user's query.
Make the response natural and conversational while including all relevant information from the results.
            """),
            ("user", "User Query: {query}\n\nWorkflow Results: {results}\n\nProvide a comprehensive response.")
        ])
        
        try:
            response_chain = response_prompt | self.llm
            final_response = response_chain.invoke({
                "query": user_query,
                "results": json.dumps(step_results, indent=2)
            })
            
            return {
                "final_response": final_response.content if hasattr(final_response, 'content') else str(final_response)
            }
            
        except Exception as e:
            logger.error(f"Error finalizing response: {e}")
            # Fallback response
            return {
                "final_response": "I processed your request but encountered an issue formatting the response. Here are the raw results: " + json.dumps(step_results, indent=2)
            }
    
    def _save_memory_node(self, state: SuperAgentState) -> Dict[str, Any]:
        """Save conversation to memory"""
        logger.info("Saving conversation memory")
        
        user_id = state["user_id"]
        user_query = state["user_query"]
        final_response = state.get("final_response", "")
        workflow_plan = state.get("workflow_plan")
        
        # Load or create memory
        memory = self._load_conversation_memory(user_id)
        if not memory:
            memory = ConversationMemory(
                user_id=user_id,
                messages=[],
                context={},
                last_workflow=None,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        
        # Add new messages
        memory.messages.append({
            "role": "user",
            "content": user_query,
            "timestamp": datetime.now().isoformat()
        })
        
        memory.messages.append({
            "role": "assistant",
            "content": final_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update context
        if workflow_plan:
            memory.last_workflow = workflow_plan.plan_id
            memory.context["last_plan"] = workflow_plan.dict()
        
        # Keep only last 20 messages to prevent database bloat
        if len(memory.messages) > 20:
            memory.messages = memory.messages[-20:]
        
        # Save memory
        self._save_conversation_memory(memory)
        
        return {}
    
    async def process_message(self, user_id: str, message: str) -> str:
        """
        Main entry point - uses intelligent routing instead of workflow planning
        """
        try:
            logger.info(f"Processing message from user {user_id}: {message}")
            
            # Load conversation context
            memory = self._load_conversation_memory(user_id)
            context = None
            if memory and memory.messages:
                # Get recent conversation for context
                recent_messages = memory.messages[-3:]
                context = " ".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages])
            
            # Route and execute with intelligent router
            result = self.router.execute_routing(message, context)
            
            # Extract the response
            primary_response = result.get("primary_response", "I couldn't process your request.")
            routing_info = result.get("routing_decision", {})
            
            # Log routing decision for debugging
            logger.info(f"Routed to {routing_info.get('primary_agent')} with confidence {routing_info.get('confidence')}")
            
            # Handle multiple agent responses if needed
            if result.get("requires_multiple_agents") and result.get("secondary_responses"):
                # Combine responses intelligently
                combined_response = self._combine_responses(primary_response, result["secondary_responses"])
                final_response = combined_response
            else:
                final_response = primary_response
            
            # Save conversation memory
            self._save_conversation_message(user_id, message, final_response)
            
            logger.info(f"Response generated for user {user_id}")
            return final_response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return f"I encountered an error processing your request: {str(e)}"
    
    def _combine_responses(self, primary_response: str, secondary_responses: Dict[str, str]) -> str:
        """Combine multiple agent responses intelligently"""
        try:
            # Use LLM to combine responses coherently
            combine_prompt = ChatPromptTemplate.from_messages([
                ("system", """
You are an expert at combining multiple AI agent responses into a single coherent answer.
Combine the responses naturally, avoiding redundancy while preserving all important information.
Make it sound like a single, well-structured response.
                """),
                ("user", """
Primary Response: {primary}

Secondary Responses: {secondary}

Combine these into a single, coherent response.
                """)
            ])
            
            combiner = combine_prompt | self.llm
            result = combiner.invoke({
                "primary": primary_response,
                "secondary": str(secondary_responses)
            })
            
            return result.content if hasattr(result, 'content') else str(result)
            
        except Exception as e:
            logger.error(f"Error combining responses: {e}")
            # Fallback to simple concatenation
            combined = primary_response
            for agent_type, response in secondary_responses.items():
                combined += f"\n\nAdditionally from {agent_type}: {response}"
            return combined
    
    def _save_conversation_message(self, user_id: str, user_message: str, bot_response: str):
        """Save conversation message to memory"""
        try:
            # Load or create memory
            memory = self._load_conversation_memory(user_id)
            if not memory:
                memory = ConversationMemory(
                    user_id=user_id,
                    messages=[],
                    context={},
                    last_workflow=None,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
            
            # Add messages
            memory.messages.extend([
                {
                    "role": "user", 
                    "content": user_message,
                    "timestamp": datetime.now().isoformat()
                },
                {
                    "role": "assistant",
                    "content": bot_response, 
                    "timestamp": datetime.now().isoformat()
                }
            ])
            
            # Keep only last 20 messages
            if len(memory.messages) > 20:
                memory.messages = memory.messages[-20:]
            
            # Save memory
            self._save_conversation_memory(memory)
            
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get statistics about routing decisions"""
        # This could be implemented to track routing patterns
        return {
            "message": "Routing statistics not implemented yet",
            "available_agents": list(self.agents.keys())
        }
    
    def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversation history for a user"""
        memory = self._load_conversation_memory(user_id)
        if memory and memory.messages:
            return memory.messages[-limit:]
        return []
    
    def clear_conversation_memory(self, user_id: str) -> bool:
        """Clear conversation memory for a user"""
        try:
            conn = sqlite3.connect(self.memory_db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))
            conn.commit()
            conn.close()
            logger.info(f"Memory cleared for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error clearing memory: {e}")
            return False


# Flask Backend API
from flask import Flask, request, jsonify
import asyncio
from threading import Thread

class SuperAgentAPI:
    """Flask API for SuperAgent"""
    
    def __init__(self, super_agent: SuperAgent, host="127.0.0.1", port=5000):
        self.super_agent = super_agent
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        
        # Setup routes
        self.app.route("/chat", methods=["POST"])(self.chat_endpoint)
        self.app.route("/history/<user_id>", methods=["GET"])(self.history_endpoint)
        self.app.route("/clear/<user_id>", methods=["DELETE"])(self.clear_endpoint)
        self.app.route("/health", methods=["GET"])(self.health_endpoint)
    
    def chat_endpoint(self):
        """Handle chat messages"""
        try:
            data = request.get_json()
            user_id = data.get("userId", "anonymous")
            message = data.get("message", "")
            
            if not message:
                return jsonify({"error": "Message is required"}), 400
            
            # Process message asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            response = loop.run_until_complete(
                self.super_agent.process_message(user_id, message)
            )
            
            loop.close()
            
            return jsonify({"reply": response})
            
        except Exception as e:
            logger.error(f"API error: {e}")
            return jsonify({"error": str(e)}), 500
    
    def history_endpoint(self, user_id):
        """Get conversation history"""
        try:
            limit = request.args.get("limit", 10, type=int)
            history = self.super_agent.get_conversation_history(user_id, limit)
            return jsonify({"history": history})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    def clear_endpoint(self, user_id):
        """Clear conversation history"""
        try:
            success = self.super_agent.clear_conversation_memory(user_id)
            return jsonify({"success": success})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    def health_endpoint(self):
        """Health check endpoint"""
        return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})
    
    def run(self, debug=False):
        """Run the Flask API"""
        logger.info(f"Starting SuperAgent API on {self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=debug)


# Main execution
if __name__ == "__main__":
    # Initialize SuperAgent
    super_agent = SuperAgent(
        api_key=os.getenv("LLM_API_KEY"),
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        database_directory="./Databases",
        knowledge_base_path="./knowledge_base"
    )
    
    # Initialize API
    api = SuperAgentAPI(super_agent)
    
    # Run API server
    api.run(debug=True)