
import json
import logging
from typing import Dict, List, Any, Optional, Union, Literal, Annotated
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import operator

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your existing agents (they're in the same directory)
from agents import (
    Chatbot, WebSearchingAgent, DatabaseQueryOrchestrator, 
    SQLQueryAgent , UnifiedMemoryAgent
)

# from tools import (
#     SystemTimeTool, MathTool, PythonExecutorTool, WebSearchTool, JavaScriptExecutorInput, 
#     WebScraperTool, FileHandlerTool, DataVisualizerTool
# )



# Enhanced State Management for LangGraph
class AgentWorkflowState(MessagesState):
    """
    Enhanced state schema that extends MessagesState with workflow tracking.

    Key improvements over original:
    - Uses LangGraph's MessagesState for proper message handling
    - Separates workflow metadata from agent execution state
    - Provides clean state isolation between requests
    """
    # Original query for reference
    original_query: str = ""

    # Selected agent information  
    selected_agent: str = ""
    agent_confidence: float = 0.0

    # Workflow tracking
    workflow_status: str = "pending"  # pending, running, completed, failed
    current_step: str = "initialization"
    completed_steps: Annotated[List[str], operator.add] = []

    # Results and context
    agent_results: Dict[str, Any] = {}
    execution_context: Dict[str, Any] = {}

    # Error handling
    error_messages: Annotated[List[str], operator.add] = []


class QueryIntent(BaseModel):
    """Structured output for agent selection - same as original but with validation"""
    primary_agent: str = Field(description="The primary agent best suited for this query")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Explanation for why this agent was chosen")
    requires_multiple_agents: bool = Field(description="Whether this query needs multiple agents")
    secondary_agents: List[str] = Field(default_factory=list)
    complexity_level: Literal["simple", "moderate", "complex"] = Field(description="Complexity assessment")


class AgentMethodCall(BaseModel):
    """Structured output for method calls with validation"""
    method: str = Field(description="Name of the method to call")
    args: List[str] = Field(default_factory=list, description="Arguments to pass")
    kwargs: Dict[str, str] = Field(default_factory=dict, description="Keyword arguments")


# class ToolNode:
#     """A node that runs the tools requested in the last AIMessage"""

#     def __init__(self, tools:list):
#         self.tools_by_name = {tool.name: tool for tool in tools}

#     def __call__(self, inputs: dict):
#         if messages := inputs.get("messages", []):
#             message = messages[-1]
#         else:
#             raise ValueError("No message found in input")
#         outputs = []
#         for tool_call in message.tool_calls:
#             tool_result = self.tools_by_name[tool_call["name"]].invoke(
#                 tool_call["args"]
#             )
#             outputs.append(
#                 ToolMessage(
#                     content=json.dumps(tool_result),
#                     name=tool_call["name"],
#                     tool_call_id=tool_call["id"],
#                 )
#             )
#         return {"messages": outputs}


class SuperAgentOrchestrator:
    """
    LangGraph-based agent orchestrator that replaces the original SuperVisor.

    Key architectural improvements:
    - Uses LangGraph's StateGraph for proper workflow management
    - Eliminates recursive execution bugs through graph-based flow
    - Provides built-in error handling and recovery
    - Implements proper state isolation between requests
    """

    def __init__(self, agents: Dict[str, Any], use_local=True, api_key=None, model_provider=None, model_name=None):
        self.agents = agents
        self.model_provider = model_provider
        self.model = model_name
        self.temperature = 0.7
        self.api_key = api_key


        # Initialize LLM with structured output capability
        if use_local:
            self.llm = ChatOllama(
                model="qwen3:4b", 
                temperature=0.1,
                format="json"
            )
        else:
            # Initialize the LLM
            self.llm = init_chat_model(
                model=self.model,
                model_provider=self.model_provider,
                temperature=self.temperature,
                api_key=self.api_key
            )


        # Load agent metadata
        with open('agent_docs.json', 'r', encoding='utf-8') as f:
            self.agent_data = json.load(f)

        # Build agent summary for selection
        self.agent_summary = self._build_agent_summary()

        # Build and compile the workflow graph
        self.workflow = self._build_workflow()

        # Configure memory for state persistence
        self.memory = MemorySaver()
        self.compiled_workflow = self.workflow.compile(
            # checkpointer=self.memory
        )

    def clear_memory(self):
        self.memory.clear()
        return True

    def _build_agent_summary(self) -> str:
        """Build formatted agent information for selection"""
        agent_info = []
        for agent_name in self.agents:
            if agent_name in self.agent_data:
                agent = self.agent_data[agent_name]
                agent_info.append(
                    f"Agent: {agent_name}\n"
                    f"Description: {agent['description']}\n"
                    f"Use Cases: {', '.join(agent['use_cases'])}\n"
                    f"Methods: {', '.join(agent['methods'].keys())}"
                )
        return '\n---\n'.join(agent_info)

    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow.

        This replaces the recursive execution logic with a proper state graph:
        1. agent_selection: Analyze query and select appropriate agent
        2. method_planning: Determine which method to call on selected agent  
        3. agent_execution: Execute the agent method
        4. result_evaluation: Check if task is complete or needs more steps
        """

        workflow = StateGraph(AgentWorkflowState)

        # Add nodes for each workflow step
        workflow.add_node("agent_selection", self._select_agent_node)
        workflow.add_node("method_planning", self._plan_method_node)  
        workflow.add_node("agent_execution", self._execute_agent_node)
        workflow.add_node("result_evaluation", self._evaluate_results_node)
        workflow.add_node("error_handler", self._handle_error_node)
        workflow.add_node("finish", self.get_result)  # Placeholder

        workflow.add_edge(START, "agent_selection")
        workflow.add_edge("agent_selection", "method_planning") 
        workflow.add_edge("method_planning", "agent_execution")
        workflow.add_edge("agent_execution", "result_evaluation")
        # result_evaluation uses Command for routing - no conditional edge needed
        workflow.add_edge("finish", END)
        workflow.add_edge("error_handler", END)

        return workflow

    def _select_agent_node(self, state: AgentWorkflowState) -> AgentWorkflowState:
        """
        Node 1: Agent Selection

        Improvement over original:
        - No recursive calls - returns updated state
        - Proper error handling with try/catch
        - Updates workflow status properly
        """
        try:
            # Extract query from messages or use original_query
            if state["messages"]:
                query = state["messages"][-1].content
            else:
                query = state.get("original_query", "")

            prompt = f"""You are a supervisor agent that selects the best agent for user queries. If the user wants you to use a particular agent then the user is right and go ahead with that agent

Available Agents:
{self.agent_summary}

For the query: "{query}"

You are a Large language model and you should know your capabilities what you can do and what you can't do. You are not updated with latest data always, so be wise.
Here's how models typically handle queries about events that have occurred later than their knowledge cutoff:

1. **Acknowledgment of Cutoff**: The model will acknowledge its knowledge cutoff and inform the user that it does not have information about events occurring after that date. This is a transparent way to manage user expectations and avoid speculation.

2. **Avoidance of Speculation**: Models are designed to avoid speculating about events or information they do not have. Instead, they might suggest looking for more recent sources or updates from other platforms.

3. **Redirecting to Other Sources**: Sometimes, models might suggest searching online or consulting other sources for information about recent events, as they cannot provide accurate or up-to-date details.

4. **Focus on Available Information**: The model will focus on providing information available up to its cutoff date, ensuring that any responses are based on the data it was trained on.
Select the most appropriate agent and provide your analysis in JSON format."""

            # Get structured intent from LLM
            intent = self.llm.with_structured_output(QueryIntent).invoke(prompt)

            if not intent:
                raise ValueError("Failed to get agent selection intent")

            # Update state with selection results
            return {
                "original_query": query,
                "selected_agent": intent.primary_agent,
                "agent_confidence": intent.confidence,
                "workflow_status": "running",
                "current_step": "agent_selected",
                "completed_steps": ["agent_selection"],
                "execution_context": {
                    "intent": intent.model_dump(),
                    "selection_reasoning": intent.reasoning
                },
                "messages": state["messages"] + [
                    AIMessage(content=f"Selected {intent.primary_agent} with {intent.confidence:.2f} confidence: {intent.reasoning}")
                ]
            }

        except Exception as e:
            logger.error(f"Error in agent selection: {str(e)}")
            return {
                "workflow_status": "failed",
                "current_step": "agent_selection_failed", 
                "error_messages": [f"Agent selection failed: {str(e)}"],
                "messages": state["messages"] + [
                    AIMessage(content=f"Error selecting agent: {str(e)}")
                ]
            }

    def _plan_method_node(self, state: AgentWorkflowState) -> AgentWorkflowState:
        """
        Node 2: Method Planning

        Determines which method to call on the selected agent.
        """
        try:
            selected_agent = state["selected_agent"]
            if not selected_agent or selected_agent not in self.agent_data:
                raise ValueError(f"Invalid selected agent: {selected_agent}")

            available_methods = self.agent_data[selected_agent]['methods']

            prompt = f"""
You are planning method execution for an agent.

Query: {state['original_query']}
Selected Agent: {selected_agent}
Available Methods: {available_methods}

Current workflow status: {state.get('current_step', 'unknown')}
Completed steps: {state.get('completed_steps', [])}

Determine the best method to call with appropriate arguments.
Return JSON format with method, args, and kwargs.
"""

            method_call = self.llm.with_structured_output(AgentMethodCall).invoke(prompt)

            if not method_call or not method_call.method:
                raise ValueError("Failed to get valid method call plan")

            return {
                "current_step": "method_planned",
                "completed_steps": state["completed_steps"] + ["method_planning"],
                "execution_context": {
                    **state.get("execution_context", {}),
                    "planned_method": method_call.model_dump()
                },
                "messages": state["messages"] + [
                    AIMessage(content=f"Planning to call {method_call.method} on {selected_agent}")
                ]
            }

        except Exception as e:
            logger.error(f"Error in method planning: {str(e)}")
            return {
                "workflow_status": "failed",
                "current_step": "method_planning_failed",
                "error_messages": state.get("error_messages", []) + [f"Method planning failed: {str(e)}"]
            }

    def _execute_agent_node(self, state: AgentWorkflowState) -> AgentWorkflowState:
        """
        Node 3: Agent Execution

        Safely executes the planned method on the selected agent.
        """
        try:
            selected_agent = state["selected_agent"]
            execution_context = state.get("execution_context", {})
            planned_method = execution_context.get("planned_method", {})

            if not planned_method:
                raise ValueError("No method planned for execution")

            # Get agent instance
            agent_instance = self.agents.get(selected_agent)
            if not agent_instance:
                raise ValueError(f"Agent instance not found: {selected_agent}")

            # Execute method safely
            method_name = planned_method["method"]
            args = planned_method.get("args", [])
            kwargs = planned_method.get("kwargs", {})

            result = self._safe_execute_method(agent_instance, method_name, *args, **kwargs)

            return {
                "current_step": "agent_executed",
                "completed_steps": state["completed_steps"] + ["agent_execution"],
                "agent_results": {
                    **state.get("agent_results", {}),
                    method_name: result
                },
                "messages": state["messages"] + [
                    AIMessage(content=f"Successfully executed {method_name} on {selected_agent}")
                ]
            }

        except Exception as e:
            logger.error(f"Error in agent execution: {str(e)}")
            return {
                "workflow_status": "failed",
                "current_step": "agent_execution_failed",
                "error_messages": state.get("error_messages", []) + [f"Agent execution failed: {str(e)}"]
            }

    def _evaluate_results_node(self, state: AgentWorkflowState):
        """
        Node 4: Result Evaluation

        Uses Command to combine state updates with routing decisions.
        This replaces the problematic recursive logic from the original.
        """
        try:
            # Check for errors first
            if state.get("error_messages"):
                return Command(
                    update={
                        "workflow_status": "failed",
                        "current_step": "evaluation_found_errors"
                    },
                    goto="error_handler"
                )

            # Check completion criteria
            completed_steps = state.get("completed_steps", [])
            max_steps = 5  # Prevent infinite loops

            # Simple completion logic - can be enhanced
            if len(completed_steps) >= max_steps:
                return Command(
                    update={
                        "workflow_status": "completed",
                        "current_step": "max_steps_reached",
                        "completed_steps": completed_steps + ["evaluation"]
                    },
                    goto="finish"
                )

            # Check if we have meaningful results
            agent_results = state.get("agent_results", {})
            if agent_results:
                return Command(
                    update={
                        "workflow_status": "completed", 
                        "current_step": "task_completed",
                        "completed_steps": completed_steps + ["evaluation"]
                    },
                    goto="finish"
                )

            # Need more work
            return Command(
                update={
                    "current_step": "needs_more_work",
                    "completed_steps": completed_steps + ["evaluation"]
                },
                goto=Command.PARENT
            )

        except Exception as e:
            logger.error(f"Error in result evaluation: {str(e)}")
            return Command(
                update={
                    "workflow_status": "failed",
                    "current_step": "evaluation_error",
                    "error_messages": state.get("error_messages", []) + [f"Evaluation failed: {str(e)}"]
                },
                goto="error_handler"
            )

    def _handle_error_node(self, state: AgentWorkflowState) -> AgentWorkflowState:
        """
        Node 5: Error Handler

        Centralized error handling and recovery.
        """
        error_messages = state.get("error_messages", [])
        logger.error(f"Workflow failed with errors: {error_messages}")

        return {
            "workflow_status": "failed",
            "current_step": "error_handled",
            "messages": state["messages"] + [
                AIMessage(content=f"Workflow failed with errors: {'; '.join(error_messages)}")
            ]
        }

    def _should_continue_or_finish(self, state: AgentWorkflowState) -> Literal["continue", "finish", "error"]:
        """
        Conditional edge function for result evaluation routing.

        This replaces the problematic recursive if-condition from original code.
        """
        # This function is called by the conditional edge to determine routing
        # The actual logic is in _evaluate_results_node which returns Command
        # This is just a placeholder - LangGraph will use the Command return
        if state.get("error_messages"):
            return "error"
        elif state.get("workflow_status") == "completed":
            return "finish"
        else:
            return "continue"

    def _safe_execute_method(self, agent_instance: Any, method_name: str, *args, **kwargs) -> Any:
        """
        Safely execute a method on an agent instance.

        Improvement over original:
        - Better error handling
        - Method validation
        - Timeout protection could be added here
        """
        try:
            if not hasattr(agent_instance, method_name):
                raise AttributeError(f"Agent {type(agent_instance).__name__} has no method '{method_name}'")

            method = getattr(agent_instance, method_name)
            if not callable(method):
                raise AttributeError(f"{method_name} is not callable")

            # Execute method
            result = method(*args, **kwargs)
            logger.info(f"Successfully executed {method_name}")
            return result

        except Exception as e:
            logger.error(f"Error executing {method_name}: {str(e)}")
            raise

    def execute_workflow(self, query: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute the workflow for a given query.

        Key improvements over original execute method:
        - No recursive calls
        - Proper state isolation via thread_id
        - Uses LangGraph's built-in execution engine
        - Returns structured results
        """
        try:
            # Create initial state
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "original_query": query,
                "workflow_status": "pending"
            }

            # Configure execution
            config = {
                "configurable": {
                    "thread_id": thread_id or f"workflow_{datetime.now().timestamp()}"
                },
                "recursion_limit": 10  # Prevent infinite loops
            }

            # Execute workflow
            final_state = self.compiled_workflow.invoke(initial_state, config=config)

            # Return structured results
            return {
                "success": final_state.get("workflow_status") == "completed",
                "status": final_state.get("workflow_status", "unknown"),
                "current_step": final_state.get("current_step", "unknown"),
                "completed_steps": final_state.get("completed_steps", []),
                "agent_used": final_state.get("selected_agent", "unknown"),
                "agent_confidence": final_state.get("agent_confidence", 0.0),
                "results": final_state.get("agent_results", {}),
                "errors": final_state.get("error_messages", []),
                "messages": [msg.content for msg in final_state.get("messages", [])],
                "thread_id": config["configurable"]["thread_id"]
            }

        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            return {
                "success": False,
                "status": "failed",
                "errors": [str(e)],
                "results": {},
                "completed_steps": final_state.get("completed_steps", []),
                "messages": []
            }
        
    def get_result(self, state: AgentWorkflowState) -> None:
        print("-"*40)
        # print("workflow_status:", state['workflow_status'])
        # print("completed_steps->", state['completed_steps'])
        print("agent_results", state['agent_results'])
        print("workflow summary", state['messages'])
        return state

# Usage Example
if __name__ == "__main__":
    # Initialize agents (same as original)
    agents = {
        "Chatbot": Chatbot(), 
        # "SQLQueryAgent": SQLQueryAgent(), 
        "DatabaseOrchestrator": DatabaseQueryOrchestrator(), 
        # "FlatFileQueryAgent": FlatFileQueryAgent(),
        "WebScrapingAgent": WebSearchingAgent(),
        # "VectorKnowledgeAgent": VectorKnowledgeAgent()
    }

    # Create orchestrator
    orchestrator = SuperAgentOrchestrator(agents)

    # Test queries
    test_queries = [
        "Extract sales data from the customer database",
        "Have a conversation about product features", 
        "Scrape product prices from ecommerce sites",
        "Analyze CSV files with customer data"
    ]

    for query in test_queries:
        print(f"\nExecuting Query: {query}")
        result = orchestrator.execute_workflow(query)

        print(f"Success: {result['success']}")
        print(f"Status: {result['status']}")
        print(f"Completed Steps: {result['completed_steps']}")
        if result['errors']:
            print(f"Errors: {result['errors']}")
        print("-" * 50)
