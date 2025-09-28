# Super Agent System - Ultimate AI Assistant Orchestrator
import json
import logging
import os
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import time
from enum import Enum
from dataclasses import dataclass, asdict
import sqlite3
import pickle

from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama
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

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
import json
import ollama

class QueryIntent(BaseModel):
    """Structured output for query classification"""
    primary_agent: str = Field(description="The primary agent best suited for this query")
    confidence: float = Field(description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Explanation for why this agent was chosen")
    requires_multiple_agents: bool = Field(description="Whether this query needs multiple agents")
    secondary_agents: List[str] = Field(default_factory=list, description="Additional agents that might be needed")
    complexity_level: str = Field(description="simple, moderate, or complex")

class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"

class WorkflowState(BaseModel):
    """Simple workflow state tracking"""
    
    # Your existing QueryIntent
    query_intent: QueryIntent
    
    # Current state
    current_status: WorkflowStatus = WorkflowStatus.PENDING
    current_agent: Optional[str] = None          
    current_step: str = "Not started"
    
    # Progress tracking
    completed_steps: List[str] = Field(default_factory=list)
    progress_percent: float = 0.0
    
    # History summary (single string)
    history_summary: str = "Workflow initiated"
    
    # Results and errors
    results: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    
    def start(self, agent_name: str, step_description: str):
        """Start the workflow"""
        self.current_status = WorkflowStatus.RUNNING
        self.current_agent = agent_name
        self.current_step = step_description
        self.history_summary = f"Started with {agent_name}: {step_description}"
    
    def update_step(self, step_description: str, agent_name: str = None):
        """Update current step"""
        if agent_name:
            self.current_agent = agent_name
        self.current_step = step_description
        self.history_summary += f" -> {step_description}"
    
    def complete_step(self, step_name: str):
        """Mark a step as completed"""
        self.completed_steps.append(step_name)
        self.progress_percent = len(self.completed_steps) * 20  # Assume max 5 steps
    
    def complete(self, final_result: Any = None):
        """Complete the workflow"""
        self.current_status = WorkflowStatus.COMPLETED
        self.current_step = "Completed"
        self.progress_percent = 100.0
        if final_result:
            self.results["final"] = final_result
        self.history_summary += " -> Completed successfully"
    
    def fail(self, error_message: str):
        """Mark workflow as failed"""
        self.current_status = WorkflowStatus.FAILED
        self.current_step = "Failed"
        self.errors.append(error_message)
        self.history_summary += f" -> Failed: {error_message}"

    def __str__(self):
        res = f"Current Status: {self.current_status}\n"
        res += f"Current Agent: {self.current_agent}\n"     
        res += f"Current Step: {self.current_step}\n"
        res += f"Progress: {self.progress_percent}%\n"
        res += f"Completed Steps: {self.completed_steps}\n"
        res += f"History Summary: {self.history_summary}\n"
        res += f"Errors: {self.errors}\n"
        return res
    

class MethodCall(BaseModel):
    method: str = Field(description="Name of the method to call")
    args: List[str] = Field(default_factory=list, description="Arguments to pass to the method")
    kwargs: Dict[str, str] = Field(default_factory=dict, description="Keyword arguments to pass to the method")

class WorkFlowAction(BaseModel):
    """Model for workflow planning decisions"""
    action: str = Field(description="Action to take start, update_step, complete_step, fail")
    agent_name: Optional[str] = None
    step_description: Optional[str] = None
    step_to_complete: Optional[str] = None
    final_result: Optional[Any] = None
    error_message: Optional[str] = None

class SuperVisor:
    def __init__(self, agents: Dict[str, Any]):
        # Initialize with structured output capability
        self.llm = ChatOllama(
            model="qwen3:4b",
            temperature=0.1,
            format="json"  # Enable JSON mode
        )
        self.agents = agents

         # Load agent metadata
        with open('agent_docs.json', 'r', encoding='utf-8') as f:
            self.agent_data = json.load(f)

        # Build agent information
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

        self.agent_summary = '\n---\n'.join(agent_info)

        self.workflow_history = []
        self.workflow_state = WorkflowState()

        
        
    def agent_selections(self, query: str) -> Optional[QueryIntent]:
        prompt = f"""You are a supervisor agent that selects the best agent for user queries.

                    Available Agents:
                    {self.agent_summary}

                    For the query: "{query}"

                    Select the most appropriate agent and provide your analysis in the exact JSON format specified.
                """
        try:
            # Direct invocation returns structured output
            intent = self.llm.with_structured_output(QueryIntent).invoke(prompt)
            return intent
            
        except Exception as e:
            print(f"Error with structured output: {e}")
            return None
        
    
    def plan_and_update_workflow(self, cur_workflow_state: WorkflowState)->Optional[WorkflowState]:
            prompt = f"""You are a supervisor agent analyzing the current workflow state to determine the next step.

                    CURRENT WORKFLOW STATE:
                    - Status: {cur_workflow_state.current_status.value}
                    - Current Agent: {cur_workflow_state.current_agent}
                    - Current Step: {cur_workflow_state.current_step}
                    - Progress: {cur_workflow_state.progress_percent}%
                    - Completed Steps: {cur_workflow_state.completed_steps}
                    - History: {cur_workflow_state.history_summary}
                    - Errors: {cur_workflow_state.errors}
                    - Original Query: {cur_workflow_state.query_intent.reasoning}

                    Determine what action should be taken next. Respond with JSON containing:
                    - "action": one of ["start", "update_step", "complete_step", "complete", "fail"]
                    - "agent_name": which agent should handle next (if applicable)
                    - "step_description": description of the next step
                    - "step_to_complete": name of step to mark complete (if action is complete_step)
                    - "final_result": result data (if action is complete)
                    - "error_message": error description (if action is fail)"""

            try:
                action_decision = self.llm.with_structured_output(WorkFlowAction).invoke(prompt)
                
                # Execute the appropriate workflow method
                if action_decision.action == "start":
                    cur_workflow_state.start(action_decision.agent_name, action_decision.step_description)
                elif action_decision.action == "update_step":
                    cur_workflow_state.update_step(action_decision.step_description, action_decision.agent_name)
                elif action_decision.action == "complete_step":
                    cur_workflow_state.complete_step(action_decision.step_to_complete)
                elif action_decision.action == "complete":
                    cur_workflow_state.complete(action_decision.final_result)
                elif action_decision.action == "fail":
                    cur_workflow_state.fail(action_decision.error_message)
                
                return cur_workflow_state
            except Exception as e:
                print(f"Error in workflow planning: {e}")
                cur_workflow_state.fail(f"Workflow planning failed: {str(e)}")
                return cur_workflow_state
        
    def use_agent(self, agent_instance: Any, method_name: str, *args, **kwargs):
        """Safely execute a method on an agent instance"""
        
        try:
            # Check if method exists and is callable
            if hasattr(agent_instance, method_name):
                method = getattr(agent_instance, method_name)
                if callable(method):
                    return method(*args, **kwargs)
                else:
                    raise AttributeError(f"{method_name} is not callable")
            else:
                raise AttributeError(f"Agent has no method '{method_name}'")
        except Exception as e:
            print(f"Error calling {method_name}: {str(e)}")
            return None


    def execute(self, query: str):
        intent = self.agent_selections(query)
        if not intent:
            print("Failed to get intent")
            return False   
        
        print(f"Selected Agent: {intent.primary_agent}")
        methods_available = self.agent_data[intent.primary_agent]['methods']

        # initialize agent
        agent = self.agents[intent.primary_agent]

        prompt = f"""
            You are a superisor agent and you have to smartly call appropriate methods  
            
            Query : {query}
            
            workflow_details shows about the progress and histroy in the task: 
            {self.workflow_state}
            
            Available methods in the class of the function
            {methods_available}

            kwargs for the agent initialization are given in the methods 

            return your answer in exact JSON format.          

            """
        
        try:
            method_req = self.llm.with_structured_output(MethodCall).invoke(prompt)
            self.use_agent(agent, method_req.method, *method_req.args, **method_req.kwargs)
            self.workflow_state = self.plan_and_update_workflow(self.workflow_state)
            print(self.workflow_state)
        
        except Exception as e:
            print(f"Error calling {method_req.method}: {str(e)}")
            return False
        
        if self.workflow_state.current_status != WorkflowStatus.PENDING or len(self.workflow_state.completed_steps) > 5:
            self.execute(query)        
        
        return True
        

# Example usage
if __name__ == "__main__":
    agents = {
        "Chatbot": Chatbot(), 
        "SQLQueryAgent": SQLQueryAgent(), 
        "DatabaseOrchestrator": DatabaseQueryOrchestrator(), 
        "FlatFileQueryAgent": FlatFileQueryAgent(),
        "WebScrapingAgent": WebScrapingAgent(),
        "VectorKnowledgeAgent": VectorKnowledgeAgent()
    }
    
    supervisor = SuperVisor(agents)
    
    # Test queries
    test_queries = [
        "Extract sales data from the customer database",
        "Have a conversation about product features",
        "Scrape product prices from ecommerce sites",
        "Analyze CSV files with customer data"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        time.sleep(1)
        supervisor.execute(query)
        # if intent:
        #     print(f"Selected Agent: {intent.primary_agent}")
        #     print(f"Confidence: {intent.confidence}")
        #     print(f"Reasoning: {intent.reasoning}")
        # else:
        #     print("Failed to get intent")

