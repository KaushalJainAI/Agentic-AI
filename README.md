# SuperAgent Ultimate AI Assistant System

A comprehensive AI assistant system that orchestrates multiple specialized agents through a unified Telegram bot interface with human-in-the-loop capabilities, dynamic workflows, conversation memory, and vector knowledge base integration.

## 🌟 System Overview

### What This System Does

The **SuperAgent** acts as an ultimate orchestrator that intelligently routes user queries to the most appropriate specialized agents:

1. **Chatbot Agent** - Basic conversational AI
2. **WebScrapingAgent** - Dynamic web research with structured extraction
3. **DatabaseQueryOrchestrator** - Intelligent database selection and querying
4. **SQLQueryAgent** - SQLite database specialist
5. **FlatFileQueryAgent** - CSV/Excel/JSON processor  
6. **VectorKnowledgeAgent** - RAG-enabled knowledge system with FAISS

### Key Features

- 🤖 **Multi-Agent Orchestration**: Automatically selects and coordinates the right agents for each task
- 🔄 **Dynamic Workflows**: Creates custom execution plans based on query complexity
- 👥 **Human-in-the-Loop**: Requests approval for sensitive or complex operations
- 🧠 **Conversation Memory**: Maintains context across sessions using SQLite storage
- 📚 **Vector Knowledge Base**: RAG integration with FAISS for contextual responses
- 📱 **Telegram Integration**: User-friendly bot interface for all interactions
- 🔍 **Intelligent Routing**: LLM-powered decision making for agent selection

## 🚀 Quick Start

### 1. Environment Setup

Create a `.env` file with your API keys:

```env
# Required API Keys
LLM_API_KEY=your_gemini_api_key_here
TELEGRAM_BOT_API_KEY=your_telegram_bot_token_here

# Optional API Keys  
TAVILY_API_KEY=your_tavily_key_for_web_search
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Directory Structure

Create these directories:
```
SuperAgent/
├── main.py                    # Main launcher
├── super_agent_system.py      # Core SuperAgent system
├── config.py                  # Configuration settings
├── paste.py                   # Your existing agents
├── telegram_bot.py           # Telegram bot interface
├── .env                      # API keys
├── Databases/               # Database files directory
│   └── description.txt      # Database descriptions
├── knowledge_base/          # Vector knowledge storage
└── conversation_memory.db   # Conversation history
```

### 4. Run the System

```bash
python main.py
```

## 🔧 Architecture Details

### SuperAgent Workflow

The SuperAgent uses a LangGraph-based workflow with these nodes:

```
User Message → Load Memory → Analyze Query → Create Plan 
     ↓
Human Approval (if needed) → Execute Workflow → Finalize Response → Save Memory
```

### Agent Selection Logic

The system intelligently routes queries based on:

- **Database queries** → DatabaseQueryOrchestrator
- **Web research needs** → WebScrapingAgent  
- **Simple conversations** → Chatbot
- **Knowledge retrieval** → VectorKnowledgeAgent
- **Multi-step tasks** → Combined agent workflows

### Memory System

- **Conversation History**: Last 20 messages per user
- **Context Preservation**: Workflow results and preferences
- **SQLite Storage**: Persistent memory across sessions
- **Automatic Cleanup**: Configurable retention periods

## 📊 Usage Examples

### Simple Query
```
User: "What's the weather today?"
SuperAgent: Routes to → Chatbot (simple response)
```

### Research Query  
```
User: "Find the latest AI research papers on RAG systems"
SuperAgent: Routes to → WebScrapingAgent → Structured extraction
```

### Database Query
```
User: "Show me sales data from last quarter" 
SuperAgent: Routes to → DatabaseQueryOrchestrator → SQL/FlatFile agents
```

### Complex Multi-Step Query
```
User: "Research company X, analyze their financials, and create a summary"
SuperAgent: Creates workflow → WebScraping + Database + Knowledge synthesis
```

### Human-in-the-Loop Example
```
User: "Delete all records older than 2020"
SuperAgent: 🚨 Requires approval → "This will delete data. Approve?"
User: "APPROVE" → Executes deletion
```

## ⚙️ Configuration

### Agent Settings
```python
AGENT_SETTINGS = {
    "vector_knowledge": {
        "embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "similarity_threshold": 0.3
    },
    "web_scraping": {
        "max_urls": 20,
        "timeout": 10
    }
}
```

### Human Approval Triggers
```python
HUMAN_APPROVAL_REQUIRED_FOR = [
    "database_modifications",
    "file_operations", 
    "external_api_calls",
    "sensitive_data_queries"
]
```

## 🔌 API Endpoints

The system exposes a REST API:

- **POST /chat** - Send messages to SuperAgent
- **GET /history/{user_id}** - Get conversation history
- **DELETE /clear/{user_id}** - Clear user memory
- **GET /health** - System health check

## 🛠️ Extending the System

### Adding New Agents

1. Create your agent class following the existing pattern
2. Add it to `AgentType` enum
3. Update `_initialize_agents()` method
4. Add execution logic in `_execute_workflow_node()`

### Custom Workflow Steps

Modify the `_create_plan_node()` to include your custom workflow logic:

```python
custom_step = WorkflowStep(
    step_id="custom_analysis",
    agent_type=AgentType.YOUR_AGENT,
    action="analyze",
    parameters={"data": query_data}
)
```

## 🔍 Monitoring & Debugging

### Logging
- All operations logged to `superagent.log`
- Real-time console output during execution
- Structured error reporting with context

### Memory Inspection
```python
# Get conversation history
history = super_agent.get_conversation_history(user_id, limit=10)

# Clear memory for testing
super_agent.clear_conversation_memory(user_id)
```

### Workflow Debugging
The system provides detailed step-by-step execution logs for troubleshooting complex workflows.

## 📈 Performance Considerations

- **Memory Management**: Automatic cleanup of old conversations
- **Concurrent Processing**: Thread-safe agent execution
- **Error Recovery**: Graceful degradation on agent failures
- **Resource Limits**: Configurable timeouts and result limits

## 🔒 Security Features

- **API Key Management**: Environment variable based security
- **Human Oversight**: Approval requirements for sensitive operations
- **Input Validation**: Pydantic-based parameter validation
- **Error Sanitization**: Safe error reporting without exposing internals

## 🚨 Troubleshooting

### Common Issues

1. **Agent Initialization Failed**
   - Check API keys in `.env` file
   - Verify model availability

2. **Memory Database Errors**
   - Ensure write permissions for SQLite file
   - Check disk space availability

3. **Telegram Bot Not Responding**
   - Verify bot token is valid
   - Check API server is running on correct port

### Debug Mode
```bash
python main.py --debug
```

## 🤝 Contributing

The system is designed for extensibility. Key areas for contribution:

- **New Agent Types**: Specialized agents for specific domains
- **Workflow Templates**: Pre-built workflows for common tasks
- **UI Improvements**: Enhanced Telegram bot interactions
- **Performance Optimization**: Caching and optimization strategies

## 📝 License

[Add your license information here]

---

🎯 **SuperAgent**: Your ultimate AI assistant orchestrator - combining the power of multiple specialized agents with intelligent routing, human oversight, and persistent memory.