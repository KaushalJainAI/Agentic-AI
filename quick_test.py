"""Quick test script to verify agent imports and basic functionality"""
import sys

print("Testing Agentic AI System...")
print("=" * 50)

# Test 1: Import all agents
print("\n[1] Testing agent imports...")
try:
    from agents import Chatbot, WebSearchingAgent, DatabaseQueryOrchestrator, UnifiedMemoryAgent
    print("[PASS] All agent classes imported successfully")
except Exception as e:
    print(f"[FAIL] Import failed: {e}")
    sys.exit(1)

# Test 2: Import orchestrator
print("\n[2] Testing orchestrator import...")
try:
    from langgraph_super_agent import SuperAgentOrchestrator
    print("[PASS] SuperAgentOrchestrator imported successfully")
except Exception as e:
    print(f"[FAIL] Orchestrator import failed: {e}")
    sys.exit(1)

# Test 3: Import host
print("\n[3] Testing host import...")
try:
    from host import SuperAgentFlaskApp
    print("[PASS] SuperAgentFlaskApp imported successfully")
except Exception as e:
    print(f"[FAIL] Host import failed: {e}")
    sys.exit(1)

# Test 4: Check config
print("\n[4] Testing config...")
try:
    from config import SuperAgentConfig
    print(f"[PASS] Config loaded - API Host: {SuperAgentConfig.API_HOST}:{SuperAgentConfig.API_PORT}")
except Exception as e:
    print(f"[FAIL] Config failed: {e}")
    sys.exit(1)

print("\n" + "=" * 50)
print("All basic tests passed!")
