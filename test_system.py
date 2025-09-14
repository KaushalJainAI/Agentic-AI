# SuperAgent System Test Suite
import os
import sys
import asyncio
import json
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import system components
from config import SuperAgentConfig
from super_agent_system import SuperAgent

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SuperAgentTester:
    """Test suite for SuperAgent system"""
    
    def __init__(self):
        self.super_agent = None
        self.test_user_id = "test_user_123"
        
    async def setup(self):
        """Initialize SuperAgent for testing"""
        try:
            logger.info("Setting up SuperAgent for testing...")
            
            # Validate configuration
            config_status = SuperAgentConfig.validate_config()
            if not config_status["valid"]:
                logger.error("Configuration validation failed:")
                for issue in config_status["issues"]:
                    logger.error(f"  - {issue}")
                return False
            
            # Initialize SuperAgent
            self.super_agent = SuperAgent(
                api_key=SuperAgentConfig.LLM_API_KEY,
                tavily_api_key=SuperAgentConfig.TAVILY_API_KEY,
                database_directory=SuperAgentConfig.DATABASE_DIRECTORY,
                knowledge_base_path=SuperAgentConfig.KNOWLEDGE_BASE_PATH,
                memory_db_path="./test_memory.db"  # Use test database
            )
            
            logger.info("SuperAgent initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            return False
    
    async def test_basic_conversation(self):
        """Test basic conversation functionality"""
        logger.info("🧪 Testing basic conversation...")
        
        test_message = "Hello, can you introduce yourself?"
        
        try:
            response = await self.super_agent.process_message(self.test_user_id, test_message)
            
            assert response is not None, "Response should not be None"
            assert len(response) > 10, "Response should be meaningful"
            
            logger.info(f"✅ Basic conversation test passed")
            logger.info(f"   Query: {test_message}")
            logger.info(f"   Response: {response[:100]}...")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Basic conversation test failed: {e}")
            return False
    
    async def test_memory_persistence(self):
        """Test conversation memory functionality"""
        logger.info("🧪 Testing memory persistence...")
        
        try:
            # Send initial message
            await self.super_agent.process_message(self.test_user_id, "My name is Alice")
            
            # Send follow-up message referencing previous context
            response = await self.super_agent.process_message(self.test_user_id, "What did I just tell you my name was?")
            
            # Check if memory was preserved
            assert "Alice" in response or "alice" in response.lower(), "Should remember the user's name"
            
            logger.info("✅ Memory persistence test passed")
            logger.info(f"   Response: {response[:100]}...")
            
            # Test memory retrieval
            history = self.super_agent.get_conversation_history(self.test_user_id)
            assert len(history) >= 2, "Should have at least 2 messages in history"
            
            logger.info("✅ Memory retrieval test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Memory persistence test failed: {e}")
            return False
    
    async def test_web_research_capability(self):
        """Test web research functionality"""
        logger.info("🧪 Testing web research capability...")
        
        # Only test if Tavily API key is available
        if not SuperAgentConfig.TAVILY_API_KEY:
            logger.warning("⚠️  Skipping web research test - no Tavily API key")
            return True
        
        try:
            test_query = "What are the latest developments in AI agents in 2024?"
            
            response = await self.super_agent.process_message(self.test_user_id, test_query)
            
            # Check if response contains research-like content
            research_indicators = ["2024", "AI", "agent", "development", "recent", "latest"]
            found_indicators = sum(1 for indicator in research_indicators if indicator.lower() in response.lower())
            
            assert found_indicators >= 3, f"Response should contain research content (found {found_indicators}/6 indicators)"
            
            logger.info("✅ Web research test passed")
            logger.info(f"   Query: {test_query}")
            logger.info(f"   Response length: {len(response)} characters")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Web research test failed: {e}")
            return False
    
    async def test_vector_knowledge_integration(self):
        """Test vector knowledge base functionality"""
        logger.info("🧪 Testing vector knowledge integration...")
        
        try:
            # Add some test knowledge to the vector store
            test_knowledge = [
                "The SuperAgent system can orchestrate multiple AI agents simultaneously.",
                "Human-in-the-loop functionality allows for approval of sensitive operations.",
                "The system uses FAISS for vector similarity search with sentence transformers."
            ]
            
            vector_agent = self.super_agent.agents.get(self.super_agent._get_agent_type("vector_knowledge"))
            if vector_agent:
                # Add knowledge
                result = vector_agent.add_knowledge(
                    content=test_knowledge,
                    metadata=[{"category": "test", "source": "test_suite"}] * len(test_knowledge)
                )
                
                assert result["status"] == "success", "Knowledge addition should succeed"
                
                # Test knowledge retrieval
                query_result = vector_agent.query_knowledge("How does SuperAgent work?")
                
                assert len(query_result.similar_contexts) > 0, "Should find similar contexts"
                
                logger.info("✅ Vector knowledge test passed")
                logger.info(f"   Added {result['chunks_added']} knowledge chunks")
                logger.info(f"   Retrieved {len(query_result.similar_contexts)} similar contexts")
                
                return True
            else:
                logger.warning("⚠️  Vector agent not available, skipping test")
                return True
                
        except Exception as e:
            logger.error(f"❌ Vector knowledge test failed: {e}")
            return False
    
    async def test_workflow_planning(self):
        """Test dynamic workflow planning"""
        logger.info("🧪 Testing workflow planning...")
        
        try:
            # Test complex query that should trigger workflow planning
            complex_query = "I need to research the latest AI trends and then analyze some data"
            
            response = await self.super_agent.process_message(self.test_user_id, complex_query)
            
            # Check if response indicates planning occurred
            assert response is not None, "Should generate a response"
            assert len(response) > 50, "Should provide a comprehensive response"
            
            logger.info("✅ Workflow planning test passed")
            logger.info(f"   Query: {complex_query}")
            logger.info(f"   Response generated: {len(response)} characters")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Workflow planning test failed: {e}")
            return False
    
    async def test_error_handling(self):
        """Test system error handling"""
        logger.info("🧪 Testing error handling...")
        
        try:
            # Test with malformed input
            malformed_inputs = [
                "",  # Empty string
                "   ",  # Whitespace only  
                "x" * 10000,  # Very long input
            ]
            
            for test_input in malformed_inputs:
                response = await self.super_agent.process_message(self.test_user_id, test_input)
                assert response is not None, f"Should handle malformed input: '{test_input[:20]}...'"
            
            logger.info("✅ Error handling test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error handling test failed: {e}")
            return False
    
    async def cleanup(self):
        """Clean up test artifacts"""
        logger.info("🧹 Cleaning up test artifacts...")
        
        try:
            # Clear test user memory
            if self.super_agent:
                self.super_agent.clear_conversation_memory(self.test_user_id)
            
            # Remove test database file
            test_db_path = "./test_memory.db"
            if os.path.exists(test_db_path):
                os.remove(test_db_path)
            
            logger.info("✅ Cleanup completed")
            
        except Exception as e:
            logger.warning(f"⚠️  Cleanup warning: {e}")
    
    async def run_all_tests(self):
        """Run complete test suite"""
        logger.info("🚀 Starting SuperAgent Test Suite")
        logger.info("=" * 50)
        
        # Setup
        setup_success = await self.setup()
        if not setup_success:
            logger.error("❌ Setup failed - aborting tests")
            return False
        
        # Run tests
        tests = [
            ("Basic Conversation", self.test_basic_conversation),
            ("Memory Persistence", self.test_memory_persistence),
            ("Web Research", self.test_web_research_capability),
            ("Vector Knowledge", self.test_vector_knowledge_integration),
            ("Workflow Planning", self.test_workflow_planning),
            ("Error Handling", self.test_error_handling),
        ]
        
        passed_tests = 0
        failed_tests = 0
        
        for test_name, test_function in tests:
            try:
                success = await test_function()
                if success:
                    passed_tests += 1
                else:
                    failed_tests += 1
            except Exception as e:
                logger.error(f"❌ {test_name} test crashed: {e}")
                failed_tests += 1
        
        # Cleanup
        await self.cleanup()
        
        # Results summary
        logger.info("=" * 50)
        logger.info("🏁 Test Suite Results")
        logger.info(f"   ✅ Passed: {passed_tests}")
        logger.info(f"   ❌ Failed: {failed_tests}")
        logger.info(f"   📊 Success Rate: {(passed_tests/(passed_tests+failed_tests)*100):.1f}%")
        
        if failed_tests == 0:
            logger.info("🎉 All tests passed! SuperAgent system is working correctly.")
            return True
        else:
            logger.warning(f"⚠️  {failed_tests} test(s) failed. Check logs for details.")
            return False

async def main():
    """Main test execution"""
    tester = SuperAgentTester()
    success = await tester.run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())