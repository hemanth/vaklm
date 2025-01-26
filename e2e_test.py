import unittest
from vaklm import vaklm

class TestDeepseekReasoning(unittest.TestCase):
    def test_reasoning_stream(self):
        """Test streaming response with reasoning from Deepseek"""
        print("\nTesting Deepseek reasoning stream...")
        
        try:
            for content, reasoning in vaklm(
                endpoint="https://api.deepseek.com/v1/chat/completions",
                model_name="deepseek-reasoner",
                user_prompt="Explain quantum computing in simple terms",
                system_prompt="You are a helpful AI assistant",
                stream=True,
                temperature=0.7,
                max_tokens=1,
                api_key="key"
            ):
                if content:
                    print(f"Content: {content}", end='', flush=True)
                if reasoning:
                    print(f"\n[Reasoning: {reasoning}]")
                    
            self.assertTrue(True)  # Basic assertion to mark test as passed
            
        except Exception as e:
            self.fail(f"Test failed with exception: {str(e)}")

    def test_reasoning_non_stream(self):
        """Test non-streaming response with reasoning from Deepseek"""
        print("\nTesting Deepseek reasoning non-stream...")
        
        try:
            content, reasoning_content = vaklm(
                endpoint="https://api.deepseek.com/v1/chat/completions",
                model_name="deepseek-reasoner",
                user_prompt="Explain general relativity in simple terms",
                system_prompt="You are a helpful AI assistant",
                stream=False,
                temperature=0.7,
                max_tokens=1,
                api_key='key'
            )
            
            self.assertIsInstance(content, str)
            self.assertIsInstance(reasoning_content, str)
            self.assertGreater(len(content), 0)
            self.assertGreater(len(reasoning_content), 0)
            print(f"Content: {content}")
            print(f"Reasoning: {reasoning_content}")
            
        except Exception as e:
            self.fail(f"Test failed with exception: {str(e)}")

if __name__ == "__main__":
    unittest.main()
