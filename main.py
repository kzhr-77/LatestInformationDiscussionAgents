import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.orchestrator import OrchestrationAgent

def main():
    topic = input("Enter a topic or URL to analyze: ")
    if not topic:
        print("Topic is required.")
        return

    print("Initializing system...")
    try:
        orchestrator = OrchestrationAgent()
        print("Running analysis...")
        result = orchestrator.invoke({"topic": topic, "messages": []})
        
        print("\n=== Final Report ===")
        print(result.get("final_report"))
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

