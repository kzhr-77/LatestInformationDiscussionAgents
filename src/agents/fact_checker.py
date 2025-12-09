from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class FactCheckerAgent:
    def __init__(self, model: ChatOpenAI):
        self.model = model

    def validate(self, claims: list) -> str:
        # TODO: Implement fact checking (Phase 2)
        return "Fact check results..."

