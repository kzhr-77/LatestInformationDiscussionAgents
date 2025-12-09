from langgraph.graph import StateGraph, END
from src.core.state import DiscussionState
from src.agents.researcher import ResearcherAgent
from src.agents.analyst_optimistic import OptimisticAnalystAgent
from src.agents.analyst_pessimistic import PessimisticAnalystAgent
from src.agents.fact_checker import FactCheckerAgent
from src.agents.reporter import ReporterAgent
from src.utils.llm import get_llm

def create_graph():
    llm = get_llm()
    
    # Initialize agents
    researcher = ResearcherAgent(llm)
    optimist = OptimisticAnalystAgent(llm)
    pessimist = PessimisticAnalystAgent(llm)
    checker = FactCheckerAgent(llm)
    reporter = ReporterAgent(llm)

    # Define nodes
    def research_node(state: DiscussionState):
        article = researcher.run(state["topic"])
        return {"article_text": article}

    def optimist_node(state: DiscussionState):
        arg = optimist.analyze(state["article_text"])
        # Mocking structured return for now
        return {"optimistic_argument": {"conclusion": arg, "evidence": []}}

    def pessimist_node(state: DiscussionState):
        arg = pessimist.analyze(state["article_text"])
        # Mocking structured return for now
        return {"pessimistic_argument": {"conclusion": arg, "evidence": []}}
    
    def checker_node(state: DiscussionState):
        critique = checker.validate([state["optimistic_argument"], state["pessimistic_argument"]])
        # Mocking return
        return {"critique": {"bias_points": [], "factual_errors": []}}

    def reporter_node(state: DiscussionState):
        report = reporter.create_report(state["messages"])
        # Mocking return
        return {"final_report": {"final_conclusion": report}}

    workflow = StateGraph(DiscussionState)

    # Add nodes
    workflow.add_node("researcher", research_node)
    workflow.add_node("optimist", optimist_node)
    workflow.add_node("pessimist", pessimist_node)
    workflow.add_node("checker", checker_node)
    workflow.add_node("reporter", reporter_node)

    # Define edges
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "optimist")
    workflow.add_edge("researcher", "pessimist")
    workflow.add_edge("optimist", "checker")
    workflow.add_edge("pessimist", "checker")
    workflow.add_edge("checker", "reporter")
    workflow.add_edge("reporter", END)

    return workflow.compile()

