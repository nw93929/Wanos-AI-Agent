from langgraph.graph import StateGraph, START, END
from .state import AgentState
from .prompts import *

'''
Defines the nodes and edges of the agent's workflow graph. Each node represents a step in the research process,
utilizing the prompts defined in prompts.py. Each edge defines the flow of data and control between these steps. Implements the Plan-Execute-Review loop. 
If the report quality is below a threshold, it loops back to the Researcher for further data gathering.
'''

def planner_node(state: AgentState):
    '''
    calls LLM with the Planner prompt to create a research plan.
    '''
    return {
        "system": PLANNER_SYSTEM,
        "input_keys": ["task"],
        "output_keys": ["plan"]
    }

def researcher_node(state: AgentState):
    '''
    calls LLM with the Researcher prompt to gather data based on the current plan step.
    '''
    return {
        "system": RESEARCHER_SYSTEM,
        "input_keys": ["plan", "research_notes"],
        "output_keys": ["research_notes"]
    }

def writer_node(state: AgentState):
    '''
    calls LLM with the Writer prompt to draft the report.
    '''
    return {
        "system": WRITER_SYSTEM,
        "input_keys": ["research_notes"],
        "output_keys": ["report"]
    }

def grader_node(state: AgentState):
    '''
    calls LLM with the Grader prompt to evaluate the report quality.
    '''
    return {
        "system": GRADER_SYSTEM,
        "input_keys": ["report", "plan"],
        "output_keys": ["score"]
    }

# graph definition
workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)    
workflow.add_node("grader", grader_node)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "grader")

def decide_to_end(state: AgentState):
    '''
    Decides whether to end the workflow or loop back to the Researcher based on the report score.
    '''
    return "researcher" if state["score"] < 85 else END

workflow.add_conditional_edges("grader", decide_to_end)
app = workflow.compile()