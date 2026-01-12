import operator
from typing import Annotated, List, TypedDict, Optional

'''
This is the agent's shared memory, it defines a TypedDict that every node in the graph will share to read and write to.
'''

class AgentState(TypedDict):
    task: str
    plan: List[str]  # steps to take created by Planner
    research_notes: Annotated[List[str], operator.add]  # operator.add to allow node collaboration without overwriting
    report: Optional[str]
    score: int  # quality score of the report from Grader
    loop_count: int