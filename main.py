import asyncio
from agents.graph import app  # Imports the compiled LangGraph
from uuid import uuid4

async def run_research(user_query: str):
    # 'config' contains the thread_id for persistence/checkpointing
    config = {"configurable": {"thread_id": str(uuid4())}}
    
    initial_state = {
        "task": user_query,
        "research_notes": [],
        "loop_count": 0
    }

    # Streaming the graph execution so the client sees progress
    async for event in app.astream(initial_state, config):
        for node_name, output in event.items():
            print(f"--- Finished Node: {node_name} ---")
            # In a real app, you'd send this to a frontend via WebSockets
    
    # Final result
    final_state = await app.aget_state(config)
    print("\nFINAL REPORT:\n", final_state.values.get("report"))

if __name__ == "__main__":
    query = "Research the impact of generative AI on PostgreSQL performance optimization."
    asyncio.run(run_research(query))