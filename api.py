"""
FastAPI Endpoint with Results Storage
======================================

This version stores research results so they can be retrieved via task_id.

NEW: After submitting a task, you can poll for results:
1. POST /research -> returns task_id
2. GET /research/{task_id} -> returns status and result

Usage from n8n:
1. HTTP Request: POST /research
2. Wait node: 30 seconds
3. HTTP Request: GET /research/{{task_id}}
4. Check if status == "completed", if not loop back to step 2
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Literal
import asyncio
from uuid import uuid4
from datetime import datetime

# Import research functions
from main import run_research
from services.results_store import get_results_store, TaskStatus

# Initialize FastAPI app
api_app = FastAPI(
    title="AI Research Agent API v2",
    description="Autonomous financial research agent with result retrieval",
    version="2.0.0"
)

# Get results store
results_store = get_results_store()

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class SingleStockResearchRequest(BaseModel):
    """Request model for single stock analysis"""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL, TSLA)")
    instructions: str = Field(
        default="Analyze financial performance and market position",
        description="Specific research instructions"
    )

class StockScreeningRequest(BaseModel):
    """Request model for screening multiple stocks"""
    mode: Literal["screening"] = Field(default="screening")
    criteria: str = Field(
        default="Warren Buffett value investing",
        description="Investment criteria to use"
    )
    max_stocks: int = Field(default=10, ge=1, le=100)
    sectors: Optional[list[str]] = None

class ResearchResponse(BaseModel):
    """Response when task is queued"""
    status: str
    task_id: str
    ticker: Optional[str] = None
    message: str
    queued_at: str
    result_url: str  # Where to GET results

class TaskResultResponse(BaseModel):
    """Response when retrieving task results"""
    task_id: str
    status: str  # queued, running, completed, failed
    result: Optional[str] = None  # The actual report (if completed)
    error: Optional[str] = None  # Error message (if failed)
    metadata: dict
    created_at: str
    updated_at: str

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def run_research_with_storage(task_id: str, query: str, metadata: dict):
    """
    Wrapper that runs research and stores the result.

    This runs in the background and updates the task status:
    1. Mark as "running"
    2. Execute research
    3. Store result (or error)
    4. Mark as "completed" or "failed"
    """
    try:
        # Update status to running
        results_store.update_status(task_id, TaskStatus.RUNNING)

        # Run the actual research (blocking)
        final_state = await run_research(query)

        # Extract report from final state
        report = final_state.get("report", "No report generated")

        # Store successful result
        results_store.store_result(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            result=report,
            metadata=metadata
        )

        print(f"[API] Task {task_id} completed successfully")

    except Exception as e:
        # Store error
        results_store.store_result(
            task_id=task_id,
            status=TaskStatus.FAILED,
            error=str(e),
            metadata=metadata
        )

        print(f"[API] Task {task_id} failed: {e}")

# ============================================================================
# API ENDPOINTS
# ============================================================================

@api_app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "AI Research Agent API v2",
        "status": "running",
        "features": ["result_storage", "task_polling"],
        "endpoints": {
            "submit_single": "POST /research",
            "submit_screening": "POST /research/screen",
            "get_results": "GET /research/{task_id}",
            "list_tasks": "GET /research",
            "health": "GET /health"
        }
    }

@api_app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "langraph": "available",
            "results_store": "available",
            "pinecone": "connected"
        }
    }

@api_app.post("/research", response_model=ResearchResponse)
async def trigger_single_stock_research(
    request: SingleStockResearchRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger research for a single stock.

    Returns task_id immediately. Use GET /research/{task_id} to retrieve results.
    """
    task_id = str(uuid4())
    query = f"{request.instructions} for {request.ticker}"

    # Store initial task status
    metadata = {
        "ticker": request.ticker,
        "instructions": request.instructions,
        "type": "single_stock",
        "created_at": datetime.now().isoformat()
    }

    results_store.store_result(
        task_id=task_id,
        status=TaskStatus.QUEUED,
        metadata=metadata
    )

    # Add task to background
    background_tasks.add_task(run_research_with_storage, task_id, query, metadata)

    return ResearchResponse(
        status="queued",
        task_id=task_id,
        ticker=request.ticker,
        message=f"Research task queued for {request.ticker}",
        queued_at=datetime.now().isoformat(),
        result_url=f"/research/{task_id}"
    )

@api_app.post("/research/screen", response_model=ResearchResponse)
async def trigger_stock_screening(
    request: StockScreeningRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger batch screening of multiple stocks.

    Returns task_id immediately. Poll GET /research/{task_id} for results.
    """
    task_id = str(uuid4())

    query = f"Screen stocks using {request.criteria} criteria. "
    if request.sectors:
        query += f"Focus on sectors: {', '.join(request.sectors)}. "
    query += f"Return top {request.max_stocks} recommendations."

    metadata = {
        "criteria": request.criteria,
        "max_stocks": request.max_stocks,
        "sectors": request.sectors,
        "type": "screening",
        "created_at": datetime.now().isoformat()
    }

    results_store.store_result(
        task_id=task_id,
        status=TaskStatus.QUEUED,
        metadata=metadata
    )

    background_tasks.add_task(run_research_with_storage, task_id, query, metadata)

    return ResearchResponse(
        status="queued",
        task_id=task_id,
        ticker=None,
        message=f"Stock screening queued with {request.criteria} criteria",
        queued_at=datetime.now().isoformat(),
        result_url=f"/research/{task_id}"
    )

@api_app.get("/research/{task_id}", response_model=TaskResultResponse)
async def get_research_results(task_id: str):
    """
    Retrieve results of a research task.

    Status codes:
    - "queued": Task is waiting to start
    - "running": Task is currently executing
    - "completed": Task finished successfully, result available
    - "failed": Task failed, error message available

    Poll this endpoint until status is "completed" or "failed".
    """
    result = results_store.get_result(task_id)

    if not result:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    return TaskResultResponse(**result)

@api_app.get("/research", response_model=list[TaskResultResponse])
async def list_recent_tasks(limit: int = 10):
    """
    List recent research tasks.

    Useful for debugging or building a dashboard.
    """
    tasks = results_store.list_recent_tasks(limit=limit)
    return [TaskResultResponse(**task) for task in tasks]

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("Starting AI Research Agent API Server v2")
    print("=" * 60)
    print("📊 Submit Research: POST /research")
    print("🔍 Submit Screening: POST /research/screen")
    print("📥 Get Results: GET /research/{task_id}")
    print("📋 List Tasks: GET /research")
    print("💚 Health Check: GET /health")
    print("=" * 60)
    print("\nNEW: Results are now stored and retrievable!")
    print("=" * 60)

    uvicorn.run(
        "api:api_app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
