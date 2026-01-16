"""
LangGraph Agent Workflow Definition
====================================

This file defines the core agent graph - the "brain" of the research system.

## For ML/DS Engineers:

### The Graph Architecture
Think of this like a state machine or directed graph where:
- **Nodes** = Functions that transform state (like layers in a neural network pipeline)
- **Edges** = Data flow between nodes
- **State** = Shared memory (dict) that accumulates information
- **Conditional Edges** = Branching logic (like if/else for the next node)

### Why Use Graphs Instead of Sequential Pipelines?
Linear pipeline:  A -> B -> C -> D
Agent graph:      A -> B -> C -> D -> (if quality low) -> B (loop back)

This allows:
1. **Self-correction**: Agent can retry with more data if output is poor
2. **Dynamic routing**: Different paths based on intermediate results
3. **Stateful iteration**: Each loop accumulates more context

### Model Strategy: Hybrid Approach
- **Cloud LLMs (GPT-4o)**: For complex reasoning (planning, writing)
  - Pros: State-of-the-art quality, no local compute
  - Cons: API costs, latency

- **Local LLMs (Phi-3 quantized)**: For simple tasks (grading, classification)
  - Pros: No API costs, runs locally, faster for simple tasks
  - Cons: Lower quality, requires GPU/RAM

### Quantization Explained (for the Phi-3 model):
Standard model: 16-bit floats (2 bytes per parameter)
Quantized model: 4-bit ints (0.5 bytes per parameter)

For a 3.8B parameter model:
- Full precision: 3.8B * 2 bytes = 7.6 GB VRAM
- 4-bit quantized: 3.8B * 0.5 bytes = 1.9 GB VRAM

This is 75% memory reduction with minimal quality loss for simple tasks.

The quantization technique used here (BitsAndBytes NF4):
- NF4 = NormalFloat 4-bit (optimized for normally distributed weights)
- Double quantization = Quantize the quantization constants themselves
- Compute dtype float16 = Do actual computations in FP16 for speed

### The Workflow Nodes:
"""

import re
import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# Load environment variables FIRST (before any API calls)
load_dotenv()

# Import project modules
from agents.state import AgentState
from agents.prompts import PLANNER_SYSTEM, WRITER_SYSTEM, GRADER_SYSTEM
from services.pinecone_llamaindex import query_pinecone_llamaindex

# ============================================================================
# GLOBAL MODELS INITIALIZATION (Lazy Loading Pattern)
# ============================================================================
# Both models are lazy-loaded to:
# 1. Ensure .env is loaded before API key access
# 2. Faster startup time
# 3. Avoid memory usage if not needed

_reasoning_model = None  # Global cache for GPT-4o

def get_reasoning_model():
    """
    Lazy-loads GPT-4o model for planning and writing.

    Returns:
        ChatOpenAI: LangChain wrapper for GPT-4o
    """
    global _reasoning_model

    if _reasoning_model is None:
        # Verify API key is present
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it in your .env file or export it."
            )

        _reasoning_model = ChatOpenAI(model="gpt-5-nano", temperature=0)
        print("[INFO] GPT-5-nano model initialized successfully")

    return _reasoning_model

# ============================================================================
# LOCAL EVALUATION MODEL (Lazy Loading Pattern)
# ============================================================================
# Instead of loading the 2GB model at import time, we load it only when grader_node
# is first called. This is called "lazy initialization" or "lazy loading".
#
# Benefits:
# - Faster startup time (important for testing, quick queries)
# - No memory usage if grading isn't needed
# - Can skip GPU allocation if only using planner/writer

_eval_model = None  # Global cache for the model

def get_eval_model():
    """
    Lazy-loads the local Phi-3 model for grading reports.

    This function implements a singleton pattern - the model is loaded once
    and cached for subsequent calls.

    Why Phi-3?
    ----------
    - Small enough to run on consumer hardware (1.9GB quantized)
    - Strong instruction-following for structured tasks
    - Fast inference for scoring tasks

    Returns:
        HuggingFacePipeline: LangChain wrapper around the model
    """
    global _eval_model

    if _eval_model is None:
        print("[INFO] Loading local Phi-3 model for grading (first run only)...")

        model_id = "microsoft/Phi-3-mini-4k-instruct"

        # Quantization Config: Reduces 16-bit to 4-bit (Saves 60-70% VRAM/RAM)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,                    # Use 4-bit quantization
            bnb_4bit_quant_type="nf4",           # NormalFloat4 - optimized for model weights
            bnb_4bit_compute_dtype=torch.float16, # Compute in FP16 (faster than FP32)
            bnb_4bit_use_double_quant=True       # Quantize the quantization constants
        )

        # Load tokenizer (converts text to numbers)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Load model with local cache persistence
        # device_map="auto" automatically splits model across CPU/GPU if needed
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto"
        )

        # Wrap in HuggingFace pipeline (higher-level API)
        hf_pipe = pipeline(
            "text-generation",              # Task type
            model=hf_model,
            tokenizer=tokenizer,
            max_new_tokens=150,             # Max length of generated text
            temperature=0.1,                # Low temperature = more deterministic
            return_full_text=False          # Only return generated text, not prompt
        )

        # Wrap in LangChain interface for compatibility
        _eval_model = HuggingFacePipeline(pipeline=hf_pipe)
        print("[INFO] Model loaded successfully.")

    return _eval_model

# ============================================================================
# AGENT NODE DEFINITIONS
# ============================================================================
# Each node is a function that:
# 1. Takes current state as input
# 2. Performs some action (LLM call, database query, etc.)
# 3. Returns a dict of updates to merge into state

def planner_node(state: AgentState) -> dict:
    """
    PLANNER NODE: Strategic thinking and task decomposition.

    Role: Acts like a senior analyst planning a research project.

    Input from state:
        - task: The user's research query

    Output to state:
        - plan: List of research steps to execute
        - loop_count: Incremented iteration counter

    LLM Strategy:
        Uses GPT-4o because planning requires:
        - Understanding complex financial concepts
        - Breaking down multi-faceted questions
        - Prioritizing information sources
    """
    new_count = state.get("loop_count", 0) + 1

    model = get_reasoning_model()  # Lazy load

    response = model.invoke([
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": state["task"]}
    ])

    return {"plan": [response.content], "loop_count": new_count}

def researcher_node(state: AgentState) -> dict:
    """
    RESEARCHER NODE: Information retrieval from vector database.

    Role: Acts like a research assistant gathering data from archives.

    How RAG Works Here:
    -------------------
    1. Takes the user's query (state["task"])
    2. Converts query to an embedding (vector) using OpenAI's embedding model
    3. Searches Pinecone vector database for similar document embeddings
    4. Returns top-k most relevant chunks (cosine similarity)
    5. Passes these chunks as context to subsequent nodes

    Input from state:
        - task: Original research question

    Output to state:
        - research_notes: Relevant text chunks from vector DB (additive)

    Why LlamaIndex?
    ---------------
    LlamaIndex is specifically built for connecting LLMs to external data.
    It handles:
    - Embedding generation
    - Vector store interfacing
    - Context window management (chunking large docs)
    - Source attribution
    """
    # Cross-service call to LlamaIndex service
    internal_context = query_pinecone_llamaindex(state["task"])

    return {"research_notes": [f"Retrieved Context: {internal_context}"]}

def writer_node(state: AgentState) -> dict:
    """
    WRITER NODE: Synthesizes research into professional report.

    Role: Acts like a financial analyst writing an investment thesis.

    Input from state:
        - research_notes: All accumulated context from researcher
        - task: Original question for reference

    Output to state:
        - report: Final markdown-formatted investment analysis

    LLM Strategy:
        Uses GPT-4o because writing requires:
        - Synthesizing multiple sources
        - Maintaining analytical tone
        - Structuring complex information
        - Creating coherent narratives
    """
    # Combine all research notes into single context string
    full_context = "\n".join(state["research_notes"])

    model = get_reasoning_model()  # Lazy load

    response = model.invoke([
        {"role": "system", "content": WRITER_SYSTEM},
        {"role": "user", "content": f"Use this context: {full_context} to complete: {state['task']}"}
    ])

    return {"report": response.content}

def grader_node(state: AgentState) -> dict:
    """
    GRADER NODE: Quality assurance scoring using local model.

    Role: Acts like a QA reviewer scoring report quality.

    Why Use Local Model Here?
    --------------------------
    Grading is a structured, deterministic task:
    - Input: Text report
    - Output: Single integer 0-100
    - No creativity required
    - Fast inference needed (< 1 second)

    This is perfect for a small local model - saves API costs and latency.

    Input from state:
        - report: The generated markdown report

    Output to state:
        - score: Integer 0-100 quality score

    Score Extraction Logic:
    -----------------------
    LLMs sometimes return extra text like "Score: 85" or "The score is 85/100"
    We use regex to extract the number robustly.

    Auto-normalization handles edge cases:
    - If model returns 8.5 (thinking 1-10 scale), we multiply by 10
    - If model returns 850 (overshoot), we cap at 100
    """
    eval_model = get_eval_model()  # Lazy-load the model

    prompt = f"{GRADER_SYSTEM}\n\nReview this report and provide a score out of 100:\n{state['report']}"
    response = eval_model.invoke(prompt)

    # Robust numeric extraction using regex
    # Finds all sequences of digits in the response
    nums = re.findall(r'\d+', str(response))

    if nums:
        # Take the last number to avoid metadata/dates at the start
        score = int(nums[-1])

        # Auto-normalize if model gives 1-10 instead of 1-100
        if score <= 10:
            score *= 10
    else:
        # Fallback if no number found (shouldn't happen with good prompt)
        score = 0

    # Cap at 100 in case of overshoot
    return {"score": min(score, 100)}

# ============================================================================
# GRAPH ASSEMBLY
# ============================================================================

workflow = StateGraph(AgentState)

# Add nodes to the graph
workflow.add_node("planner", planner_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)
workflow.add_node("grader", grader_node)

# Define linear edges (deterministic flow)
workflow.add_edge(START, "planner")        # Always start with planning
workflow.add_edge("planner", "researcher")  # Plan -> Research
workflow.add_edge("researcher", "writer")   # Research -> Write
workflow.add_edge("writer", "grader")       # Write -> Grade

# ============================================================================
# CONDITIONAL ROUTING LOGIC
# ============================================================================

def decide_to_end(state: AgentState) -> str:
    """
    Determines whether to loop back for more research or end the workflow.

    This is the "intelligence" of the self-correction loop.

    Exit Conditions:
    ----------------
    1. MAX_LOOPS reached (prevent infinite loops)
       - Safety mechanism in case quality never improves

    2. Quality threshold met (score >= 85)
       - Report is good enough, no need for more research

    Loop Condition:
    ---------------
    - Score < 85 AND loops < 3
       - Report needs improvement, gather more data

    Returns:
        "end": Finish workflow, return final report
        "researcher": Go back to researcher node for more context
    """
    # Safety check: prevent infinite loops
    if state.get("loop_count", 0) >= 3:
        return "end"

    # Quality check: is the report good enough?
    if state.get("score", 0) < 85:
        return "researcher"  # Loop back for more data

    return "end"  # Report is satisfactory

# Add the conditional edge
workflow.add_conditional_edges(
    "grader",           # After grading
    decide_to_end,      # Use this function to decide next step
    {
        "researcher": "researcher",  # If function returns "researcher", go there
        "end": END                   # If function returns "end", finish
    }
)

# Compile the graph into an executable application
app = workflow.compile()

# ============================================================================
# WORKFLOW DIAGRAM
# ============================================================================
"""
Visual representation of the graph:

    START
      |
      v
   PLANNER (GPT-4o)
      |
      v
  RESEARCHER (Pinecone RAG)
      |
      v
   WRITER (GPT-4o)
      |
      v
   GRADER (Phi-3 local)
      |
      v
   Decision Point:
      - score >= 85 OR loops >= 3? -> END
      - score < 85 AND loops < 3? -> RESEARCHER (loop)

This creates a feedback loop where poor-quality reports trigger more research.

Example execution:
1. User asks: "Analyze Tesla stock"
2. Planner: Creates research steps
3. Researcher: Queries Pinecone for Tesla financial data
4. Writer: Generates report from findings
5. Grader: Scores report = 72/100 (too low)
6. Loop back to Researcher with accumulated context
7. Researcher: Queries for more Tesla data
8. Writer: Regenerates report with more context
9. Grader: Scores report = 89/100 (passes!)
10. END: Return final report
"""
