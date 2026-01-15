# AI-Research-Agent

Autonomous financial research agent using LangGraph with a Plan-Execute-Review loop. Queries internal Pinecone vector store for structured research and produces validated, high-accuracy investment analysis reports.

## Architecture Overview

This agent implements a self-correcting workflow optimized for financial stock research:

```
┌─────────┐      ┌────────────┐       ┌────────┐       ┌────────┐
│ PLANNER │─────▶│ RESEARCHER │─────▶│ WRITER │─────▶│ GRADER │
└─────────┘      └────────────┘       └────────┘       └────────┘
   GPT-4o         Pinecone RAG         GPT-4o          Phi-3
                        ▲                                  │
                        │         score < 85?              │
                        └──────────────────────────────────┘
                                (loop back)
```

### Key Features:
- **Hybrid LLM Strategy**: Cloud models (GPT-4o) for complex reasoning, local quantized models (Phi-3) for grading
- **RAG Integration**: Pinecone vector database with LlamaIndex for document retrieval
- **Quality Assurance**: Automated scoring with feedback loops (loops until score ≥ 85 or 3 iterations)
- **Production-Ready**: Docker Compose setup with Redis task queues, PostgreSQL storage, and scheduled execution

## Project Structure

### agents/
**The core orchestration layer**

- **graph.py**: LangGraph workflow definition
  - Defines 4 nodes: Planner, Researcher, Writer, Grader
  - Implements conditional looping logic
  - Lazy-loads Phi-3 model (2GB quantized) for efficiency

- **prompts.py**: System prompts for each agent persona
  - Optimized for financial analysis with specific output requirements
  - Enforces structured scoring (0-100 scale) for grader

- **state.py**: TypedDict defining shared memory across nodes
  - Tracks task, plan, research_notes (additive), report, score, loop_count
  - Uses `operator.add` for accumulating research across loops

### services/
**Data integration layer**

- **pinecone_llamaindex.py**: RAG implementation
  - Connects to Pinecone vector store via LlamaIndex
  - Embeds queries using OpenAI's `text-embedding-3-small`
  - Returns top-k relevant document chunks via cosine similarity

- **postgres_client.py**: Secure database client
  - Connection pooling (1-10 connections)
  - Parameterized queries to prevent SQL injection
  - Helper functions for common financial queries

- **search_tools.py**: External web search integration (Tavily API)

### data/
**Document processing and embeddings**

- **embeddings.py**: Document chunking and vectorization pipeline
  - Uses LlamaIndex `TokenTextSplitter` (512 tokens, 50 overlap)
  - Generates embeddings for upload to Pinecone

### evaluation/
**Quality control**

- **scorer.py**: Structured LLM-based report evaluation
  - Uses Pydantic models to enforce score + critique output format
  - Calibrates against quality thresholds

### workers/
**Background execution and scheduling**

- **scheduler.py**: APScheduler cron jobs
  - Example: Daily research briefings at 9:00 AM
  - Async-compatible wrapper for LangGraph workflows

- **tasks.py**: Celery task definitions
  - Offloads research to background workers
  - Enables concurrent multi-query processing

### tests/
- **test_graph.py**: Unit tests for graph nodes
- **test_eval.py**: Validates scorer accuracy
- **test_cases.py**: End-to-end workflow tests with must-include fact checking
- **test_docker.py**: Infrastructure health checks (Postgres, Redis connectivity)

## Configuration Files

**config.py**: Centralized environment variable management

**requirements.txt**: Production dependencies
- langgraph>=0.4.0 - Workflow orchestration
- llama-index>=0.12.0 - RAG framework
- transformers>=4.48.0, bitsandbytes>=0.49.0 - Local model quantization
- pinecone-client>=5.4.0 - Vector database
- celery>=5.4.0, redis>=5.2.0 - Task queuing

**docker-compose.yml**: Multi-container setup
- agent_api: Main scheduler service
- worker: Celery research executor
- redis: Message broker with health checks
- db: PostgreSQL for metadata/results storage

**Dockerfile**: Unstructured.io base image for PDF processing

**main.py**: CLI entry point for manual execution

## Workflow Explained

### 1. Planner Node (GPT-4o)
- Decomposes user query into research steps
- Prioritizes SEC filings, earnings transcripts, industry benchmarks
- Defines success criteria for each step

### 2. Researcher Node (Pinecone + LlamaIndex)
- Converts query to embedding vector
- Searches Pinecone for similar document chunks (cosine similarity)
- Returns top-3 most relevant contexts
- Accumulates findings across loop iterations

### 3. Writer Node (GPT-4o)
- Synthesizes research into markdown investment report
- Enforces structure: Executive Summary, Financials, Valuation, Risks, Thesis
- Includes both bullish and bearish perspectives

### 4. Grader Node (Phi-3 Local)
- Scores report 0-100 on:
  - Data density (40 pts)
  - Source credibility (30 pts)
  - Analytical balance (30 pts)
- Deducts points for missing sections or vague statements

### 5. Conditional Loop
- If score < 85 AND loops < 3: Return to Researcher for more context
- Else: Return final report

## Getting Started

### Prerequisites
- Docker & Docker Compose
- Python 3.10+
- GPU recommended for Phi-3 model (CPU fallback supported)

### Environment Variables
Create `.env` file:
```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=financial-docs
POSTGRES_URI=postgresql://user:password@db:5432/research_db
TAVILY_API_KEY=tvly-...
```

### Run Locally
```bash
# Build and start services
docker-compose up --build

# Or run manually
python main.py
```

### Example Query
```python
query = "Analyze Tesla's Q4 2024 financial performance and competitive position in the EV market"
asyncio.run(run_research(query))
```

## Tech Stack Rationale

**Why LangGraph over LangChain?**
- Stateful workflows with checkpoints
- Conditional routing (loop back if quality low)
- Better observability with streaming updates

**Why Pinecone over Chroma/FAISS?**
- Serverless vector database (no infra management)
- Sub-100ms query latency at scale
- Metadata filtering for time-series financial data

**Why Quantize Phi-3?**
- 75% memory reduction (7.6GB → 1.9GB)
- Sufficient quality for structured scoring tasks
- Eliminates API costs for grading

**Why Celery + Redis?**
- Horizontal scaling (add more worker containers)
- Retry logic for failed research tasks
- Priority queues for urgent vs batch queries

## Performance Characteristics

- **Cold start**: ~15-20 seconds (loading Phi-3 model)
- **Warm execution**: 30-60 seconds per research query
- **Peak throughput**: 10-20 queries/minute (with 5 workers)
- **Memory**: 4GB per worker (2GB model + 2GB overhead)

## Future Enhancements

- [ ] Real-time data ingestion from financial APIs
- [ ] Multi-agent debates (bull vs bear agents)
- [ ] Portfolio-level analysis (correlations, risk metrics)
- [ ] Integration with trading platforms for signal generation
- [ ] Custom embedding fine-tuning on financial terminology

## Contributing

This is a personal learning project, but feedback welcome via Issues.

## License

MIT License - see LICENSE file
