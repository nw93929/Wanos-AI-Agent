# Quick Start Guide

Get the AI Research Agent running in 5 minutes.

## Step 1: Clone & Setup (1 min)

```bash
# Clone repo
git clone <your-repo-url>
cd AI-Research-Agent

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Configure Environment (2 min)

```bash
# Copy template
cp .env.example .env

# Edit .env with your API keys
# Get keys from:
# - OpenAI: https://platform.openai.com/api-keys
# - Pinecone: https://app.pinecone.io/
# - FMP (optional): https://financialmodelingprep.com/developer/docs/
```

**Minimal .env for single-stock analysis**:
```bash
OPENAI_API_KEY=sk-your-key-here
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=financial-docs
```

**Full .env for stock screening**:
```bash
OPENAI_API_KEY=sk-your-key-here
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=financial-docs
FMP_API_KEY=your-fmp-key  # For screening
```

## Step 3: Verify Setup (1 min)

```bash
# Check environment variables
python check_env.py

# Should see:
# ✅ All required environment variables are set!
```

## Step 4: Start API Server (30 sec)

```bash
python api.py

# Should see:
# ============================================================
# Starting AI Research Agent API Server
# ============================================================
# INFO:     Started server process
# INFO:     Uvicorn running on http://0.0.0.0:8000
```

## Step 5: Test It! (30 sec)

### Option A: Automated Tests
```bash
# In another terminal
python test_api.py

# Should see:
# ✅ ALL TESTS PASSED!
```

### Option B: Manual Test (curl)
```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "instructions": "Quick analysis"}'

# Should return:
# {
#   "status": "queued",
#   "task_id": "uuid-here",
#   "ticker": "AAPL",
#   ...
# }
```

### Option C: Manual Test (Browser)
1. Open http://localhost:8000/docs
2. Try the `/research` endpoint with Swagger UI
3. Input: `{"ticker": "AAPL"}`
4. Click "Execute"

## Usage Examples

### Single Stock Analysis
```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "TSLA",
    "instructions": "Analyze Q4 2024 earnings and competitive position"
  }'
```

### Stock Screening (Warren Buffett Style)
```bash
curl -X POST http://localhost:8000/research/screen \
  -H "Content-Type: application/json" \
  -d '{
    "criteria": "Warren Buffett value investing",
    "max_stocks": 10,
    "sectors": ["Technology", "Healthcare"]
  }'
```

### Stock Screening (Growth Stocks)
```bash
curl -X POST http://localhost:8000/research/screen \
  -H "Content-Type: application/json" \
  -d '{
    "criteria": "High growth tech with strong momentum",
    "max_stocks": 5
  }'
```

## n8n Integration

### Workflow Setup
1. Add HTTP Request node in n8n
2. Configure:
   - Method: `POST`
   - URL: `http://localhost:8000/research`
   - Body: JSON
   ```json
   {
     "ticker": "{{$json.ticker}}",
     "instructions": "Analyze financial performance"
   }
   ```
3. Add Schedule Trigger (optional)
   - Run daily at 9 AM
   - Process list of tickers

### Example n8n Workflow
```
[Schedule: Daily 9AM]
    |
    v
[Webhook Trigger] → [HTTP Request: /research/screen]
    |
    v
[Wait 5 minutes] → [Parse JSON Response]
    |
    v
[Notion: Create Page with Top Stocks]
```

## What's Next?

### Customize the Agent
- **Change investment strategy**: Edit `agents/prompts_screening.py`
- **Adjust screening filters**: Edit `agents/screening_graph.py` (quick_filter_node)
- **Add data sources**: Edit `services/market_data.py`

### Monitor Performance
```bash
# View logs
tail -f api.log

# Check API health
curl http://localhost:8000/health
```

### Production Deployment
1. **Use Docker**:
   ```bash
   docker-compose up --build
   ```

2. **Deploy to cloud**:
   - Railway.app (easiest)
   - Render.com
   - AWS ECS
   - DigitalOcean App Platform

3. **Set environment variables in platform**

4. **Update n8n webhook URL** to deployed URL

## Troubleshooting

### "API key not set" error
- Run `python check_env.py`
- Verify `.env` file exists
- Check API key starts with `sk-`

### "Connection refused" error
- Make sure API server is running: `python api.py`
- Check port 8000 is not in use

### "No stocks passed filters" (screening)
- Filters may be too strict
- See TROUBLESHOOTING.md for solutions

### More issues?
- See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- Check logs: `python api.py 2>&1 | tee api.log`

## Documentation

- **[README.md](README.md)**: Full project overview
- **[SCREENING_GUIDE.md](SCREENING_GUIDE.md)**: Stock screening deep dive
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)**: Common issues & solutions
- **API Docs**: http://localhost:8000/docs (when server is running)

## Architecture Diagram

```
┌─────────┐     ┌──────────┐     ┌──────────────┐
│   n8n   │────▶│ FastAPI  │────▶│  LangGraph   │
│         │     │  Server  │     │   Agent      │
└─────────┘     └──────────┘     └──────────────┘
                                        │
                       ┌────────────────┼────────────────┐
                       │                │                │
                       v                v                v
                 ┌──────────┐    ┌──────────┐    ┌──────────┐
                 │  OpenAI  │    │ Pinecone │    │   SEC    │
                 │  GPT-4o  │    │ Vector DB│    │  EDGAR   │
                 └──────────┘    └──────────┘    └──────────┘
```

## Performance

- **Single stock**: 30-60 seconds
- **Screening 500 stocks**: 10-15 minutes
- **Cold start** (first run): +15 seconds (loading Phi-3 model)

## Cost Estimates (per query)

### Single Stock Analysis
- OpenAI API: ~$0.05 (3 GPT-4o calls)
- Pinecone: Free tier sufficient
- **Total**: $0.05

### Stock Screening (10 stocks)
- OpenAI API: ~$0.50 (50+ GPT-4o calls)
- FMP API: Free tier (250/day limit)
- **Total**: $0.50

### Ways to Reduce Costs
1. Use GPT-4o-mini instead of GPT-4o ($0.15/M tokens vs $5/M)
2. Cache screening results daily
3. Use local Phi-3 for all grading (saves ~40%)
4. Batch LLM calls (score 5 stocks per prompt)

## Support

- **Issues**: Open GitHub issue
- **Questions**: Check existing issues first
- **Contributions**: PRs welcome!

---

**Ready to start?** Run `python api.py` and visit http://localhost:8000/docs 🚀
