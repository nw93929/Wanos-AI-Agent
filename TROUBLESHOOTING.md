# Troubleshooting Guide

## Error: "The api_key client option must be set"

**Problem**: OpenAI API key not found

**Solution**:
1. Check if `.env` file exists in project root
   ```bash
   ls -la .env  # Should see the file
   ```

2. Verify `.env` contains your API key:
   ```bash
   cat .env | grep OPENAI_API_KEY
   # Should show: OPENAI_API_KEY=sk-...
   ```

3. If `.env` doesn't exist, create it from template:
   ```bash
   cp .env.example .env
   # Then edit .env with your actual keys
   ```

4. Run environment checker:
   ```bash
   python check_env.py
   ```

5. **IMPORTANT**: Never commit `.env` to git!
   ```bash
   git status  # Should NOT show .env
   ```

## Error: Import errors (ModuleNotFoundError)

**Problem**: Missing dependencies

**Solution**:
```bash
pip install -r requirements.txt
```

If still failing, try upgrading pip:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

## Error: "Connection refused" or "Cannot connect to localhost:8000"

**Problem**: API server not running

**Solution**:
1. Start the API server in one terminal:
   ```bash
   python api.py
   ```

2. Test from another terminal:
   ```bash
   python test_api.py
   ```

3. Or test with curl:
   ```bash
   curl http://localhost:8000/health
   ```

## Error: "Internal Server Error" (500) when submitting tasks

**Problem**: Redis connection failure when `REDIS_URL` environment variable is set but Redis isn't running

**Symptoms**:
- POST /research returns 500 error
- API logs show: `ConnectionError: Error 10061 connecting to localhost:6379`

**Solution**:
1. **Option A**: Remove `REDIS_URL` from your .env file if you don't need Redis
   ```bash
   # Remove or comment out this line in .env:
   # REDIS_URL=redis://localhost:6379/0
   ```

2. **Option B**: Install and start Redis if you want persistent storage
   - **Windows**: Download Redis from https://github.com/microsoftarchive/redis/releases
   - **Linux**: `sudo apt-get install redis-server && sudo systemctl start redis`
   - **macOS**: `brew install redis && brew services start redis`

**Note**: The API now automatically falls back to in-memory storage if Redis connection fails. Results will be lost when the server restarts, but the API will continue to function.

## Error: FastAPI validation error (422)

**Problem**: Missing required fields in request

**Solution**: Check request body includes all required fields

**Single stock research requires**:
```json
{
  "ticker": "AAPL"  // Required
  "instructions": "..."  // Optional
}
```

**Stock screening requires**:
```json
{
  "mode": "screening",  // Optional (defaults to "screening")
  "criteria": "...",    // Optional (defaults to "Warren Buffett")
  "max_stocks": 10,     // Optional (defaults to 10)
  "sectors": [...]      // Optional
}
```

## Error: "Rate limit exceeded" (429)

**Problem**: Too many API calls to external services

**Solutions**:

### For OpenAI:
- Check usage at: https://platform.openai.com/usage
- Upgrade plan or wait for rate limit reset
- Reduce `max_stocks` parameter

### For Financial Modeling Prep:
- Free tier: 250 calls/day
- Each stock screened = 2-3 API calls
- Limit screening to 50-100 stocks at a time
- Upgrade to paid tier: https://financialmodelingprep.com/pricing

## Error: "CUDA out of memory" or "Killed"

**Problem**: Phi-3 model requires too much RAM

**Solutions**:
1. **Use CPU instead of GPU**: Already configured with `device_map="auto"`

2. **Disable local grading model**: Comment out grader node in graph.py:
   ```python
   # workflow.add_node("grader", grader_node)
   # workflow.add_edge("writer", "grader")
   workflow.add_edge("writer", END)  # Skip grading
   ```

3. **Use GPT-4o for grading** (costs ~$0.01 per report):
   ```python
   def grader_node(state: AgentState) -> dict:
       model = get_reasoning_model()
       # ... use model instead of eval_model
   ```

## Error: Pinecone connection failed

**Problem**: Invalid Pinecone credentials or index doesn't exist

**Solutions**:
1. Verify API key: https://app.pinecone.io/
2. Check index name matches your Pinecone index
3. Create index if needed:
   ```python
   from pinecone import Pinecone
   pc = Pinecone(api_key="your-key")
   pc.create_index(
       name="financial-docs",
       dimension=1536,  # OpenAI embedding size
       metric="cosine"
   )
   ```

## Error: "No stocks passed filters"

**Problem**: Screening filters too strict

**Solution**: Relax filters in `agents/screening_graph.py`:
```python
# Before (strict):
if (market_cap > 1_000_000_000 and
    roe > 0.15 and
    debt_to_equity < 1.0):

# After (relaxed):
if (market_cap > 500_000_000 and  # Lower threshold
    roe > 0.10 and                # Accept 10%+ ROE
    debt_to_equity < 1.5):        # Allow more leverage
```

## Slow Performance (> 30 minutes)

**Problem**: Sequential API calls are slow

**Short-term solutions**:
1. Reduce universe size (screen S&P 100 instead of 500)
2. Reduce `max_stocks` parameter (screen top 5 instead of 10)
3. Cache fundamentals data daily

**Long-term solution**: Implement async/parallel API calls
```python
import asyncio
import aiohttp

async def fetch_many(tickers):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_one(session, ticker) for ticker in tickers]
        return await asyncio.gather(*tasks)
```

## Docker Issues

### Container won't start
```bash
# Check logs
docker-compose logs agent_api

# Rebuild from scratch
docker-compose down -v
docker-compose build --no-cache
docker-compose up
```

### Environment variables not loading in Docker
Make sure `docker-compose.yml` has:
```yaml
services:
  agent_api:
    env_file: .env  # This line!
```

## n8n Integration Issues

### Webhook not receiving data
1. Check n8n HTTP Request node settings:
   - Method: POST
   - URL: `http://localhost:8000/research`
   - Body: JSON
   - Headers: `Content-Type: application/json`

2. Test endpoint directly first:
   ```bash
   curl -X POST http://localhost:8000/research \
     -H "Content-Type: application/json" \
     -d '{"ticker": "AAPL"}'
   ```

### Task queued but no results
Background tasks run async - they don't return results directly.

**Options**:
1. **Implement result polling**: Store results in Redis/DB, poll with task_id
2. **Use webhooks**: Have agent POST results back to n8n webhook when done
3. **Check logs**: Results are printed to console
   ```bash
   docker-compose logs -f agent_api | grep "FINAL RESEARCH"
   ```

## GitHub Security Check

Before pushing to public repo:
```bash
# 1. Verify .gitignore works
echo "test-secret" > .env
git status  # Should NOT show .env

# 2. Search for accidentally committed secrets
git log -p | grep -i "api.key\|sk-\|secret"

# 3. If secrets found in history, use git-filter-repo:
pip install git-filter-repo
git-filter-repo --path .env --invert-paths
```

## Still Stuck?

1. **Check logs**:
   ```bash
   python api.py 2>&1 | tee api.log
   ```

2. **Enable debug mode**:
   ```python
   # In api.py
   uvicorn.run(..., log_level="debug")
   ```

3. **Test individual components**:
   ```bash
   # Test graph without API
   python main.py

   # Test data fetching
   python -c "from services.market_data import check_api_health; print(check_api_health())"
   ```

4. **Create minimal reproduction**:
   ```python
   # test_minimal.py
   from dotenv import load_dotenv
   load_dotenv()

   from langchain_openai import ChatOpenAI
   model = ChatOpenAI(model="gpt-4o")
   print(model.invoke("Say hello"))
   ```

5. **Open an issue**: Include:
   - Error message (sanitize API keys!)
   - Python version: `python --version`
   - OS: Windows/Mac/Linux
   - Steps to reproduce
