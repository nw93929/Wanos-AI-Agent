from apscheduler.schedulers.blocking import BlockingScheduler
from main import run_research
import asyncio

def job_wrapper(query):
    print(f"--- Triggering Scheduled Task: {query} ---")
    asyncio.run(run_research(query))

scheduler = BlockingScheduler()

# List of tickers for analysis
tickers = ["NVDA", "TSLA", "KO", "DIS", "JOBY", "DVLT", "IREN", "AMPX", "TMQ"]

for ticker in tickers:
    scheduler.add_job(
        job_wrapper,       
        'cron', 
        day_of_week='mon', 
        hour=5, 
        minute=0, 
        args=[f"Analyze {ticker}'s performance and market position"]
    )

if __name__ == "__main__":
    print("Scheduler started. Reports scheduled for every Monday at 5:00 AM.")
    print("Press Ctrl+C to exit.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass
