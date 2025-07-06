import time
import subprocess
from datetime import datetime
from experiment_helpers import gpu_details
import os

gpu_details.print_details()

def run_job():
    print(f"[{datetime.now()}] Running job...")

    # Step 1: Run find_oom_jobs.py and save output to oom.sh
    with open("oom.sh", "w") as f:
        subprocess.run(["python", "find_oom_jobs.py"], stdout=f, check=True)
    
    # Step 2: Run the generated oom.sh script
    subprocess.run(["bash", "oom.sh"], check=True)

    print(f"[{datetime.now()}] Job completed.")



count=0
while True:
    run_job()
    time.sleep(6 * 60 * 60)  # Wait for 6 hours
    count+=1
