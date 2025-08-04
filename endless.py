import time
import subprocess
from datetime import datetime
from experiment_helpers import gpu_details
import os


gpu_details.print_details()

def get_pending_job_ids(user):
    result = subprocess.run(
        ["squeue", "-u", user, "-t", "PENDING", "-h", "-o", "%A"],
        capture_output=True,
        text=True,
        check=True
    )
    job_ids = result.stdout.strip().split("\n")
    return [jid for jid in job_ids if jid]

def run_job():
    print(f"[{datetime.now()}] Running job...")

    if os.getcwd().find("jlb638")!=-1:
        pending=get_pending_job_ids("jlb638")
    else:
        pending=get_pending_job_ids("jbaker15")

    print("penidnign jobs: ", len(pending))
    if len(pending)==0:
        # Step 1: Run find_oom_jobs.py and save output to oom.sh
        with open("oom.sh", "w") as f:
            subprocess.run(["python", "find_oom_jobs.py"], stdout=f, check=True)

        with open('oom.sh', 'r') as f:
            num_lines = sum(1 for _ in f)

        print(f"Number of jobs: {num_lines}")
        
        # Step 2: Run the generated oom.sh script
        subprocess.run(["bash", "oom.sh"], check=True)

    print(f"[{datetime.now()}] Job completed.")





count=0
while True:
    run_job()
    time.sleep(10 * 60)  # Wait for 10 minutes
    count+=1
