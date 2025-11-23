import json
import os
import random

os.makedirs("trainer_output", exist_ok=True)

def create_dummy_log(filename):
    log = []
    for epoch in range(1, 16):
        log.append({
            "epoch": epoch,
            "loss": random.uniform(0.1, 2.0) - (epoch * 0.05),
            "step": epoch * 100
        })
        log.append({
            "epoch": epoch,
            "eval_accuracy": 0.5 + (epoch * 0.03),
            "step": epoch * 100
        })
    
    with open(filename, "w") as f:
        json.dump(log, f)

create_dummy_log("trainer_output/lora_history.json")
create_dummy_log("trainer_output/frozen_history.json")
create_dummy_log("trainer_output/full_history.json")
print("Dummy data created.")
