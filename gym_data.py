#!/usr/bin/env python3
"""
gym_data.py

Simulates a gym environment for ML practice:
 - 15 machines arranged in clusters by muscle group (Push, Pull, Leg),
   placed in the x–z plane (y=0).
 - 5 people, each with a random 3-machine workout plan (no duplicates).
 - People move smoothly at 1 unit/sec toward each machine's x–z coordinate,
   wait if the machine is occupied, then use it for 600 seconds.
 - No machine occupancy is triggered just by proximity; a machine is only
   marked as occupied if a person actually starts using it.

Every second (frame), we output:
  - One row per machine (columns: x, y, z, time, object, paired, role).
  - One row per person (columns: x, y, z, time, object, paired, role).
  (We do NOT include a "state" column for people.)

Usage:
  python3 gym_data.py
"""

import pandas as pd
import numpy as np
import random
import math

# ----------------------------
# Simulation Parameters
# ----------------------------
target_frames = 5000
frame_interval = 1        # 1 second per frame
arrival_threshold = 0.1   # threshold to consider arrival
usage_time = 600          # 600 sec usage time
speed = 1.0               # person moves at 1 unit/sec in x–z

# ----------------------------
# Machine Layout
# ----------------------------
push_machines = [
    "Machine Incline Bench 1",
    "Incline Bench 1",
    "Incline Bench 2"
]
pull_machines = [
    "Cable Pull Down 1",
    "Cable Pull Down 2",
    "Cable Pull Down 3",
    "Back Row Machine 1",
    "Back Row Machine 2",
    "Back Row Machine 3",
    "Bicep Curls Machine 1"
]
leg_machines = [
    "Leg Press 1",
    "Squat Rack 1",
    "Leg Extension 1",
    "Leg Curl 1",
    "Calf Raise 1"
]

machine_names = push_machines + pull_machines + leg_machines  # total 15

# Helper function to place machines in a small cluster
def place_cluster(machines, start_x, start_z, dx=2.0, dz=2.0):
    """
    Places the given machines in a cluster, row by row,
    starting at (start_x, start_z). Returns a dict {machine_name: (x, 0, z)}.
    """
    positions = {}
    row_size = 3
    current_x = start_x
    current_z = start_z
    for i, m in enumerate(machines):
        positions[m] = (current_x, 0.0, current_z)
        current_x += dx
        if (i + 1) % row_size == 0:
            current_x = start_x
            current_z += dz
    return positions

# We place the clusters in different areas of x–z.
push_positions = place_cluster(push_machines, start_x=10.0, start_z=0.0, dx=2.0, dz=2.0)
pull_positions = place_cluster(pull_machines, start_x=25.0, start_z=0.0, dx=2.0, dz=2.0)
leg_positions  = place_cluster(leg_machines,  start_x=50.0, start_z=0.0, dx=2.0, dz=2.0)

machine_positions = {}
machine_positions.update(push_positions)
machine_positions.update(pull_positions)
machine_positions.update(leg_positions)

# Occupancy tracker
occupied_until = {m: -1 for m in machine_names}

# ----------------------------
# People
# ----------------------------
num_people = 5
people = []
for i in range(num_people):
    plan = random.sample(machine_names, 3)
    x_init = np.random.uniform(0.0, 60.0)
    z_init = np.random.uniform(-2.0, 10.0)
    p = {
        "id": f"Person{i+1}",
        "workout_plan": plan,
        "current_target_index": 0,
        "state": "traveling",   # internal state, not output in CSV
        "position": [x_init, z_init],
        "usage_end_time": None
    }
    people.append(p)

# ----------------------------
# Main Simulation
# ----------------------------
rows = []

for frame_index in range(target_frames):
    current_time = frame_index * frame_interval

    # 1) Update each person's logic
    for person in people:
        # If they've used all machines, set them to exit
        if person["state"] == "finished":
            person["position"] = [0.0, 0.0]
            continue

        # If they still have a target
        if person["current_target_index"] < len(person["workout_plan"]):
            target_machine = person["workout_plan"][person["current_target_index"]]
            tx, _, tz = machine_positions[target_machine]
        else:
            target_machine = None

        if person["state"] == "traveling":
            px, pz = person["position"]
            dist = math.hypot(tx - px, tz - pz)
            if dist < arrival_threshold:
                person["position"] = [tx, tz]
                person["state"] = "waiting"
            else:
                dx = tx - px
                dz = tz - pz
                norm = math.hypot(dx, dz)
                step = min(speed, norm)
                person["position"] = [
                    px + (dx / norm) * step,
                    pz + (dz / norm) * step
                ]

        elif person["state"] == "waiting":
            # check if machine is free
            if current_time >= occupied_until[target_machine]:
                person["state"] = "using"
                person["usage_end_time"] = current_time + usage_time
                occupied_until[target_machine] = current_time + usage_time
            # else remain waiting

        elif person["state"] == "using":
            if current_time >= person["usage_end_time"]:
                person["current_target_index"] += 1
                if person["current_target_index"] < len(person["workout_plan"]):
                    person["state"] = "traveling"
                else:
                    person["state"] = "finished"
                    person["position"] = [0.0, 0.0]

    # 2) Machine rows
    machine_rows = []
    for m in machine_names:
        x_val, y_val, z_val = machine_positions[m]
        paired_val = 1 if current_time < occupied_until[m] else 0
        machine_rows.append({
            "x": x_val,
            "y": y_val,
            "z": z_val,
            "time": current_time,
            "object": m,
            "paired": paired_val,
            "role": "machine"
        })

    # 3) Person rows (no "state" in CSV)
    person_rows = []
    for person in people:
        px, pz = person["position"]
        person_rows.append({
            "x": px,
            "y": 0.0,
            "z": pz,
            "time": current_time,
            "object": person["id"],
            "paired": np.nan,
            "role": "person"
            # "state" not included in CSV
        })

    rows.extend(machine_rows)
    rows.extend(person_rows)

df = pd.DataFrame(rows)
df.to_csv("gym_data.csv", index=False)
print(f"gym_data.csv generated with {len(df)} rows, spanning {df['time'].max()} seconds.")