#!/usr/bin/env python3
"""
ML_Pipeline.py

This script:
  1. Loads gym_data.csv containing gym occupancy data.
  2. Filters to machine rows and trains a Random Forest classifier to predict occupancy,
     but only if the CSV file has been updated.
  3. Prompts the user for a muscle group ("push", "pull", or "legs") and a total workout duration (in minutes).
  4. Selects the machines corresponding to the chosen group (ensuring no duplicate base machines).
  5. Simulates schedules using a fixed per‑machine usage time (7 minutes) and a fixed travel time.
     It evaluates orderings of the selected machine group using branch and bound search.
  6. Outputs the best ordering (schedule) with minimal total wait time if at least two machines can be fit;
     otherwise, it reports that the gym cannot accommodate a workout within that duration.
  7. Prints the order to do the workout along with the time you arrive, when the machine is free,
     wait time, usage start, and usage finish.

Usage:
  python ML_Pipeline.py
"""

import concurrent.futures
import os
import threading
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

# Global cache for model and CSV timestamp.
cached_model = None
cached_encoder = None
cached_df = None
cached_csv_mtime = None

# Cache for workout plans
# Key: (muscle_group, duration_min)
# Value: (best_ordering, best_schedule, best_total_wait)
workout_cache: Dict[Tuple[str, int], Tuple[List[str], List[Dict[str, Any]], float]] = {}

def load_data(csv_file):
    df = pd.read_csv(csv_file)
    return df

def preprocess_data(df):
    numeric_features = df[['x', 'y', 'z', 'time']].values
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    objects_encoded = encoder.fit_transform(df[['object']])
    X = np.hstack([numeric_features, objects_encoded])
    y = df['paired'].values
    return X, y, encoder

def train_model(csv_file):
    """
    Train the model only if the CSV has been updated.
    Uses caching to avoid retraining if the file is unchanged.
    """
    global cached_model, cached_encoder, cached_df, cached_csv_mtime

    mtime = os.path.getmtime(csv_file)
    if cached_model is not None and cached_csv_mtime == mtime:
        print(f"[INFO] Using cached model trained on {len(cached_df)} samples.")
        return cached_model, cached_encoder, cached_df

    df = load_data(csv_file)
    # Filter to machine rows only
    df = df[df['role'] == "machine"]
    df = df.dropna(subset=['paired'])
    df['paired'] = df['paired'].astype(int)

    X, y, encoder = preprocess_data(df)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    print(f"[INFO] Model trained on {len(df)} samples.")

    # Update the cache.
    cached_model = model
    cached_encoder = encoder
    cached_df = df
    cached_csv_mtime = mtime

    return model, encoder, df

def build_feature_vector(machine_info, t, encoder):
    import pandas as pd
    base_df = pd.DataFrame([[machine_info['x'], machine_info['y'], machine_info['z'], t]],
                           columns=["x", "y", "z", "time"])
    obj_df = pd.DataFrame([[machine_info['object']]], columns=["object"])
    obj_encoded = encoder.transform(obj_df)
    X = np.hstack([base_df.values, obj_encoded])
    return X

def next_free_time(model, encoder, machine_info, arrival_time, resolution=1, max_search=3600):
    t = arrival_time
    end_time = arrival_time + max_search
    while t <= end_time:
        feature_vector = build_feature_vector(machine_info, t, encoder)
        prediction = model.predict(feature_vector)
        if prediction[0] == 0:
            return t
        t += resolution
    return None

def simulate_schedule(model, encoder, df, ordering, start_time, travel_time, usage_time, total_duration, resolution=1):
    schedule = []
    current_time = start_time
    total_wait = 0
    for machine in ordering:
        machine_rows = df[df['object'] == machine]
        if machine_rows.empty:
            print(f"[WARN] No data found for machine {machine}.")
            return None, None
        row = machine_rows.iloc[0]
        machine_info = {"x": row["x"], "y": row["y"], "z": row["z"], "object": machine}
        arrival_time = current_time
        free_time = next_free_time(model, encoder, machine_info, arrival_time, resolution)
        if free_time is None:
            print(f"[WARN] Could not find free time for machine {machine} starting at {arrival_time}.")
            return None, None
        wait_time = free_time - arrival_time
        # If the free time plus usage would exceed the total workout duration, break out.
        if free_time + usage_time > total_duration:
            break
        total_wait += wait_time
        usage_start = free_time
        usage_finish = usage_start + usage_time
        schedule.append({
            "machine": machine,
            "arrival_time": arrival_time,
            "free_time": free_time,
            "wait_time": wait_time,
            "usage_start": usage_start,
            "usage_finish": usage_finish
        })
        current_time = usage_finish + travel_time
        if current_time > total_duration:
            break
    # Require at least 2 machines in the schedule.
    if len(schedule) < 2:
        return None, None
    return schedule, total_wait

def get_base_name(machine):
    """
    Returns the base name of a machine by stripping a trailing digit if it exists.
    For example, "Cable Pull Down 1" -> "Cable Pull Down"
    """
    tokens = machine.split()
    if tokens[-1].isdigit():
        return " ".join(tokens[:-1])
    return machine

def unique_machines(machine_list):
    """
    Filters the provided list so that only one machine per base name is kept.
    """
    seen = set()
    unique = []
    for m in machine_list:
        base = get_base_name(m)
        if base not in seen:
            unique.append(m)
            seen.add(base)
    return unique

def find_best_plan_branch_and_bound(model, encoder, df, machines, start_time, travel_time, usage_time, total_duration, resolution=1):
    # Shared best solution variables and a lock for thread safety.
    best_total_wait = float('inf')
    best_schedule = None
    best_ordering = None
    lock = threading.Lock()

    def recursive_search(current_ordering, remaining_machines, current_time, accumulated_wait):
        nonlocal best_total_wait, best_schedule, best_ordering

        # If current ordering has at least two machines, simulate the schedule.
        if len(current_ordering) >= 2:
            schedule, total_wait = simulate_schedule(model, encoder, df, current_ordering, start_time, travel_time, usage_time, total_duration, resolution)
            if schedule is not None:
                with lock:
                    if total_wait < best_total_wait:
                        best_total_wait = total_wait
                        best_schedule = schedule
                        best_ordering = tuple(current_ordering)

        if not remaining_machines:
            return

        # Compute base names already used in the current ordering.
        used_bases = {get_base_name(m) for m in current_ordering}
        for machine in remaining_machines:
            # Skip machines whose base name is already selected.
            if get_base_name(machine) in used_bases:
                continue

            machine_rows = df[df['object'] == machine]
            if machine_rows.empty:
                continue
            row = machine_rows.iloc[0]
            machine_info = {"x": row["x"], "y": row["y"], "z": row["z"], "object": machine}

            arrival_time = current_time
            free_time = next_free_time(model, encoder, machine_info, arrival_time, resolution)
            if free_time is None:
                continue
            if free_time + usage_time > total_duration:
                continue
            wait_time = free_time - arrival_time
            new_accumulated_wait = accumulated_wait + wait_time

            with lock:
                if new_accumulated_wait >= best_total_wait:
                    continue

            new_current_time = free_time + usage_time + travel_time
            new_ordering = current_ordering + [machine]
            new_remaining = [m for m in remaining_machines if m != machine]
            recursive_search(new_ordering, new_remaining, new_current_time, new_accumulated_wait)

    # Launch the first level of recursion concurrently.
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for machine in machines:
            initial_ordering = [machine]
            remaining = [m for m in machines if m != machine]
            futures.append(executor.submit(recursive_search, initial_ordering, remaining, start_time, 0))
        concurrent.futures.wait(futures)

    return best_ordering, best_schedule, best_total_wait

def find_best_plan_simple(muscle_group: str, duration_min: int):
    """
    A simplified interface to get the best gym workout plan.

    Parameters:
      muscle_group (str): "push", "pull", or "legs"
      duration_min (int): Total workout duration in minutes

    Returns:
      tuple: (best_ordering, best_schedule, best_total_wait)
    """
    # Check cache first
    cache_key = (muscle_group.lower(), duration_min)
    if cache_key in workout_cache:
        print(f"[INFO] Using cached workout plan for {muscle_group} with {duration_min} minutes duration")
        return workout_cache[cache_key]

    total_duration = duration_min * 60  # convert minutes to seconds
    usage_time_per_machine = 420       # 7 minutes in seconds
    travel_time = 60                   # 60 seconds travel time between machines
    resolution = 1

    # Define machine groups.
    push_machines = [
        "Machine Incline Bench 1",
        "Incline Bench 1",
        "Incline Bench 2",
        "Chest Press Machine 1"
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

    # Choose the list and filter to unique machines based on base name.
    if muscle_group.lower() == "push":
        user_workout_plan = unique_machines(push_machines)
    elif muscle_group.lower() == "pull":
        user_workout_plan = unique_machines(pull_machines)
    elif muscle_group.lower() == "legs":
        user_workout_plan = unique_machines(leg_machines)
    else:
        print("Invalid muscle group input. Defaulting to 'push'.")
        user_workout_plan = unique_machines(push_machines)

    csv_file = "gym_data.csv"
    model, encoder, df = train_model(csv_file)
    start_time = 0

    best_ordering, best_schedule, best_total_wait = find_best_plan_branch_and_bound(
        model, encoder, df,
        user_workout_plan,
        start_time,
        travel_time,
        usage_time_per_machine,
        total_duration,
        resolution
    )

    if best_schedule is None:
        print("[ERROR] Could not determine a valid schedule. The gym cannot accommodate a workout with at least 2 machines within the given duration.")
    else:
        # Cache the result
        workout_cache[cache_key] = (best_ordering, best_schedule, best_total_wait)

    return best_ordering, best_schedule, best_total_wait

def main():
    # Prompt user for muscle group and total workout duration.
    group_input = input("Enter muscle group (push, pull, legs): ").strip().lower()
    duration_input = input("Enter total workout duration (in minutes, e.g., 30, 45, 60): ").strip()
    try:
        total_duration_min = int(duration_input)
    except ValueError:
        print("Invalid duration input. Defaulting to 30 minutes.")
        total_duration_min = 30

    print(f"[INFO] Evaluating all orderings for muscle group '{group_input}'")
    print(f"[INFO] Total workout duration: {total_duration_min} minutes.\n")

    csv_file = "gym_data.csv"
    model, encoder, df = train_model(csv_file)

    # Define machine groups (same as in find_best_plan_simple) and filter for unique machines.
    push_machines = unique_machines([
        "Machine Incline Bench 1",
        "Incline Bench 1",
        "Incline Bench 2",
        "Chest Press Machine 1"
    ])
    pull_machines = unique_machines([
        "Cable Pull Down 1",
        "Cable Pull Down 2",
        "Cable Pull Down 3",
        "Back Row Machine 1",
        "Back Row Machine 2",
        "Back Row Machine 3",
        "Bicep Curls Machine 1"
    ])
    leg_machines = unique_machines([
        "Leg Press 1",
        "Squat Rack 1",
        "Leg Extension 1",
        "Leg Curl 1",
        "Calf Raise 1"
    ])

    if group_input == "push":
        user_workout_plan = push_machines
    elif group_input == "pull":
        user_workout_plan = pull_machines
    elif group_input == "legs":
        user_workout_plan = leg_machines
    else:
        print("Invalid group input. Defaulting to 'push'.")
        user_workout_plan = push_machines

    usage_time_per_machine = 420  # 7 minutes in seconds
    travel_time = 60              # 60 seconds
    total_duration = total_duration_min * 60  # convert minutes to seconds
    resolution = 1

    print(f"[INFO] Machines: {user_workout_plan}")
    print(f"[INFO] Each machine usage is fixed at {usage_time_per_machine/60:.1f} minutes, with {travel_time} sec travel time between machines.\n")

    start_time = 0
    best_ordering, best_schedule, best_total_wait = find_best_plan_branch_and_bound(
        model, encoder, df,
        user_workout_plan,
        start_time,
        travel_time,
        usage_time_per_machine,
        total_duration,
        resolution
    )
    if best_schedule is None:
        print("[ERROR] Could not determine a valid schedule. The gym cannot accommodate a workout with at least 2 machines within the given duration.")
        return

    # Print the detailed workout plan.
    print("\nWorkout Plan Schedule:")
    for item in best_schedule:
        print(f"Machine: {item['machine']}")
        print(f"  Arrival Time: {item['arrival_time']:.1f} sec")
        print(f"  Free Time: {item['free_time']:.1f} sec")
        print(f"  Wait Time: {item['wait_time']:.1f} sec")
        print(f"  Usage Start: {item['usage_start']:.1f} sec")
        print(f"  Usage Finish: {item['usage_finish']:.1f} sec")
    overall_end = best_schedule[-1]['usage_finish']
    print(f"\nWorkout End Time: {overall_end:.1f} sec")
    print(f"Total Wait Time: {best_total_wait:.1f} sec")

if __name__ == "__main__":
    main()
