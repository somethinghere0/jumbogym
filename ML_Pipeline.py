#!/usr/bin/env python3
"""
ML_Pipeline.py

This script:
  1. Loads gym_data.csv containing gym occupancy data.
  2. Filters to machine rows and trains a Random Forest classifier to predict occupancy.
  3. Prompts the user for a muscle group ("push", "pull", or "legs") and a total workout duration (in minutes).
  4. Selects the machines corresponding to the chosen group.
  5. Simulates schedules using a fixed per‑machine usage time (7 minutes) and a fixed travel time.
     It evaluates orderings of the selected machine group using branch and bound search,
     while ensuring no duplicate base machines are used in a workout.
  6. Outputs the best ordering (schedule) with minimal total wait time if at least two machines can be fit;
     otherwise, it reports that the gym cannot accommodate a workout within that duration.
  7. Prints progress info during evaluation.

Usage:
  python ML_Pipeline.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder


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
    df = load_data(csv_file)
    # Filter to machine rows only
    df = df[df['role'] == "machine"]
    df = df.dropna(subset=['paired'])
    df['paired'] = df['paired'].astype(int)

    X, y, encoder = preprocess_data(df)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    print(f"[INFO] Model trained on {len(df)} samples.")
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
    # We require that at least 2 machines are scheduled.
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

def find_best_plan_branch_and_bound(model, encoder, df, machines, start_time, travel_time, usage_time, total_duration, resolution=1):
    best_total_wait = float('inf')
    best_schedule = None
    best_ordering = None

    def recursive_search(current_ordering, remaining_machines, current_time, accumulated_wait):
        nonlocal best_total_wait, best_schedule, best_ordering

        # If current ordering has at least two machines, simulate the schedule.
        if len(current_ordering) >= 2:
            schedule, total_wait = simulate_schedule(model, encoder, df, current_ordering, start_time, travel_time, usage_time, total_duration, resolution)
            if schedule is not None and total_wait < best_total_wait:
                best_total_wait = total_wait
                best_schedule = schedule
                best_ordering = tuple(current_ordering)

        if not remaining_machines:
            return

        # Compute base names already used in the current ordering.
        used_bases = {get_base_name(m) for m in current_ordering}
        for machine in remaining_machines:
            # Skip machines that are of the same base type already selected.
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

            # Prune branch if accumulated wait is already worse than best found.
            if new_accumulated_wait >= best_total_wait:
                continue

            new_current_time = free_time + usage_time + travel_time
            new_ordering = current_ordering + [machine]
            new_remaining = [m for m in remaining_machines if m != machine]
            recursive_search(new_ordering, new_remaining, new_current_time, new_accumulated_wait)

    recursive_search([], machines, start_time, 0)
    return best_ordering, best_schedule, best_total_wait

def find_best_plan_simple(muscle_group, duration_min):

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

    if muscle_group.lower() == "push":
        user_workout_plan = push_machines
    elif muscle_group.lower() == "pull":
        user_workout_plan = pull_machines
    elif muscle_group.lower() == "legs":
        user_workout_plan = leg_machines
    else:
        print("Invalid muscle group input. Defaulting to 'push'.")
        user_workout_plan = push_machines

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
    total_duration = total_duration_min * 60  # total workout duration in seconds

    # Fixed per-machine usage time and travel time.
    usage_time_per_machine = 420  # 7 minutes in seconds (modified from 600 sec)
    travel_time = 60              # 60 seconds
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

    if group_input == "push":
        user_workout_plan = push_machines
    elif group_input == "pull":
        user_workout_plan = pull_machines
    elif group_input == "legs":
        user_workout_plan = leg_machines
    else:
        print("Invalid group input. Defaulting to 'push'.")
        user_workout_plan = push_machines

    print(f"[INFO] Evaluating all orderings for muscle group '{group_input}' with machines: {user_workout_plan}")
    print(f"[INFO] Total workout duration: {total_duration_min} minutes.")
    print(f"[INFO] Each machine usage is fixed at {usage_time_per_machine/60:.1f} minutes, with {travel_time} sec travel time between machines.\n")

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
        return

    print(f"\nBest Ordering: {best_ordering}")
    print("Workout Plan Schedule:")
    for item in best_schedule:
        print(f"Machine: {item['machine']}")
        print(f"  Arrival Time: {item['arrival_time']:.1f} sec")
        print(f"  Free Time: {item['free_time']:.1f} sec")
        print(f"  Wait Time: {item['wait_time']:.1f} sec")
        print(f"  Usage Start: {item['usage_start']:.1f} sec")
        print(f"  Usage Finish: {item['usage_finish']:.1f} sec")
    print(f"\nTotal Wait Time: {best_total_wait:.1f} sec")

if __name__ == "__main__":
    main()
