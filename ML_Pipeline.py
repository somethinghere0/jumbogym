
#!/usr/bin/env python3
"""
ML_Pipeline.py

This script:
  1. Loads gym_data.csv containing gym occupancy data.
  2. Filters to machine rows and trains a Random Forest classifier to predict occupancy.
  3. Evaluates all orderings for a given workout plan by simulating a schedule.
     For each machine in the plan, it finds the earliest free time (after arrival).
  4. Outputs the best ordering (schedule) with minimal total wait time.
  5. Prints progress info during evaluation.
     
Usage:
  python ML_Pipeline.py
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import itertools

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

def simulate_schedule(model, encoder, df, ordering, start_time, travel_time, usage_time, resolution=1):
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
    return schedule, total_wait

def find_best_plan(model, encoder, df, workout_plan, start_time, travel_time, usage_time, resolution=1):
    best_total_wait = float('inf')
    best_schedule = None
    best_ordering = None

    permutations = list(itertools.permutations(workout_plan))
    total_permutations = len(permutations)
    print(f"[INFO] Evaluating {total_permutations} orderings...")
    
    for i, ordering in enumerate(permutations, start=1):
        schedule, total_wait = simulate_schedule(model, encoder, df, list(ordering),
                                                 start_time, travel_time, usage_time, resolution)
        print(f"[DEBUG] Order {i}/{total_permutations}: {ordering}, Total Wait = {total_wait if total_wait is not None else 'N/A'} sec")
        if schedule is None:
            continue
        if total_wait < best_total_wait:
            best_total_wait = total_wait
            best_schedule = schedule
            best_ordering = ordering

    return best_ordering, best_schedule, best_total_wait

def main():
    csv_file = "gym_data.csv"
    model, encoder, df = train_model(csv_file)
    
    # For demonstration, let's pick one machine from each group as a plan:
    user_workout_plan = ["Incline Bench 1", "Cable Pull Down 1", "Leg Press 1"]
    
    start_time = 0            # Start at 0 sec
    travel_time = 60          # 60 sec travel
    usage_time = 10 * 60      # 600 sec usage
    resolution = 1            # 1 sec resolution

    print(f"[INFO] Evaluating all orderings for workout plan: {user_workout_plan}\n")
    best_ordering, best_schedule, best_total_wait = find_best_plan(model, encoder, df,
                                                                   user_workout_plan,
                                                                   start_time,
                                                                   travel_time,
                                                                   usage_time,
                                                                   resolution)
    if best_schedule is None:
        print("[ERROR] Could not determine a valid schedule.")
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