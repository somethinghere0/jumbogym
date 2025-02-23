from enum import Enum
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ML_Pipeline import find_best_plan_simple  # Corrected import statement
from ML_Pipeline import train_model

app = FastAPI()

# Load and train the model once at startup so subsequent API calls are fast.
CSV_FILE = "gym_data.csv"
model, encoder, df = train_model(CSV_FILE)

# Request model for the schedule endpoint.
class ScheduleRequest(BaseModel):
    workout_plan: List[str]            # e.g. ["Incline Bench 1", "Cable Pull Down 1", "Leg Press 1"]
    start_time: Optional[int] = 0      # Start time in seconds
    travel_time: Optional[int] = 60    # Travel time in seconds between machines
    usage_time: Optional[int] = 600    # Usage time (in seconds) at each machine
    resolution: Optional[int] = 1      # Time resolution for simulation

# Response model for clarity (optional but recommended).
class ScheduleResponse(BaseModel):
    best_ordering: List[str]
    schedule: List[dict]
    total_wait: float
    total_duration: float  # Total duration of the workout in seconds

class MuscleGroup(str, Enum):
    PUSH = "push"
    PULL = "pull"
    LEGS = "legs"

class WorkoutRequest(BaseModel):
    muscle_group: MuscleGroup
    workout_duration: Literal[30, 45, 60]


@app.post("/createWorkout")
def create_workout(request: WorkoutRequest):
    # Call the ML pipeline to get the best workout plan
    best_ordering, best_schedule, best_total_wait = find_best_plan_simple(
        request.muscle_group.value,
        request.workout_duration
    )

    # Handle case where no valid schedule was found
    if best_schedule is None:
        raise HTTPException(
            status_code=400,
            detail="Could not determine a valid schedule. The gym cannot accommodate a workout with at least 2 machines within the given duration."
        )

    # Format the response according to ScheduleResponse model
    total_duration = max(machine["usage_finish"] for machine in best_schedule)
    response = ScheduleResponse(
        best_ordering=list(best_ordering),
        schedule=best_schedule,
        total_wait=best_total_wait,
        total_duration=total_duration
    )

    return response

@app.get("/health")
def health_check():
    return {"status": "ok"}
