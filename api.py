from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from ML_Pipeline import train_model, find_best_plan  # Corrected import statement

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

@app.post("/schedule", response_model=ScheduleResponse)
def get_schedule(request: ScheduleRequest):
    best_ordering, best_schedule, best_total_wait = find_best_plan(
        model, encoder, df,
        workout_plan=request.workout_plan,
        start_time=request.start_time,
        travel_time=request.travel_time,
        usage_time=request.usage_time,
        resolution=request.resolution
    )
    if best_schedule is None:
        raise HTTPException(status_code=400, detail="No valid schedule found.")
    return {
        "best_ordering": list(best_ordering),
        "schedule": best_schedule,
        "total_wait": best_total_wait
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}