from pydantic import BaseModel

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float

class HealthCheckResponse(BaseModel):
    status: str

class ModelInfoResponse(BaseModel):
    project_name: str
    model_name: str
    model_alias: str