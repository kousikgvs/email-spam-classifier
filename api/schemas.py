from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The text message to classify")


class PredictResponse(BaseModel):
    message: str
    prediction: str  # "Spam" or "Ham"
