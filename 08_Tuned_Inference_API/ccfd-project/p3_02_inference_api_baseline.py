"""
Credit Card Fraud Detection - Inference API (P3_01 Baseline)

Uses sklearn Pipeline - preprocessing handled automatically!
No manual scaling/encoding needed. Much simpler!

Endpoints:
- POST /inference - Predict if transaction is fraud
- GET /health - Health check
- GET /ui - Web interface
- GET / - API info
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator
import pandas as pd
from typing import Literal
import uvicorn
from pathlib import Path

from utils.model_utils import load_pipeline, get_available_pipelines

# ============================================================================
# Configuration
# ============================================================================
MODEL_NAME = 'p3_01_baseline'
MODEL_VERSION = "1.0.0"
API_PORT = 8000

# Consistent threshold handling across all inference APIs
# Baseline uses 0.5 as default
DEFAULT_THRESHOLD = 0.5

# ============================================================================
# Global Pipeline
# ============================================================================
PIPELINE = None
METADATA = None


def load_model_artifacts():
    """Load trained model for inference."""
    global PIPELINE, METADATA
    
    try:
        PIPELINE, METADATA = load_pipeline(MODEL_NAME)
        print(f"✅ Model loaded successfully")
        return True
        
    except FileNotFoundError as e:
        print(f"❌ Model not found: {e}")
        print(f"   Available: {get_available_pipelines()}")
        return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


# ============================================================================
# Lifespan Context Manager
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown events."""
    print(f"🚀 Starting Fraud Detection API (P3_01 Baseline)...")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Port: {API_PORT}")
    load_model_artifacts()
    yield
    print("👋 Shutting down API...")


# ============================================================================
# FastAPI App
# ============================================================================
app = FastAPI(
    title="Credit Card Fraud Detection API (P3_01 Baseline)",
    description="Real-time fraud detection using sklearn Pipeline",
    version=MODEL_VERSION,
    lifespan=lifespan
)


# ============================================================================
# Pydantic Models
# ============================================================================
class TransactionRequest(BaseModel):
    """Request schema for fraud prediction - BASE features only"""
    
    transaction_amount: float = Field(
        ..., gt=0, description="Transaction amount in USD", examples=[1500.00]
    )
    transaction_hour: int = Field(
        ..., ge=0, le=23, description="Hour of transaction (0-23)", examples=[2]
    )
    days_since_last_txn: int = Field(
        ..., ge=0, description="Days since customer's last transaction", examples=[0]
    )
    avg_transaction_amount: float = Field(
        ..., gt=0, description="Customer's average transaction amount", examples=[120.00]
    )
    merchant_category: Literal[
        'grocery', 'gas_station', 'restaurant', 'retail',
        'online_shopping', 'pharmacy', 'entertainment', 'utilities',
        'travel', 'electronics', 'furniture', 'jewelry'
    ] = Field(..., description="Type of merchant", examples=["electronics"])
    card_present: Literal['yes', 'no'] = Field(
        ..., description="Was card physically present?", examples=["no"]
    )
    international: Literal['yes', 'no'] = Field(
        ..., description="Is this an international transaction?", examples=["yes"]
    )
    transaction_count_24h: int = Field(
        ..., ge=1, description="Number of transactions in last 24 hours", examples=[7]
    )
    
    @field_validator('transaction_amount', 'avg_transaction_amount')
    @classmethod
    def round_to_2_decimals(cls, v):
        return round(v, 2)


class PredictionResponse(BaseModel):
    """Response schema for fraud prediction"""
    model_config = {'protected_namespaces': ()}
    
    prediction: Literal['FRAUD', 'LEGITIMATE']
    probability: float = Field(..., description="Probability of fraud (0-1)")
    threshold: float = Field(..., description="Decision threshold used")
    risk_score: Literal['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    confidence: float = Field(..., description="Model confidence (0-1)")
    model_name: str = Field(..., description="Model used for prediction")
    message: str


class HealthResponse(BaseModel):
    """Health check response"""
    model_config = {'protected_namespaces': ()}
    
    status: str
    model_loaded: bool
    model_name: str
    model_type: str
    version: str


# ============================================================================
# Helper Functions
# ============================================================================
def prepare_input(data: TransactionRequest) -> pd.DataFrame:
    """
    Convert request to DataFrame for Pipeline.
    
    No manual encoding/scaling needed!
    Pipeline handles everything internally.
    """
    # Just convert to DataFrame - Pipeline does the rest!
    input_dict = data.model_dump()
    df = pd.DataFrame([input_dict])
    return df


def calculate_risk_score(probability: float) -> str:
    """Calculate risk score based on fraud probability"""
    if probability < 0.30:
        return "LOW"
    elif probability < 0.60:
        return "MEDIUM"
    elif probability < 0.90:
        return "HIGH"
    else:
        return "CRITICAL"


# ============================================================================
# API Endpoints
# ============================================================================
@app.get("/", tags=["Info"])
async def root():
    """API information"""
    return {
        "name": "Credit Card Fraud Detection API (P3_01 Baseline)",
        "version": MODEL_VERSION,
        "status": "running",
        "model": MODEL_NAME,
        "model_loaded": PIPELINE is not None,
        "approach": "sklearn Pipeline",
        "endpoints": {
            "POST /inference": "Predict fraud for a transaction",
            "GET /health": "Health check",
            "GET /ui": "Web interface",
            "GET /docs": "API documentation (Swagger UI)"
        }
    }


@app.get("/ui", response_class=HTMLResponse, tags=["Web Interface"])
async def web_interface():
    """Web interface for testing"""
    template_path = Path(__file__).parent / "templates" / "index.html"
    
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Error: Template not found</h1>",
            status_code=500
        )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check if API and model are healthy"""
    return HealthResponse(
        status="healthy" if PIPELINE is not None else "unhealthy",
        model_loaded=PIPELINE is not None,
        model_name=METADATA.get('model_name', 'unknown') if METADATA else "not loaded",
        model_type=METADATA.get('model_type', 'unknown') if METADATA else "unknown",
        version=MODEL_VERSION
    )


@app.post("/inference", response_model=PredictionResponse, tags=["Prediction"])
async def predict_fraud(transaction: TransactionRequest):
    """
    Predict if a credit card transaction is fraudulent.
    
    Pipeline handles all preprocessing automatically!
    """
    if PIPELINE is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please run training first."
        )
    
    try:
        # Prepare input as DataFrame
        df = prepare_input(transaction)
        
        # Pipeline handles preprocessing + prediction
        # Use predict_proba + threshold pattern for consistency
        probability = PIPELINE.predict_proba(df)[0]
        fraud_prob = float(probability[1])
        prediction = 1 if fraud_prob >= DEFAULT_THRESHOLD else 0
        
        prediction_label = "FRAUD" if prediction == 1 else "LEGITIMATE"
        risk_score = calculate_risk_score(fraud_prob)
        confidence = min(float(abs(fraud_prob - DEFAULT_THRESHOLD) * 2), 1.0)
        
        if prediction_label == "FRAUD":
            message = f"⚠️ HIGH RISK: {fraud_prob:.1%} probability of fraud. Recommend blocking."
        else:
            message = f"✅ LOW RISK: {(1-fraud_prob):.1%} confidence legitimate. Safe to approve."
        
        return PredictionResponse(
            prediction=prediction_label,
            probability=fraud_prob,
            threshold=DEFAULT_THRESHOLD,
            risk_score=risk_score,
            confidence=confidence,
            model_name=MODEL_NAME,
            message=message
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ============================================================================
# Run Application
# ============================================================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=API_PORT, log_level="info")
