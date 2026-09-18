from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.decision_engine import make_decision


app = FastAPI(title="FreightWise AI")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "project": "FreightWise AI",
        "status": "Backend is running"
    }


@app.get("/decision")
def get_decision(
    cargo_quantity: float,
    cargo_type: str,
    origin_port: str,
    destination_port: str,
    coal_price: float,
    selling_price: float = 0,
    forecast_days: int = 7
):

    result = make_decision(
        cargo_quantity,
        coal_price,
        origin_port,
        destination_port,
        selling_price
    )

    result["cargo_type"] = cargo_type
    result["origin_port"] = origin_port
    result["destination_port"] = destination_port
    result["input_coal_price"] = coal_price
    result["selling_price"] = selling_price
    result["forecast_days"] = forecast_days

    return result


@app.get("/forecast")
def get_forecast(
    origin_port: str,
    destination_port: str,
    coal_price: float
):

    result = make_decision(
        1,
        coal_price,
        origin_port,
        destination_port,
        0
    )

    if "error" in result:
        return result

    return {
        "status": "success",
        "forecast": result.get("forecast"),
        "best_route": result.get("best_route"),
        "current_freight_rate": result.get("current_freight_rate"),
        "predicted_freight_rate": result.get("predicted_freight_rate"),
        "rate_difference": result.get("rate_difference"),
        "market_trend": result.get("market_trend"),
        "weather_risk": result.get("weather_risk"),
        "decision": result.get("decision")
    }