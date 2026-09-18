import math
import pandas as pd

from backend.data_loader import (
    load_freight_data,
    load_vessel_data,
    load_route_data,
    load_weather_data
)

from backend.forecast_engine import calculate_forecast


def make_decision(
    cargo_quantity,
    coal_price,
    origin_port,
    destination_port,
    selling_price=0
):

    # --------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------

    freight = load_freight_data()
    vessels = load_vessel_data()
    routes = load_route_data()
    weather = load_weather_data()

    cargo_quantity = float(cargo_quantity)
    coal_price = float(coal_price)
    selling_price = float(selling_price or 0)

    if cargo_quantity <= 0:
        return {
            "error": "Cargo quantity must be greater than 0"
        }

    if coal_price < 0:
        return {
            "error": "Purchase price cannot be negative"
        }

    if freight.empty:
        return {
            "error": "Freight data is empty"
        }

    if routes.empty:
        return {
            "error": "Route data is empty"
        }

    if vessels.empty:
        return {
            "error": "Vessel data is empty"
        }

    # --------------------------------------------------
    # PREPARE FREIGHT DATA
    # --------------------------------------------------

    freight = freight.copy()

    freight["date"] = pd.to_datetime(
        freight["date"],
        errors="coerce"
    )

    freight = freight.dropna(
        subset=["date"]
    )

    # Use user's latest coal price
    freight["coal_price"] = coal_price

    # --------------------------------------------------
    # ML FORECAST
    # --------------------------------------------------

    forecast = calculate_forecast(
        freight,
        routes,
        weather
    )

    if forecast is None:
        forecast = pd.DataFrame()

    # --------------------------------------------------
    # FILTER USER ROUTE
    # --------------------------------------------------

    requested_route = (
        str(origin_port).strip()
        + "-"
        + str(destination_port).strip()
    )

    route_matches = routes[
        routes["route"]
        .astype(str)
        .str.lower()
        .str.strip()
        ==
        requested_route.lower()
    ].copy()

    if route_matches.empty:

        route_matches = routes[
            routes["route"]
            .astype(str)
            .str.lower()
            .str.contains(
                str(origin_port).lower().strip(),
                na=False
            )
            &
            routes["route"]
            .astype(str)
            .str.lower()
            .str.contains(
                str(destination_port).lower().strip(),
                na=False
            )
        ].copy()

    # --------------------------------------------------
    # LATEST FREIGHT
    # --------------------------------------------------

    freight_sorted = freight.sort_values(
        "date"
    )

    latest_freight = (
        freight_sorted
        .groupby("route")
        .tail(1)
        .copy()
    )

    # --------------------------------------------------
    # LATEST WEATHER
    # --------------------------------------------------

    weather = weather.copy()

    weather["date"] = pd.to_datetime(
        weather["date"],
        errors="coerce"
    )

    weather = weather.dropna(
        subset=["date"]
    )

    latest_weather = (
        weather
        .sort_values("date")
        .groupby("route")
        .tail(1)
        .copy()
    )

    # --------------------------------------------------
    # ROUTE ANALYSIS
    # --------------------------------------------------

    route_analysis = []

    routes_to_check = (
        route_matches
        if not route_matches.empty
        else routes.copy()
    )

    for _, route in routes_to_check.iterrows():

        route_name = route["route"]

        route_freight = latest_freight[
            latest_freight["route"]
            == route_name
        ]

        if route_freight.empty:
            continue

        freight_row = route_freight.iloc[0]

        current_rate = float(
            freight_row["freight_rate"]
        )

        route_weather = latest_weather[
            latest_weather["route"]
            == route_name
        ]

        if route_weather.empty:

            weather_risk = "Medium"
            temperature_c = 0
            wind_speed_knots = 0
            wave_height_m = 0
            rainfall_mm = 0

        else:

            weather_row = route_weather.iloc[0]

            weather_risk = str(
                weather_row.get(
                    "weather_risk",
                    "Medium"
                )
            )

            temperature_c = float(
                weather_row.get(
                    "temperature_c",
                    0
                )
            )

            wind_speed_knots = float(
                weather_row.get(
                    "wind_speed_knots",
                    0
                )
            )

            wave_height_m = float(
                weather_row.get(
                    "wave_height_m",
                    0
                )
            )

            rainfall_mm = float(
                weather_row.get(
                    "rainfall_mm",
                    0
                )
            )

        # --------------------------------------------------
        # WEATHER PENALTY
        # --------------------------------------------------

        weather_penalty_map = {
            "Low": 0,
            "Medium": 8,
            "High": 18
        }

        weather_penalty = (
            weather_penalty_map.get(
                weather_risk,
                8
            )
        )

        # --------------------------------------------------
        # PORT STATUS
        # --------------------------------------------------

        port_status = str(
            route.get(
                "status",
                "Operational"
            )
        )

        if port_status.lower() in [
            "closed",
            "suspended"
        ]:

            port_penalty = 30

        else:

            port_penalty = 0

        # --------------------------------------------------
        # ROUTE SCORE
        # --------------------------------------------------

        base_rate = float(
            route.get(
                "base_freight_rate",
                current_rate
            )
        )

        voyage_days = float(
            route.get(
                "avg_voyage_days",
                0
            )
        )

        distance_nm = float(
            route.get(
                "distance_nm",
                route.get(
                    "distance",
                    0
                )
            )
        )

        rate_difference = (
            current_rate
            - base_rate
        )

        route_score = (
            100
            - abs(rate_difference) * 3
            - voyage_days * 1.5
            - weather_penalty
            - port_penalty
        )

        route_score = max(
            0,
            min(
                100,
                route_score
            )
        )

        if port_status.lower() in [
            "closed",
            "suspended"
        ]:

            route_suitability = "Not Suitable"

        elif weather_risk == "High":

            route_suitability = "High Risk"

        elif route_score >= 70:

            route_suitability = "Highly Suitable"

        elif route_score >= 45:

            route_suitability = "Suitable"

        else:

            route_suitability = "Less Suitable"

        route_analysis.append(
            {
                "route": route_name,
                "current_freight_rate": current_rate,
                "base_freight_rate": base_rate,
                "distance_nm": distance_nm,
                "avg_voyage_days": voyage_days,
                "weather_risk": weather_risk,
                "temperature_c": temperature_c,
                "wind_speed_knots": wind_speed_knots,
                "wave_height_m": wave_height_m,
                "rainfall_mm": rainfall_mm,
                "port_status": port_status,
                "route_score": round(
                    route_score,
                    2
                ),
                "route_suitability":
                    route_suitability
            }
        )

    if not route_analysis:

        return {
            "error":
                "No suitable route found for the selected origin and destination"
        }

    # --------------------------------------------------
    # BEST ROUTE
    # --------------------------------------------------

    route_analysis.sort(
        key=lambda x: x["route_score"],
        reverse=True
    )

    best_route = route_analysis[0]

    selected_route_name = best_route[
        "route"
    ]

    # --------------------------------------------------
    # FORECAST FOR SELECTED ROUTE
    # --------------------------------------------------

    route_forecast = pd.DataFrame()

    if not forecast.empty:

        route_forecast = forecast[
            forecast["route"]
            .astype(str)
            ==
            str(selected_route_name)
        ].copy()

    # --------------------------------------------------
    # CURRENT FREIGHT
    # --------------------------------------------------

    current_freight_rate = float(
        best_route[
            "current_freight_rate"
        ]
    )

    # --------------------------------------------------
    # PREDICTED FREIGHT
    # --------------------------------------------------

    if not route_forecast.empty:

        predicted_freight_rate = float(
            route_forecast.iloc[0].get(
                "predicted_freight_rate",
                current_freight_rate
            )
        )

    else:

        predicted_freight_rate = (
            current_freight_rate
        )

    rate_difference = (
        predicted_freight_rate
        - current_freight_rate
    )

    if rate_difference > 0.5:

        market_trend = "Rising"

    elif rate_difference < -0.5:

        market_trend = "Falling"

    else:

        market_trend = "Stable"

    # --------------------------------------------------
    # WEATHER RISK SCORE
    # --------------------------------------------------

    weather_score_map = {
        "Low": 0,
        "Medium": 1,
        "High": 2
    }

    weather_score = weather_score_map.get(
        best_route["weather_risk"],
        1
    )

    wind_score = 0

    if best_route["wind_speed_knots"] >= 30:
        wind_score = 2

    elif best_route["wind_speed_knots"] >= 20:
        wind_score = 1

    wave_score = 0

    if best_route["wave_height_m"] >= 4:
        wave_score = 2

    elif best_route["wave_height_m"] >= 2.5:
        wave_score = 1

    rain_score = 0

    if best_route["rainfall_mm"] >= 50:
        rain_score = 2

    elif best_route["rainfall_mm"] >= 20:
        rain_score = 1

    port_score = 2 if (
        str(
            best_route["port_status"]
        ).lower()
        in [
            "closed",
            "suspended"
        ]
    ) else 0

    risk_points = (
        weather_score
        + wind_score
        + wave_score
        + rain_score
        + port_score
    )

    if risk_points >= 6:

        overall_risk = "High"

    elif risk_points >= 3:

        overall_risk = "Medium"

    else:

        overall_risk = "Low"

    # --------------------------------------------------
    # VESSEL SELECTION
    # --------------------------------------------------

    available_vessels = vessels[
        vessels[
            "status"
        ]
        .astype(str)
        .str.lower()
        ==
        "available"
    ].copy()

    if available_vessels.empty:

        return {
            "error":
                "No available vessels found"
        }

    available_vessels = (
        available_vessels
        .sort_values(
            "capacity_mt",
            ascending=False
        )
    )

    selected_vessels = []

    remaining_cargo = (
        cargo_quantity
    )

    for _, vessel in (
        available_vessels.iterrows()
    ):

        capacity = float(
            vessel["capacity_mt"]
        )

        selected_vessels.append(
            vessel
        )

        remaining_cargo -= capacity

        if remaining_cargo <= 0:
            break

    if remaining_cargo > 0:

        return {
            "error":
                "Insufficient vessel capacity for the selected cargo"
        }

    selected_vessel_df = pd.DataFrame(
        selected_vessels
    )

    vessels_required = len(
        selected_vessels
    )

    total_vessel_capacity = float(
        selected_vessel_df[
            "capacity_mt"
        ].sum()
    )

    capacity_utilization = (
        cargo_quantity
        /
        total_vessel_capacity
        *
        100
    )

    primary_vessel = (
        selected_vessel_df.iloc[0]
    )

    selected_vessel = {
        "vessel_name":
            primary_vessel["vessel_name"],

        "vessel_type":
            primary_vessel["vessel_type"],

        "capacity_mt":
            float(
                primary_vessel["capacity_mt"]
            ),

        "status":
            primary_vessel["status"]
    }

    # --------------------------------------------------
    # PROFIT CALCULATION
    # --------------------------------------------------

    cargo_value = (
        cargo_quantity
        * coal_price
    )

    purchase_cost = cargo_value

    current_freight_cost = (
        cargo_quantity
        * current_freight_rate
    )

    forecast_freight_cost = (
        cargo_quantity
        * predicted_freight_rate
    )

    if selling_price > 0:

        revenue = (
            cargo_quantity
            * selling_price
        )

        current_profit = (
            revenue
            - purchase_cost
            - current_freight_cost
        )

        forecast_profit = (
            revenue
            - purchase_cost
            - forecast_freight_cost
        )

        break_even_selling_price = (
            purchase_cost
            + forecast_freight_cost
        ) / cargo_quantity

        max_acceptable_freight_rate = (
            selling_price
            - coal_price
        )

    else:

        revenue = 0

        current_profit = 0

        forecast_profit = 0

        break_even_selling_price = (
            coal_price
            + predicted_freight_rate
        )

        max_acceptable_freight_rate = (
            0
        )

    if cargo_quantity > 0:

        profit_per_mt = (
            current_profit
            / cargo_quantity
        )

    else:

        profit_per_mt = 0

    if revenue > 0:

        current_profit_margin = (
            current_profit
            / revenue
            * 100
        )

        forecast_profit_margin = (
            forecast_profit
            / revenue
            * 100
        )

    else:

        current_profit_margin = 0

        forecast_profit_margin = 0

    # --------------------------------------------------
    # AI DECISION ENGINE
    # --------------------------------------------------

    decision_score = 50

    decision_reasons = []

    # Port

    if (
        str(
            best_route["port_status"]
        ).lower()
        in [
            "closed",
            "suspended"
        ]
    ):

        decision_score -= 40

        decision_reasons.append(
            "Port status is not operational."
        )

    else:

        decision_score += 10

        decision_reasons.append(
            "Selected route has an operational port."
        )

    # Weather

    if best_route["weather_risk"] == "High":

        decision_score -= 30

        decision_reasons.append(
            "High weather risk increases shipment risk."
        )

    elif best_route["weather_risk"] == "Medium":

        decision_score -= 10

        decision_reasons.append(
            "Medium weather risk requires monitoring."
        )

    else:

        decision_score += 10

        decision_reasons.append(
            "Weather conditions are relatively favorable."
        )

    # Overall risk

    if overall_risk == "High":

        decision_score -= 25

        decision_reasons.append(
            "Overall shipment risk is high."
        )

    elif overall_risk == "Medium":

        decision_score -= 10

        decision_reasons.append(
            "Overall shipment risk is moderate."
        )

    else:

        decision_score += 10

        decision_reasons.append(
            "Overall shipment risk is low."
        )

    # Vessel

    if vessels_required == 1:

        decision_score += 10

        decision_reasons.append(
            "Available vessel capacity is sufficient."
        )

    else:

        decision_score -= 5

        decision_reasons.append(
            "Multiple vessels are required."
        )

    # Freight trend

    if market_trend == "Rising":

        decision_score += 10

        decision_reasons.append(
            "Forecast indicates freight rates may rise."
        )

    elif market_trend == "Falling":

        decision_score -= 5

        decision_reasons.append(
            "Forecast indicates freight rates may fall."
        )

    else:

        decision_reasons.append(
            "Freight market is relatively stable."
        )

    # Profitability

    if selling_price > 0:

        if current_profit < 0:

            decision_score -= 25

            decision_reasons.append(
                "Current estimated profit is negative."
            )

        elif current_profit > 0:

            decision_score += 20

            decision_reasons.append(
                "Current estimated shipment profit is positive."
            )

        if forecast_profit < current_profit:

            decision_score -= 10

            decision_reasons.append(
                "Forecast freight cost may reduce future profit."
            )

        elif forecast_profit > current_profit:

            decision_score += 10

            decision_reasons.append(
                "Forecast indicates improved profit potential."
            )

    # Route score

    if best_route["route_score"] >= 70:

        decision_score += 10

        decision_reasons.append(
            "Selected route has a strong route score."
        )

    elif best_route["route_score"] < 45:

        decision_score -= 10

        decision_reasons.append(
            "Selected route has a lower route score."
        )

    decision_score = max(
        0,
        min(
            100,
            round(decision_score)
        )
    )

    # --------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------

    if (
        str(
            best_route["port_status"]
        ).lower()
        in [
            "closed",
            "suspended"
        ]
    ):

        decision = "AVOID"

    elif overall_risk == "High":

        decision = "AVOID"

    elif (
        selling_price > 0
        and current_profit < 0
    ):

        decision = "WAIT"

    elif decision_score >= 75:

        decision = "BUY & SHIP NOW"

    elif decision_score >= 50:

        decision = "BOOK"

    elif decision_score >= 25:

        decision = "WAIT"

    else:

        decision = "AVOID"

    # --------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------

    confidence = round(
        60
        + abs(
            decision_score - 50
        ) * 0.8
    )

    confidence = max(
        50,
        min(
            95,
            confidence
        )
    )

    # --------------------------------------------------
    # 7-DAY FORECAST OUTPUT
    # --------------------------------------------------

    forecast_records = []

    if not route_forecast.empty:

        route_forecast = (
            route_forecast
            .sort_values("date")
            .head(7)
        )

        forecast_records = (
            route_forecast
            .to_dict(
                orient="records"
            )
        )

        # Convert pandas timestamps
        for item in forecast_records:

            if "date" in item:

                item["date"] = (
                    pd.to_datetime(
                        item["date"]
                    ).strftime(
                        "%Y-%m-%d"
                    )
                )

            # Make values JSON safe
            for key, value in list(
                item.items()
            ):

                if pd.isna(value):

                    item[key] = None

                elif hasattr(
                    value,
                    "item"
                ):

                    item[key] = value.item()

    # --------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------

    result = {

        "status":
            "success",

        "cargo_quantity_mt":
            cargo_quantity,

        "coal_price":
            coal_price,

        "selling_price":
            selling_price,

        "cargo_value":
            round(
                cargo_value,
                2
            ),

        "purchase_cost":
            round(
                purchase_cost,
                2
            ),

        "origin_port":
            origin_port,

        "destination_port":
            destination_port,

        # Freight

        "current_freight_rate":
            round(
                current_freight_rate,
                2
            ),

        "predicted_freight_rate":
            round(
                predicted_freight_rate,
                2
            ),

        "rate_difference":
            round(
                rate_difference,
                2
            ),

        "market_trend":
            market_trend,

        # Route

        "best_route":
            best_route,

        "route_analysis":
            route_analysis,

        # Weather

        "weather_risk":
            best_route[
                "weather_risk"
            ],

        "temperature_c":
            best_route[
                "temperature_c"
            ],

        "wind_speed_knots":
            best_route[
                "wind_speed_knots"
            ],

        "wave_height_m":
            best_route[
                "wave_height_m"
            ],

        "rainfall_mm":
            best_route[
                "rainfall_mm"
            ],

        "overall_risk":
            overall_risk,

        "risk_points":
            risk_points,

        # Vessel

        "selected_vessel":
            selected_vessel,

        "vessels_required":
            vessels_required,

        "total_vessel_capacity":
            total_vessel_capacity,

        "capacity_utilization":
            round(
                capacity_utilization,
                2
            ),

        # Profit

        "revenue":
            round(
                revenue,
                2
            ),

        "current_freight_cost":
            round(
                current_freight_cost,
                2
            ),

        "forecast_freight_cost":
            round(
                forecast_freight_cost,
                2
            ),

        "current_profit":
            round(
                current_profit,
                2
            ),

        "forecast_profit":
            round(
                forecast_profit,
                2
            ),

        "profit_per_mt":
            round(
                profit_per_mt,
                2
            ),

        "current_profit_margin":
            round(
                current_profit_margin,
                2
            ),

        "forecast_profit_margin":
            round(
                forecast_profit_margin,
                2
            ),

        "break_even_selling_price":
            round(
                break_even_selling_price,
                2
            ),

        "max_acceptable_freight_rate":
            round(
                max_acceptable_freight_rate,
                2
            ),

        # AI

        "decision":
            decision,

        "decision_score":
            decision_score,

        "confidence":
            confidence,

        "decision_reasons":
            decision_reasons,

        # IMPORTANT:
        # This is what forecast.html needs

        "forecast":
            forecast_records,

        "forecast_days":
            len(
                forecast_records
            )
    }

    return result