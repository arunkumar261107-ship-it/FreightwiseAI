import pandas as pd
from sklearn.linear_model import LinearRegression


def calculate_forecast(freight, routes, weather):

    freight = freight.copy()
    routes = routes.copy()
    weather = weather.copy()

    # ---------------------------------------------------------
    # DATE PREPARATION
    # ---------------------------------------------------------

    freight["date"] = pd.to_datetime(
        freight["date"]
    )

    weather["date"] = pd.to_datetime(
        weather["date"]
    )

    # ---------------------------------------------------------
    # WEATHER RISK MAPPING
    # ---------------------------------------------------------

    risk_mapping = {
        "Low": 0,
        "Medium": 1,
        "High": 2
    }

    freight["weather_risk_score"] = (
        freight["weather_risk"]
        .map(risk_mapping)
        .fillna(1)
    )

    # ---------------------------------------------------------
    # SORT DATA
    # ---------------------------------------------------------

    freight = freight.sort_values(
        ["route", "date"]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # DATE NUMBER
    # ---------------------------------------------------------

    global_min_date = freight["date"].min()

    freight["date_number"] = (
        freight["date"]
        - global_min_date
    ).dt.days

    # ---------------------------------------------------------
    # ML FEATURES
    # ---------------------------------------------------------

    features = [
        "date_number",
        "coal_price",
        "fuel_price",
        "demand",
        "weather_risk_score"
    ]

    target = "freight_rate"

    X = freight[features].fillna(0)

    y = freight[target].fillna(
        freight[target].mean()
    )

    # ---------------------------------------------------------
    # TRAIN ML MODEL
    # ---------------------------------------------------------

    model = LinearRegression()

    model.fit(
        X,
        y
    )

    # ---------------------------------------------------------
    # HISTORICAL PREDICTION
    # ---------------------------------------------------------

    freight["predicted_freight_rate"] = (
        model.predict(X)
    )

    # ---------------------------------------------------------
    # LATEST FREIGHT DATA
    # ---------------------------------------------------------

    latest_freight = (
        freight
        .sort_values("date")
        .groupby("route")
        .tail(1)
        .copy()
    )

    # ---------------------------------------------------------
    # LATEST WEATHER DATA
    # ---------------------------------------------------------

    latest_weather = (
        weather
        .sort_values("date")
        .groupby("route")
        .tail(1)
        .copy()
    )

    # ---------------------------------------------------------
    # FUTURE FORECAST
    # ---------------------------------------------------------

    future_rows = []

    forecast_days = 7

    for _, row in latest_freight.iterrows():

        route_name = row["route"]

        last_date = row["date"]

        route_weather = latest_weather[
            latest_weather["route"]
            == route_name
        ]

        if route_weather.empty:

            temperature_c = 0
            wind_speed_knots = 0
            wave_height_m = 0
            rainfall_mm = 0
            weather_risk = "Medium"

        else:

            weather_row = route_weather.iloc[0]

            temperature_c = float(
                weather_row["temperature_c"]
            )

            wind_speed_knots = float(
                weather_row["wind_speed_knots"]
            )

            wave_height_m = float(
                weather_row["wave_height_m"]
            )

            rainfall_mm = float(
                weather_row["rainfall_mm"]
            )

            weather_risk = weather_row[
                "weather_risk"
            ]

        weather_score = risk_mapping.get(
            weather_risk,
            1
        )

        # -----------------------------------------------------
        # NEXT 7 DAYS
        # -----------------------------------------------------

        for day in range(
            1,
            forecast_days + 1
        ):

            future_date = (
                last_date
                + pd.Timedelta(
                    days=day
                )
            )

            future_date_number = (
                future_date
                - global_min_date
            ).days

            future_input = pd.DataFrame(
                [
                    {
                        "date_number":
                            future_date_number,

                        "coal_price":
                            row["coal_price"],

                        "fuel_price":
                            row["fuel_price"],

                        "demand":
                            row["demand"],

                        "weather_risk_score":
                            weather_score
                    }
                ]
            )

            predicted_rate = float(
                model.predict(
                    future_input[features]
                )[0]
            )

            # -------------------------------------------------
            # IMPORTANT:
            # freight_rate is retained for compatibility
            # with decision_engine.py
            # -------------------------------------------------

            current_rate = float(
                row["freight_rate"]
            )

            rate_difference = (
                predicted_rate
                - current_rate
            )

            if rate_difference > 0.5:

                market_trend = "Rising"

            elif rate_difference < -0.5:

                market_trend = "Falling"

            else:

                market_trend = "Stable"

            future_rows.append(
                {
                    "route":
                        route_name,

                    "date":
                        future_date,

                    "forecast_day":
                        day,

                    # Compatibility column
                    "freight_rate":
                        current_rate,

                    "current_freight_rate":
                        current_rate,

                    "predicted_freight_rate":
                        round(
                            predicted_rate,
                            2
                        ),

                    "rate_difference":
                        round(
                            rate_difference,
                            2
                        ),

                    "market_trend":
                        market_trend,

                    "coal_price":
                        float(
                            row["coal_price"]
                        ),

                    "fuel_price":
                        float(
                            row["fuel_price"]
                        ),

                    "demand":
                        float(
                            row["demand"]
                        ),

                    "weather_risk":
                        weather_risk,

                    "temperature_c":
                        temperature_c,

                    "wind_speed_knots":
                        wind_speed_knots,

                    "wave_height_m":
                        wave_height_m,

                    "rainfall_mm":
                        rainfall_mm
                }
            )

    # ---------------------------------------------------------
    # FUTURE FORECAST DATAFRAME
    # ---------------------------------------------------------

    future_forecast = pd.DataFrame(
        future_rows
    )

    # ---------------------------------------------------------
    # ROUTE INFORMATION
    # ---------------------------------------------------------

    if not future_forecast.empty:

        future_forecast = future_forecast.merge(
            routes[
                [
                    "route",
                    "base_freight_rate",
                    "avg_voyage_days"
                ]
            ],
            on="route",
            how="left"
        )

    # ---------------------------------------------------------
    # RETURN FORECAST
    # ---------------------------------------------------------

    return future_forecast


# -------------------------------------------------------------
# DIRECT TEST
# -------------------------------------------------------------

if __name__ == "__main__":

    freight = pd.read_csv(
        "data/freight_data.csv"
    )

    routes = pd.read_csv(
        "data/route_data.csv"
    )

    weather = pd.read_csv(
        "data/weather_data.csv"
    )

    forecast = calculate_forecast(
        freight,
        routes,
        weather
    )

    print()
    print(
        "FREIGHTWISE AI - 7 DAY ML FORECAST"
    )
    print(
        "=" * 90
    )

    if forecast.empty:

        print(
            "No forecast data available."
        )

    else:

        print(
            forecast[
                [
                    "route",
                    "date",
                    "forecast_day",
                    "freight_rate",
                    "predicted_freight_rate",
                    "rate_difference",
                    "market_trend",
                    "weather_risk"
                ]
            ].to_string(
                index=False
            )
        )