import pandas as pd


def load_freight_data():
    return pd.read_csv(
        "data/freight_data.csv"
    )


def load_vessel_data():
    return pd.read_csv(
        "data/vessel_data.csv"
    )


def load_route_data():
    return pd.read_csv(
        "data/route_data.csv"
    )


def load_weather_data():
    return pd.read_csv(
        "data/weather_data.csv"
    )


def load_data():
    freight = load_freight_data()
    vessels = load_vessel_data()
    routes = load_route_data()
    weather = load_weather_data()

    return (
        freight,
        vessels,
        routes,
        weather
    )


if __name__ == "__main__":

    freight, vessels, routes, weather = load_data()

    print(
        "Freight data:",
        freight.shape
    )

    print(
        "Vessel data:",
        vessels.shape
    )

    print(
        "Route data:",
        routes.shape
    )

    print(
        "Weather data:",
        weather.shape
    )