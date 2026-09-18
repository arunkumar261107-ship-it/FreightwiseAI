import pandas as pd


def clean_freight_data(df):
    df = df.copy()

    # Convert date column
    df["date"] = pd.to_datetime(df["date"])

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Convert numeric columns
    numeric_columns = [
        "freight_rate",
        "coal_price",
        "fuel_price",
        "demand"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Remove rows with missing important values
    df = df.dropna(
        subset=[
            "date",
            "route",
            "freight_rate",
            "coal_price",
            "fuel_price",
            "demand",
            "weather_risk"
        ]
    )

    return df


if __name__ == "__main__":
    df = pd.read_csv("data/freight_data.csv")

    cleaned_df = clean_freight_data(df)

    print("Original rows:", len(df))
    print("Cleaned rows:", len(cleaned_df))
    print("\nCleaned data:")
    print(cleaned_df)