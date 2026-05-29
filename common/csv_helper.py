import pandas as pd


def read_csv(path: str) -> pd.DataFrame:

    df = pd.read_csv(path)
    df.where(pd.notnull(df), None)

    return df