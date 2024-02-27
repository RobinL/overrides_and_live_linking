import pandas as pd
import splink.duckdb.comparison_library as cl
from IPython.display import display
from splink.datasets import splink_datasets
from splink.duckdb.blocking_rule_library import block_on
from splink.duckdb.linker import DuckDBLinker

records = [
    {
        "unique_id": 1,
        "first_name": "john",
        "surname": "Smith",
        "dob": "1980-07-14",
        "postcode_fake": "AB12 3CD",
        "ground_truth_cluster": "John Smith 1",
    },
    {
        "unique_id": 2,
        "first_name": "john",
        "surname": "Smith",
        "dob": "1980-07-14",
        "postcode_fake": "AB12 3CD",
        "ground_truth_cluster": "John Smith 1",
    },
    # This record gets updated and moves to a different cluster.
    {
        "unique_id": 3,
        "first_name": "john",
        "surname": "smith",
        "dob": "1980-07-14",
        "postcode_fake": "AB12 3CD",
        "ground_truth_cluster": "John Smith 2",
    },
    {
        "unique_id": 4,
        "first_name": "john",
        "surname": "smith",
        "dob": "1985-11-02",
        "postcode_fake": "XY2 9YU",
        "ground_truth_cluster": "John Smith 2",
    },
    {
        "unique_id": 5,
        "first_name": "john",
        "surname": "smith",
        "dob": "1985-11-02",
        "postcode_fake": "XY2 9YU",
        "ground_truth_cluster": "John Smith 2",
    },
    # Unaffected records
    {
        "unique_id": 6,
        "first_name": "Robin",
        "surname": "Linacre",
        "dob": "1980-01-01",
        "postcode_fake": "HH11 1HH",
        "ground_truth_cluster": "No overrides needed",
    },
    {
        "unique_id": 7,
        "first_name": "Robyn",
        "surname": "Linacre",
        "dob": "1980-01-01",
        "postcode_fake": "HH11 1HH",
        "ground_truth_cluster": "No overrides needed",
    },
]


df_tricky = pd.DataFrame(records)

display(df_tricky)

linker = DuckDBLinker(df_tricky, "settings_for_overrides.json")


df_predict = linker.predict()
df_predict.as_pandas_dataframe()
df_clusters = linker.cluster_pairwise_predictions_at_threshold(df_predict, 0.5)

df_clusters.as_pandas_dataframe()


# But then and update occurs:
records[2]["dob"] = "1985-11-02"
records[2]["postcode_fake"] = "XY2 9YU"


# How do we avoid having to do everything again?
df_new = pd.DataFrame(records)

linker = DuckDBLinker(df_new, "settings_for_overrides.json")

df_predict = linker.predict()
df_predict.as_pandas_dataframe()
df_clusters = linker.cluster_pairwise_predictions_at_threshold(df_predict, 0.5)

df_clusters.as_pandas_dataframe()
