# from https://github.com/moj-analytical-services/core_person_record_experiments.git
import pandas as pd
import splink.duckdb.comparison_library as cl
from IPython.display import display
from splink.datasets import splink_datasets
from splink.duckdb.blocking_rule_library import block_on
from splink.duckdb.linker import DuckDBLinker

records = [
    {
        "first_name": "Ryan",
        "surname": "Trick",
        "dob": "1980-07-14",
        "postcode_fake": "AB12 3CD",
        "ground_truth_cluster": "Ryan Tricky",
    },
    {
        "first_name": "Ryan",
        "surname": "Tricky",
        "dob": "1980-07-14",
        "postcode_fake": "AB12 3CD",
        "ground_truth_cluster": "Ryan Tricky",
    },
    {
        "first_name": "Ryan",
        "surname": "Tricky",
        "dob": "1980-07-14",
        "postcode_fake": "AB12 3CD",
        "ground_truth_cluster": "Ryan Tricky",
    },
    # Bryan is Ryan's twin brother
    # Need some manual overrides to prevent
    # these receords linking
    {
        "first_name": "Bryan",
        "surname": "Tricky",
        "dob": "1980-07-14",
        "postcode_fake": "AB12 3CD",
        "ground_truth_cluster": "Bryan Tricky",
    },
    {
        "first_name": "Bryan",
        "surname": "Tricky",
        "dob": "1980-07-14",
        "postcode_fake": "XY14 2ZZ",
        "ground_truth_cluster": "Bryan Tricky",
    },
    {
        "first_name": "Jane",
        "surname": "Single",
        "dob": "1975-11-03",
        "postcode_fake": "PQ12 3RS",
        "ground_truth_cluster": "Jane got married",
    },
    {
        "first_name": "Jane",
        "surname": "Single",
        "dob": "1975-11-03",
        "postcode_fake": "PQ12 3RS",
        "ground_truth_cluster": "Jane got married",
    },
    # Jane got married and moved house so probably needs
    # some form of manual override to link this record
    {
        "first_name": "Jane",
        "surname": "Married",
        "dob": "1975-11-03",
        "postcode_fake": "MO12 VED",
        "ground_truth_cluster": "Jane got married",
    },
    # More normal records
    {
        "first_name": "Robin",
        "surname": "Linacre",
        "dob": "1980-01-01",
        "postcode_fake": "HH11 1HH",
        "ground_truth_cluster": "No overrides needed",
    },
    {
        "first_name": "Robyn",
        "surname": "Linacre",
        "dob": "1980-01-01",
        "postcode_fake": "HH11 1HH",
        "ground_truth_cluster": "No overrides needed",
    },
]


df_tricky = pd.DataFrame(records)
df_tricky["unique_id"] = df_tricky.index

display(df_tricky)

linker = DuckDBLinker(df_tricky, "settings_for_overrides.json")


df_predict = linker.predict()
df_predict.as_pandas_dataframe()
df_clusters = linker.cluster_pairwise_predictions_at_threshold(df_predict, 1e-50)


linker.cluster_studio_dashboard(
    df_predict,
    df_clusters,
    "cluster_studio_no_overrides.html",
    overwrite=True,
    cluster_ids=[0],
)

with open("cluster_studio_no_overrides.html", "r") as file:
    data = file.read().replace(
        'label: "Choose metric for node colour: ",',
        'label: "Choose metric for node colour: ", value: "ground_truth_cluster"',
    )

with open("cluster_studio_no_overrides.html", "w") as file:
    file.write(data)
