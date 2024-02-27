import pandas as pd
import splink.duckdb.comparison_library as cl
from splink.datasets import splink_datasets
from splink.duckdb.blocking_rule_library import block_on
from splink.duckdb.linker import DuckDBLinker

df = splink_datasets.historical_50k

df
settings_dict = {
    "link_type": "dedupe_only",
    "blocking_rules_to_generate_predictions": [],
    "comparisons": [
        cl.exact_match(
            "first_name",
            term_frequency_adjustments=True,
        ),
        cl.jaro_winkler_at_thresholds(
            "surname", distance_threshold_or_thresholds=[0.9, 0.8]
        ),
        cl.levenshtein_at_thresholds("dob", distance_threshold_or_thresholds=[1, 2]),
        cl.levenshtein_at_thresholds(
            "postcode_fake", distance_threshold_or_thresholds=[1, 2]
        ),
    ],
    "retain_intermediate_calculation_columns": True,
}


linker = DuckDBLinker(df, settings_dict)

linker.estimate_probability_two_random_records_match(
    [block_on(["first_name", "surname", "dob"])], recall=0.6
)

linker.estimate_u_using_random_sampling(target_rows=1e8)

linker.estimate_parameters_using_expectation_maximisation(
    block_on(["first_name", "surname"])
)

linker.estimate_parameters_using_expectation_maximisation(
    block_on(["dob", "substr(postcode_fake, 1,3)"])
)

linker.save_model_to_json("settings_for_overrides.json", overwrite=True)

# Plan - crete two cluster studios
