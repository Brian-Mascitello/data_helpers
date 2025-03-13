from typing import Dict, List, Tuple

CacheType = Dict[Tuple[str, str, int, bool], List]

import pandas as pd
from google.cloud import bigquery


def get_column_names(
    client: bigquery.Client, dataset_id: str, table_id: str
) -> List[str]:
    """Fetch column names for a given BigQuery table."""
    query = f"""
        SELECT column_name
        FROM `{dataset_id}.INFORMATION_SCHEMA.COLUMNS`
        WHERE table_name = '{table_id}'
    """
    return [row["column_name"] for row in client.query(query).result()]


def get_distinct_values(
    client: bigquery.Client,
    dataset_id: str,
    table_id: str,
    column: str,
    limit: int = None,
    order_by: bool = True,
) -> List:
    """
    Retrieves distinct values from a specific column with an optional limit and ordering.

    Args:
        client (bigquery.Client): BigQuery client instance.
        dataset_id (str): The dataset ID.
        table_id (str): The table ID.
        column (str): The column to analyze.
        limit (int, optional): Limit the distinct values to the top 'limit' entries.
        order_by (bool, optional): Whether to order the results before applying the limit.

    Returns:
        List: List of distinct values from the column.
    """
    query = f"SELECT DISTINCT `{column}` FROM `{dataset_id}.{table_id}`"
    if limit is not None:
        if order_by:
            query += f" ORDER BY `{column}` LIMIT {limit}"
        else:
            query += f" LIMIT {limit}"
    result = client.query(query).to_dataframe()
    return result.iloc[:, 0].dropna().tolist()


def get_cached_distinct_values(
    client: bigquery.Client,
    dataset_id: str,
    table_id: str,
    column: str,
    limit: int,
    order_by: bool,
    cache: CacheType,
) -> List:
    """
    Retrieves distinct values using cache to avoid repeated queries.

    Args:
        client (bigquery.Client): BigQuery client instance.
        dataset_id (str): The dataset ID.
        table_id (str): The table ID.
        column (str): The column to analyze.
        limit (int): The limit for distinct values.
        order_by (bool): Whether to order the results before limiting.
        cache (List): Dictionary for caching results.

    Returns:
        List: Cached list of distinct values.
    """
    key = (table_id, column, limit, order_by)
    if key not in cache:
        cache[key] = get_distinct_values(
            client, dataset_id, table_id, column, limit, order_by
        )
    return cache[key]


def compute_overlap_ratio(values_B: List, values_A: List) -> float:
    """
    Computes the percentage of values in list B that exist in list A.

    Args:
        values_B (List): Distinct values from table B.
        values_A (List): Distinct values from table A.

    Returns:
        float: Percentage of values in B that exist in A.
    """
    set_A = set(values_A)
    set_B = set(values_B)

    if not set_B:  # Prevent division by zero
        return 0.0

    overlap = len(set_B.intersection(set_A)) / len(set_B)
    return round(overlap * 100, 2)  # Convert to percentage


def find_potential_join_keys(
    client: bigquery.Client,
    dataset_id: str,
    table_A: str,
    table_B: str,
    distinct_limit_A: int = 1000,
    distinct_limit_B: int = 1000,
    order_by: bool = True,
) -> pd.DataFrame:
    """
    Identifies potential join keys between two tables based on distinct value overlap, ignoring column names.

    Args:
        client (bigquery.Client): BigQuery client instance.
        dataset_id (str): The dataset ID.
        table_A (str): First table ID (reference table).
        table_B (str): Second table ID (foreign table).
        distinct_limit_A (int): Limit for distinct values from Table A columns.
        distinct_limit_B (int): Limit for distinct values from Table B columns.
        order_by (bool): Whether to order the values before limiting.

    Returns:
        pd.DataFrame: DataFrame showing potential join keys ranked by highest overlap.
    """
    # Get column names
    columns_A = get_column_names(client, dataset_id, table_A)
    columns_B = get_column_names(client, dataset_id, table_B)

    # Create a cache for distinct values
    distinct_cache = {}
    join_candidates = []

    for col_B in columns_B:
        # Get distinct values for the current column in Table B from cache
        values_B = get_cached_distinct_values(
            client,
            dataset_id,
            table_B,
            col_B,
            distinct_limit_B,
            order_by,
            distinct_cache,
        )

        best_match = None
        best_overlap = 0

        for col_A in columns_A:
            # Get distinct values for the current column in Table A from cache
            values_A = get_cached_distinct_values(
                client,
                dataset_id,
                table_A,
                col_A,
                distinct_limit_A,
                order_by,
                distinct_cache,
            )

            # Compute overlap ratio
            overlap_ratio = compute_overlap_ratio(values_B, values_A)

            if overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_match = col_A  # Track the best-matching column

        if best_match and best_overlap > 0:  # Ignore 0% matches
            join_candidates.append(
                {
                    "Table B Column": col_B,
                    "Best Match in Table A": best_match,
                    "Overlap Ratio (%)": best_overlap,
                }
            )

    # Convert results into a DataFrame and sort by overlap ratio in descending order
    return pd.DataFrame(join_candidates).sort_values(
        by="Overlap Ratio (%)", ascending=False
    )


def main():
    """Main function to find potential join keys between two tables."""
    client = bigquery.Client()
    dataset_id = "your_project.your_dataset"
    table_A = "table_A"
    table_B = "table_B"

    # User-defined parameters for sampling distinct values
    distinct_limit_A = 1000
    distinct_limit_B = 1000
    # Set order_by to False if you don't want to order values before limiting.
    order_by = True

    df_potential_keys = find_potential_join_keys(
        client,
        dataset_id,
        table_A,
        table_B,
        distinct_limit_A=distinct_limit_A,
        distinct_limit_B=distinct_limit_B,
        order_by=order_by,
    )
    print(df_potential_keys)


if __name__ == "__main__":
    main()
