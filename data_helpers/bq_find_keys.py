from typing import Dict, List, Optional, Tuple

CacheType = Dict[Tuple[str, str, int, bool], List]

import pandas as pd
from google.cloud import bigquery
from tqdm import tqdm


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
        cache (CacheType): Dictionary for caching results.

    Returns:
        List: Cached list of distinct values.
    """
    key = (table_id, column, limit, order_by)
    if key not in cache:
        cache[key] = get_distinct_values(
            client, dataset_id, table_id, column, limit, order_by
        )
    return cache[key]


def compute_overlap_ratio(source: List, target: List) -> float:
    """
    Computes the percentage of distinct values in the target list that are also found in the source list.

    Args:
        source (List): List of distinct values from one table.
        target (List): List of distinct values from the other table (denominator).

    Returns:
        float: Percentage of values in target found in source.
    """
    set_source = set(source)
    set_target = set(target)
    if not set_target:
        return 0.0
    overlap = len(set_source.intersection(set_target)) / len(set_target)
    overlap = round(overlap * 100, 2)
    return overlap


def find_potential_join_keys(
    client: bigquery.Client,
    dataset_id_A: str,
    table_A: str,
    dataset_id_B: str,
    table_B: str,
    columns_A: Optional[List[str]] = None,
    columns_B: Optional[List[str]] = None,
    distinct_limit_A: int = 1000,
    distinct_limit_B: int = 1000,
    order_by: bool = True,
) -> pd.DataFrame:
    """
    Identifies potential join keys between two tables based on distinct value overlap.

    This function treats table A as the primary table and, for each column in table A,
    finds the best matching column in table B. It computes two overlap ratios:
      - Overlap Ratio A (%): Percentage of distinct values in table A that are found in table B.
      - Overlap Ratio B (%): Percentage of distinct values in table B that are found in table A.

    Args:
        client (bigquery.Client): BigQuery client instance.
        dataset_id_A (str): Dataset ID for table A.
        table_A (str): Table A ID (reference table).
        dataset_id_B (str): Dataset ID for table B.
        table_B (str): Table B ID (foreign table).
        columns_A (Optional[List[str]]): List of columns to check in table A. If None, all columns are used.
        columns_B (Optional[List[str]]): List of columns to check in table B. If None, all columns are used.
        distinct_limit_A (int): Limit for distinct values from table A columns.
        distinct_limit_B (int): Limit for distinct values from table B columns.
        order_by (bool): Whether to order the values before limiting.

    Returns:
        pd.DataFrame: DataFrame with:
          - Table A Column
          - Best Match in Table B
          - Overlap Ratio A (%)
          - Overlap Ratio B (%)
    """
    # If column lists aren't provided, fetch all columns.
    if columns_A is None:
        columns_A = get_column_names(client, dataset_id_A, table_A)
    if columns_B is None:
        columns_B = get_column_names(client, dataset_id_B, table_B)

    distinct_cache: CacheType = {}
    join_candidates = []

    # Process each column in table A with a progress bar.
    for col_A in tqdm(columns_A, desc="Processing Table A columns"):
        values_A = get_cached_distinct_values(
            client,
            dataset_id_A,
            table_A,
            col_A,
            distinct_limit_A,
            order_by,
            distinct_cache,
        )
        if not values_A:
            continue

        best_match = None
        best_avg_overlap = 0.0
        best_overlap_A = 0.0
        best_overlap_B = 0.0

        for col_B in columns_B:
            values_B = get_cached_distinct_values(
                client,
                dataset_id_B,
                table_B,
                col_B,
                distinct_limit_B,
                order_by,
                distinct_cache,
            )
            if not values_B:
                continue

            # Compute the two overlap ratios using the generic function.
            overlap_A = compute_overlap_ratio(
                values_B, values_A
            )  # Percentage of A found in B.
            overlap_B = compute_overlap_ratio(
                values_A, values_B
            )  # Percentage of B found in A.
            avg_overlap = (overlap_A + overlap_B) / 2

            if avg_overlap > best_avg_overlap:
                best_avg_overlap = avg_overlap
                best_match = col_B
                best_overlap_A = overlap_A
                best_overlap_B = overlap_B

        if best_match and best_avg_overlap > 0:
            join_candidates.append(
                {
                    "Table A Column": col_A,
                    "Best Match in Table B": best_match,
                    "Overlap Ratio A (%)": best_overlap_A,
                    "Overlap Ratio B (%)": best_overlap_B,
                }
            )

    return pd.DataFrame(join_candidates).sort_values(
        by="Overlap Ratio A (%)", ascending=False
    )


def main():
    """Main function to find potential join keys between two tables."""
    client = bigquery.Client()

    dataset_id_A = "your_project.your_datasetA"
    table_A = "table_A"
    columns_A = ["acol1", "acol2"]

    dataset_id_B = "your_project.your_datasetB"
    table_B = "table_B"
    columns_B = ["bcol1", "bcol2"]

    distinct_limit_A = 1000
    distinct_limit_B = 1000
    order_by = True

    df_potential_keys = find_potential_join_keys(
        client,
        dataset_id_A,
        table_A,
        dataset_id_B,
        table_B,
        columns_A=columns_A,
        columns_B=columns_B,
        distinct_limit_A=distinct_limit_A,
        distinct_limit_B=distinct_limit_B,
        order_by=order_by,
    )
    print(df_potential_keys)


if __name__ == "__main__":
    main()
