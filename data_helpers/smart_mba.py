from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from pandas import DataFrame
from tqdm.auto import tqdm


def prepare_filtered_transactions(
    df: DataFrame,
    group_by_col: str,
    item_col: str,
    include_items: List[str],
    min_items_in_transaction: int = 2
) -> DataFrame:
    """
    Prepares a one-hot encoded dataframe of transactions for frequent itemset mining.

    Filters transactions based on:
    - A target list of items (include_items)
    - A minimum number of distinct items in a transaction (min_items_in_transaction)

    Args:
        df: Raw input dataframe containing individual item rows.
        group_by_col: Column to group by for transactions (e.g. CustomerID or OrderID).
        item_col: Column containing the item to analyze (e.g. ProductName or ProductEdition).
        include_items: List of items to filter for (only transactions containing at least one are kept).
        min_items_in_transaction: Minimum number of items a transaction must contain to be included.

    Returns:
        One-hot encoded dataframe for use in Apriori or FP-Growth.
    """
    grouped = df.groupby(group_by_col)[item_col].apply(list).reset_index()

    if include_items:
        # Original
        # grouped = grouped[grouped[item_col].apply(
        #     lambda items: any(p in items for p in include_items)
        # )]

        # Set-intersection mask.
        include_set = set(include_items)
        mask = grouped[item_col].apply(lambda items: bool(set(items) & include_set))
        grouped = grouped[mask]

    if min_items_in_transaction > 1:
        grouped = grouped[grouped[item_col].apply(lambda x: len(set(x)) >= min_items_in_transaction)]

    # transactions = grouped[item_col].tolist()
    transactions = [list(set(items)) for items in grouped[item_col]]
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    return df_encoded


def run_filtered_rules(
    df_encoded: DataFrame,
    antecedents: List[str],
    min_support: float = 0.02,
    min_confidence: float = 0.5,
    algorithm: str = "apriori",
    max_len: int = 3
) -> DataFrame:
    """
    Runs Apriori or FP-Growth and filters rules to include only those with specified antecedents.

    Adds a 'frequency' column to show the raw count of rule occurrences.

    Args:
        df_encoded: One-hot encoded transaction dataframe.
        antecedents: List of items to match in the rule's antecedent (LHS).
        min_support: Minimum support threshold for frequent itemsets.
        min_confidence: Minimum confidence threshold for rule generation.
        algorithm: 'apriori' or 'fpgrowth' to choose the mining method.
        max_len: Maximum size of itemsets to consider during mining.

    Returns:
        Dataframe of association rules with support, confidence, lift, and frequency.
    """
    if algorithm.lower() == "fpgrowth":
        frequent = fpgrowth(df_encoded, min_support=min_support, use_colnames=True, max_len=max_len)
    elif algorithm.lower() == "apriori":
        frequent = apriori(df_encoded, min_support=min_support, use_colnames=True, max_len=max_len)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Choose 'apriori' or 'fpgrowth'.")

    rules = association_rules(frequent, metric='confidence', min_threshold=min_confidence)

    if antecedents:
        rules = rules[rules['antecedents'].apply(lambda x: any(a in x for a in antecedents))]

    rules['frequency'] = rules['support'] * len(df_encoded)

    return rules[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'frequency']]


def run_configurable_scenarios(
    df: DataFrame,
    account_col: str,
    order_col: str,
    product_name_col: str,
    product_names_to_include: List[str],
    edition_col: str,
    editions_to_include: List[str],
    min_support: float = 0.02,
    min_confidence: float = 0.5,
    min_items_in_transaction: int = 2,
    algorithm: str = "apriori",
    max_len: int = 3,
    output_dir: Union[str, Path] = 'mb_output'
) -> Dict[str, DataFrame]:
    """
    Runs market basket analysis for 4 different scenarios:
    - Customer + ProductName
    - Customer + ProductEdition
    - Order + ProductName
    - Order + ProductEdition

    Outputs results to CSV and returns the rules as a dictionary of DataFrames.

    Args:
        df: The full input dataframe containing customer, order, and product info.
        account_col: Column name representing the customer (e.g. 'CustomerID').
        order_col: Column name representing the order (e.g. 'OrderID').
        product_name_col: Column with general product names.
        product_names_to_include: List of product names to focus the analysis on.
        edition_col: Column with product edition identifiers.
        editions_to_include: List of editions to focus the analysis on.
        min_support: Minimum support threshold.
        min_confidence: Minimum confidence threshold.
        min_items_in_transaction: Filter out transactions smaller than this.
        algorithm: 'apriori' or 'fpgrowth' for mining.
        output_dir: Folder to write the CSV outputs to.

    Returns:
        Dictionary of rule DataFrames, keyed by scenario.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    scenarios = [
        (account_col, product_name_col, product_names_to_include),
        (account_col, edition_col, editions_to_include),
        (order_col, product_name_col, product_names_to_include),
        (order_col, edition_col, editions_to_include)
    ]

    results: Dict[str, DataFrame] = {}

    for group_by, item_col, filter_items in tqdm(scenarios, desc="Running MB scenarios", leave=True):
        scenario_name = f"{group_by} + {item_col}"
        print(f"\nProcessing scenario: {scenario_name}")

        print("  Preparing transactions...")
        df_encoded = prepare_filtered_transactions(
            df=df,
            group_by_col=group_by,
            item_col=item_col,
            include_items=filter_items,
            min_items_in_transaction=min_items_in_transaction
        )
        print(f"  Transactions after filtering: {len(df_encoded)}")

        print("  Mining rules...")
        rules = run_filtered_rules(
            df_encoded=df_encoded,
            antecedents=filter_items,
            min_support=min_support,
            min_confidence=min_confidence,
            algorithm=algorithm,
            max_len=max_len
        )
        print(f"  Rules generated: {len(rules)}")

        key = f"{group_by.lower()}_{item_col.lower()}_{algorithm.lower()}"
        file_path = output_path / f"{key}_rules.csv"
        rules.to_csv(file_path, index=False)
        print(f"  Rules saved to: {file_path}")

        results[key] = rules

    return results


def main() -> None:
    """
    Example entry point to run the full pipeline.
    Loads a CSV, defines config, and runs analysis.
    """
    import pandas as pd

    # Load your data
    df = pd.read_csv("your_data.csv")

    # Define your configuration
    account_col = "CustomerID"
    order_col = "OrderID"
    product_name_col = "ProductName"
    product_names = ["Product A", "Product B"]

    edition_col = "ProductEdition"
    editions = ["Edition A1", "Edition B1"]

    # Algorithm choice: "apriori" or "fpgrowth"
    algorithm = "fpgrowth"

    # Run analysis
    results = run_configurable_scenarios(
        df=df,
        account_col=account_col,
        order_col=order_col,
        product_name_col=product_name_col,
        product_names_to_include=product_names,
        edition_col=edition_col,
        editions_to_include=editions,
        min_support=0.02,
        min_confidence=0.5,
        min_items_in_transaction=2,
        algorithm=algorithm,
        max_len=3,
        output_dir="mb_output"
    )

    # Display top rules from each scenario
    for key, rules_df in results.items():
        if not rules_df.empty:
            top_rules = rules_df.sort_values(by="lift", ascending=False).head(5)
            print(top_rules.to_string(index=False))
        else:
            print(f"No rules found for scenario: {key}")



if __name__ == "__main__":
    main()
