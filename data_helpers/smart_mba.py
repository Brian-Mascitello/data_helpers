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
    grouped = df.groupby(group_by_col)[item_col].apply(list).reset_index()

    if include_items:
        grouped = grouped[grouped[item_col].apply(
            lambda items: any(p in items for p in include_items)
        )]

    if min_items_in_transaction > 1:
        grouped = grouped[grouped[item_col].apply(lambda x: len(set(x)) >= min_items_in_transaction)]

    transactions = grouped[item_col].tolist()
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    return df_encoded


def run_filtered_rules(
    df_encoded: DataFrame,
    antecedents: List[str],
    min_support: float = 0.02,
    min_confidence: float = 0.5,
    algorithm: str = "apriori"
) -> DataFrame:
    """
    Runs Apriori or FP-Growth and filters rules to include only those with specified antecedents.
    Adds frequency as a column.
    """
    if algorithm.lower() == "fpgrowth":
        frequent = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
    elif algorithm.lower() == "apriori":
        frequent = apriori(df_encoded, min_support=min_support, use_colnames=True)
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
    output_dir: Union[str, Path] = 'mb_output'
) -> Dict[str, DataFrame]:
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
        df_encoded = prepare_filtered_transactions(
            df=df,
            group_by_col=group_by,
            item_col=item_col,
            include_items=filter_items,
            min_items_in_transaction=min_items_in_transaction
        )

        rules = run_filtered_rules(
            df_encoded=df_encoded,
            antecedents=filter_items,
            min_support=min_support,
            min_confidence=min_confidence,
            algorithm=algorithm
        )

        key = f"{group_by.lower()}_{item_col.lower()}_{algorithm.lower()}"
        file_path = output_path / f"{key}_rules.csv"
        rules.to_csv(file_path, index=False)
        results[key] = rules

    return results


def main() -> None:
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
        output_dir="mb_output"
    )


if __name__ == "__main__":
    main()
