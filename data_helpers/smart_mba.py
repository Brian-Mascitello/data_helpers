from pathlib import Path
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from pandas import DataFrame
from tqdm.auto import tqdm


def save_rules_network_graph(
    rules_df: pd.DataFrame, output_path: Path, key: str, max_rules: int = 100
) -> None:
    """
    Build and save a directed network graph of the top `max_rules` association rules.
    Nodes = individual items (products), edges = antecedent → consequent, edge width ∝ lift.
    """
    # 1) Select only the top rules by lift
    top_rules = rules_df.sort_values("lift", ascending=False).head(max_rules)

    # 2) Build the graph from that subset
    G = nx.DiGraph()
    for _, row in top_rules.iterrows():
        ant = list(row["antecedents"])
        cons = list(row["consequents"])
        for a in ant:
            for c in cons:
                G.add_edge(a, c, weight=row["lift"])

    if G.number_of_edges() == 0:
        return  # nothing to draw

    # 3) Layout and draw
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(G, pos, node_size=300)
    edges = G.edges(data=True)
    widths = [edata["weight"] * 2 for (_, _, edata) in edges]
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=[(u, v) for u, v, _ in edges],
        width=widths,
        arrowstyle="->",
        arrowsize=10,
    )
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.axis("off")
    plt.tight_layout()

    # 4) Save
    png_path = output_path / f"{key}_network_top{max_rules}.png"
    plt.savefig(str(png_path), format="PNG", dpi=300)
    plt.close()


def prepare_filtered_transactions(
    df: DataFrame,
    group_by_col: str,
    item_col: str,
    include_items: List[str],
    min_items_in_transaction: int = 2,
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

    # After grouping but before any filtering:
    if grouped.empty:
        return pd.DataFrame()

    if include_items:
        # Set-intersection mask.
        include_set = set(include_items)
        mask = grouped[item_col].apply(lambda items: bool(set(items) & include_set))
        grouped = grouped[mask].copy()

        # If nothing matched your include_items, exit early:
        if grouped.empty:
            return pd.DataFrame()

    if min_items_in_transaction > 1:
        grouped = grouped[
            grouped[item_col].apply(lambda x: len(set(x)) >= min_items_in_transaction)
        ]

    # transactions = grouped[item_col].tolist()
    transactions = [list(set(items)) for items in grouped[item_col]]
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    return df_encoded


def run_filtered_rules(
    df_encoded: DataFrame,
    items_to_include: List[str],
    min_support: float = 0.02,
    min_confidence: float = 0.5,
    algorithm: str = "apriori",
    max_len: int = 3,
    only_single_antecedent: bool = True,
    filter_side: str = "both",
) -> DataFrame:
    """
    Runs Apriori or FP-Growth and filters rules to include only those with specified antecedents.

    Adds a 'frequency' column to show the raw count of rule occurrences.

    Args:
        df_encoded: One-hot encoded transaction dataframe.
        items_to_include: List of items to match in the rule's antecedent (LHS).
        min_support: Minimum support threshold for frequent itemsets.
        min_confidence: Minimum confidence threshold for rule generation.
        algorithm: 'apriori' or 'fpgrowth' to choose the mining method.
        max_len: Maximum size of itemsets to consider during mining.
        only_single_antecedent: If True, only keep rules where LHS contains exactly one item.
        filter_side: str = "both"
            Which side of the rule to filter on. Options are:
            - "antecedent": only filter LHS
            - "consequent": only filter RHS
            - "both": either side may contain the item(s)

    Returns:
        Dataframe of association rules with support, confidence, lift, and frequency.
    """
    print("  Generating frequent itemsets...")
    if algorithm.lower() == "fpgrowth":
        frequent = fpgrowth(
            df_encoded, min_support=min_support, use_colnames=True, max_len=max_len
        )
    elif algorithm.lower() == "apriori":
        frequent = apriori(
            df_encoded, min_support=min_support, use_colnames=True, max_len=max_len
        )
    else:
        raise ValueError(
            f"Unsupported algorithm: {algorithm}. Choose 'apriori' or 'fpgrowth'."
        )

    print("  Generating association rules...")
    rules = association_rules(
        frequent, metric="confidence", min_threshold=min_confidence
    )

    if items_to_include:
        print(f"  Rules before filtering: {len(rules)}")
        if filter_side not in {"antecedent", "consequent", "both"}:
            raise ValueError(
                "filter_side must be 'antecedent', 'consequent', or 'both'"
            )

        if only_single_antecedent:
            if filter_side == "antecedent":
                rules = rules[
                    rules["antecedents"].apply(
                        lambda x: len(x) == 1 and any(a in x for a in items_to_include)
                    )
                ]
            elif filter_side == "consequent":
                rules = rules[
                    rules["consequents"].apply(
                        lambda x: any(a in x for a in items_to_include)
                    )
                ]
            elif filter_side == "both":
                rules = rules[
                    rules["antecedents"].apply(
                        lambda x: len(x) == 1 and any(a in x for a in items_to_include)
                    )
                    | rules["consequents"].apply(
                        lambda x: any(a in x for a in items_to_include)
                    )
                ]
        else:
            if filter_side == "antecedent":
                rules = rules[
                    rules["antecedents"].apply(
                        lambda x: any(a in x for a in items_to_include)
                    )
                ]
            elif filter_side == "consequent":
                rules = rules[
                    rules["consequents"].apply(
                        lambda x: any(a in x for a in items_to_include)
                    )
                ]
            elif filter_side == "both":
                rules = rules[
                    rules["antecedents"].apply(
                        lambda x: any(a in x for a in items_to_include)
                    )
                    | rules["consequents"].apply(
                        lambda x: any(a in x for a in items_to_include)
                    )
                ]

        print(f"  Rules after item filtering: {len(rules)}")

    rules["frequency"] = rules["support"] * len(df_encoded)

    return rules[
        ["antecedents", "consequents", "support", "confidence", "lift", "frequency"]
    ]


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
    only_single_antecedent: bool = False,
    human_readable: bool = True,
    filter_side: str = "both",
    output_dir: Union[str, Path] = "mb_output",
) -> Dict[str, DataFrame]:
    """
    Runs market basket analysis for 4 different scenarios:
    - Customer + ProductName
    - Customer + ProductEdition
    - Order + ProductName
    - Order + ProductEdition

    Exports results to CSV and returns rule DataFrames.

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
        max_len: Maximum size of itemsets to consider during mining.
        only_single_antecedent: If True, only keep rules where LHS contains exactly one item.
        human_readable: Changes frozensets to comma-separated strings.
        filter_side: str = "both"
            Which side of the rule to filter on. Options are:
            - "antecedent": only filter LHS
            - "consequent": only filter RHS
            - "both": either side may contain the item(s)
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
        (order_col, edition_col, editions_to_include),
    ]

    results: Dict[str, DataFrame] = {}

    for group_by, item_col, filter_items in tqdm(
        scenarios, desc="Running MB scenarios", leave=True
    ):
        scenario_name = f"{group_by} + {item_col}"
        print(f"\nProcessing scenario: {scenario_name}")

        print("  Preparing transactions...")
        df_encoded = prepare_filtered_transactions(
            df=df,
            group_by_col=group_by,
            item_col=item_col,
            include_items=filter_items,
            min_items_in_transaction=min_items_in_transaction,
        )
        print(f"  Transactions after filtering: {len(df_encoded)}")

        print("  Mining rules...")
        rules = run_filtered_rules(
            df_encoded=df_encoded,
            items_to_include=filter_items,
            min_support=min_support,
            min_confidence=min_confidence,
            algorithm=algorithm,
            max_len=max_len,
            only_single_antecedent=only_single_antecedent,
            filter_side=filter_side,
        )

        if rules.empty:
            print("  No rules to export. Skipping this scenario.")

        else:
            print(f"  Rules generated: {len(rules)}")
            # Convert to human-readable if desired
            if human_readable:
                rules["antecedents_str"] = rules["antecedents"].apply(
                    lambda x: ", ".join(sorted(x))
                )
                rules["consequents_str"] = rules["consequents"].apply(
                    lambda x: ", ".join(sorted(x))
                )
                export_columns = [
                    "antecedents_str",
                    "consequents_str",
                    "support",
                    "confidence",
                    "lift",
                    "frequency",
                ]
            else:
                export_columns = [
                    "antecedents",
                    "consequents",
                    "support",
                    "confidence",
                    "lift",
                    "frequency",
                ]

            # Sort by key metrics
            rules = rules.sort_values(
                by=["lift", "confidence", "frequency"], ascending=[False, False, False]
            )

            # Export
            key = f"{group_by.lower()}_{item_col.lower()}_{algorithm.lower()}"
            file_path = output_path / f"{key}_rules.csv"
            rules.to_csv(file_path, index=False, columns=export_columns)
            print(f"  Rules saved to: {file_path}")

            results[key] = rules

            save_rules_network_graph(rules, output_path, key, max_rules=100)
            print(
                f"  Top-100 rules network graph saved to: {output_path / f'{key}_network_top100.png'}"
            )

    return results


def save_all_to_excel(
    results: Dict[str, DataFrame],
    output_path: Union[str, Path],
    filename: str = "all_rules.xlsx",
) -> None:
    """
    Save all rule DataFrames into a single Excel workbook with one sheet per scenario.
    """
    excel_path = Path(output_path) / filename
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        for sheet_name, df in results.items():
            # Truncate sheet name if necessary (Excel max sheet name length = 31)
            safe_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)
    print(f"\nAll rules saved to single Excel file: {excel_path}")


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
        min_support=0.01,
        min_confidence=0.5,
        min_items_in_transaction=2,
        algorithm=algorithm,
        max_len=3,
        only_single_antecedent=False,
        human_readable=True,
        filter_side="both",  # or "antecedent", "consequent"
        output_dir="mb_output",
    )

    save_all_to_excel(results, output_path="mb_output")

    # Display top rules from each scenario
    for key, rules_df in results.items():
        if not rules_df.empty:
            top_rules = rules_df.sort_values(by="lift", ascending=False).head(5)
            print(top_rules.to_string(index=False))
        else:
            print(f"No rules found for scenario: {key}")


if __name__ == "__main__":
    main()
