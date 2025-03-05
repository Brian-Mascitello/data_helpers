#!/usr/bin/env python3
import ast
import os
import sys


def find_functions_in_directory(directory="."):
    """
    List all Python files in the specified directory (non-recursive),
    extract function names, and print them along with their file paths.

    :param directory: The directory to search for Python files.
    """
    if not os.path.exists(directory):
        print(
            f"Error: The directory {os.path.abspath(directory)} does not exist.",
            file=sys.stderr,
        )
        return

    try:
        files = os.listdir(directory)
    except OSError as e:
        print(f"Error accessing directory {directory}: {e}", file=sys.stderr)
        return

    for file in files:
        if file.endswith(".py"):
            filepath = os.path.join(directory, file)
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                try:
                    tree = ast.parse(f.read(), filename=filepath)
                    functions = [
                        node.name
                        for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef)
                        and not node.name.startswith("__")
                    ]
                    for func in functions:
                        print(f"{filepath}: {func}()")
                except SyntaxError:
                    print(f"Syntax error in {filepath}", file=sys.stderr)


if __name__ == "__main__":
    # Print current working directory for debugging
    print("Current working directory:", os.getcwd())

    find_functions_in_directory("./data_helpers")
