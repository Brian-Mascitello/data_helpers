#!/usr/bin/env python3
import ast
import os
import sys


def find_functions_in_directory(directory="."):
    """
    Walk through all Python files in the specified directory and subdirectories,
    extract function names, and print them along with their file paths.

    :param directory: The root directory to start searching from (default is current directory).
    """
    for root, _dirs, files in os.walk(directory):
        # Skip the "venv" directory to avoid scanning virtual environments
        if "venv" in root.split(os.sep):
            continue

        # Iterate over all files
        for file in files:
            if file.endswith(".py"):  # Process only Python files
                filepath = os.path.join(root, file)
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    try:
                        # Parse the Python file to extract function definitions
                        tree = ast.parse(f.read(), filename=filepath)
                        functions = [
                            node.name
                            for node in ast.walk(tree)
                            if isinstance(node, ast.FunctionDef)
                            and not node.name.startswith("__")
                        ]

                        # Print function names along with their file paths
                        for func in functions:
                            print(f"{filepath}: {func}()")
                    except SyntaxError:
                        # If a syntax error is encountered, print a warning
                        print(f"Syntax error in {filepath}", file=sys.stderr)


if __name__ == "__main__":
    """
    Example usage:
    Suppose we have a file structure like this:
    ├── scripts/
    │   ├── find_functions.py
    │   ├── example_script.py  (contains function definitions)

    If `example_script.py` contains:

    def my_function():
        pass

    Running `python scripts/find_functions.py` will output:
    scripts/example_script.py: my_function()
    """

    # Run the function on the current directory
    find_functions_in_directory(".")
