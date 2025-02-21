## **Column Utilities**

This document provides examples of how to use the functions in `column_utils.py` to handle column name conflicts and standardize column names.

---

## **Handling Column Name Conflicts**
If the specified column name already exists in the DataFrame, the function `get_unique_column_name` automatically appends a number to create a unique column name. This ensures that existing data is not overwritten while preserving the original column.

```python
import pandas as pd
from column_utils import get_unique_column_name

# Create a DataFrame with duplicate column names
df = pd.DataFrame({"names": ["Alice", "Alicia"], "Group_Number": [1, 1]})

# Get a unique column name
unique_name = get_unique_column_name(df, "Group_Number")
print(unique_name)  # Output: "Group_Number_1"

# Add a new column using the unique name
df[unique_name] = df["Group_Number"]
print(df.columns)  # Will contain "Group_Number_1" if "Group_Number" already exists.
```

### **Output:**
```
Group_Number_1
Index(['names', 'Group_Number', 'Group_Number_1'], dtype='object')
    names  Group_Number  Group_Number_1
0   Alice             1               1
1  Alicia             1               1
```

---

## **Standardizing Column Names**
The function `standardize_column_names` allows you to:
- Convert column names to **lowercase, uppercase, title case, or capitalize the first letter**.
- Replace **spaces** with either **underscores (`_`)** or **single spaces (` `)**.
- Remove **special characters**.

```python
import pandas as pd
from column_utils import standardize_column_names

# Create a DataFrame with messy column names
df = pd.DataFrame({"First Name": [1, 2], "Last Name!": [3, 4], "e-MAIL Address": [5, 6]})

# Standardize column names
df = standardize_column_names(df)
print(df.columns)  # Default: Lowercase with underscores
```

### **Output:**
```
['first_name', 'last_name', 'e_mail_address']
```

### **Customizing Separator and Case Style**
You can customize the separator and case style:

```python
# Title Case with Spaces
df = standardize_column_names(df, separator=" ", case="title")
print(df.columns)  # ['First Name', 'Last Name', 'E Mail Address']

# Uppercase with Underscores
df = standardize_column_names(df, separator="_", case="upper")
print(df.columns)  # ['FIRST_NAME', 'LAST_NAME', 'E_MAIL_ADDRESS']

# Capitalize First Letter (Sentence Case)
df = standardize_column_names(df, separator="_", case="capitalize")
print(df.columns)  # ['First_name', 'Last_name', 'E_mail_address']
```
