## **Handling Column Name Conflicts**
If the specified `group_column` name already exists in the DataFrame, the function automatically appends a number to create a unique column name. This ensures that existing data is not overwritten while preserving the original column.
```python
import pandas as pd
from fuzzy_grouping import assign_similarity_groups, get_unique_column_name

df = pd.DataFrame({"names": ["Alice", "Alicia"], "Group_Number": [1, 1]})

unique_name = get_unique_column_name(df, "Group_Number")
print(unique_name)  # Output: "Group_Number_1"

grouped_df = assign_similarity_groups(df, name_column="names", group_column="Group_Number")
print(grouped_df.columns)  # Will contain "Group_Number_1" if "Group_Number" already exists.

print(grouped_df)
```

Output:
```
Group_Number_1
Index(['names', 'Group_Number', 'Group_Number_1'], dtype='object')
    names  Group_Number  Group_Number_1
0   Alice             1               1
1  Alicia             1               1
```
