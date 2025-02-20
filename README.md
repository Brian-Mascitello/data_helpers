# Fuzzy Grouping Utility Examples

This document provides examples of how to use the `assign_similarity_groups` function with different parameter settings.

---

## **Basic Usage (Default Parameters)**
```python
from fuzzy_grouping import assign_similarity_groups
import pandas as pd

df = pd.DataFrame({"names": ["Alice", "Alicia", "Bob", "Bobby", "Robert"]})

grouped_df = assign_similarity_groups(df, name_column="names")
print(grouped_df)
```

Output:
```
    names  Group_Number
0   Alice             1
1  Alicia             1
2     Bob             2
3   Bobby             2
4  Robert             3
```

---

## **Changing the `group_column` Name**
By default, the column storing group numbers is called `Group_Number`. You can customize this by using the group_column parameter.
```python
grouped_df = assign_similarity_groups(df, name_column="names", group_column="Similarity_Group")
print(grouped_df)
```

Output:
```
    names  Similarity_Group
0   Alice                 1
1  Alicia                 1
2     Bob                 2
3   Bobby                 2
4  Robert                 3
```

---

## **Setting a Custom `start_group` Number**
```python
grouped_df = assign_similarity_groups(df, name_column="names", start_group=10)
print(grouped_df)
```

Output:
```
    names  Group_Number
0   Alice            10
1  Alicia            10
2     Bob            11
3   Bobby            11
4  Robert            12
```

---

## **Adjusting the Similarity `threshold`**
The `threshold` parameter determines the minimum similarity score for grouping names together. A lower `threshold` results in more groups, while a higher `threshold` allows for more relaxed matching.
```python
grouped_df = assign_similarity_groups(df, name_column="names", threshold=30)
print(grouped_df)
```

Output:
```
    names  Group_Number
0   Alice             1
1  Alicia             1
2     Bob             2
3   Bobby             2
4  Robert             2
```

---

## **Enforcing Case Sensitivity (case_insensitive=False)**
```python
df_case = pd.DataFrame({"names": ["Alice", "ALICE", "Bob", "BOB", "Bobby"]})

grouped_df = assign_similarity_groups(df_case, name_column="names", case_insensitive=False)
print(grouped_df)
```

Output:
```
   names  Group_Number
0  ALICE             1
1  Alice             2
2    BOB             3
3    Bob             4
4  Bobby             4
```

---

## **Using `presorted=True` for an Already-Sorted DataFrame**
If your DataFrame is already sorted by the name_column, enabling presorted=True can speed up processing without changing the index. However, when presorted=False (default), the function may internally reorder the DataFrame, which can result in a modified index. If maintaining the original index is important, consider resetting or reassigning it after processing.
```python
df_sorted = df.sort_values(by="names")
grouped_df = assign_similarity_groups(df_sorted, name_column="names", presorted=True)
print(grouped_df)
```

Output:
```
    names  Group_Number
0   Alice             1
1  Alicia             1
2     Bob             2
3   Bobby             2
4  Robert             3
```

---

## **Enabling `verbose=True` to Debug Similarity Comparisons**
```python
grouped_df = assign_similarity_groups(df, name_column="names", verbose=True)
print(grouped_df)
```

Output:
```
Comparing: 'alice' ↔ 'alicia' | Similarity: 73 | Threshold: 70 → Same Group
Comparing: 'alicia' ↔ 'bob' | Similarity: 0 | Threshold: 70 → New Group
Comparing: 'bob' ↔ 'bobby' | Similarity: 75 | Threshold: 70 → Same Group
Comparing: 'bobby' ↔ 'robert' | Similarity: 36 | Threshold: 70 → New Group
    names  Group_Number
0   Alice             1
1  Alicia             1
2     Bob             2
3   Bobby             2
4  Robert             3
```

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
