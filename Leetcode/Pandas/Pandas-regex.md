
## **Method: `.str.contains()`**

This method is used to check if a string in a Series contains a substring or pattern.

#### **Example: Filter rows where a column contains email addresses with a specific domain**

```python
import pandas as pd

# Sample DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
        'Email': ['alice@gmail.com', 'bob@yahoo.com', 'charlie@gmail.com', 'diana@outlook.com']}
df = pd.DataFrame(data)

# Use .str.contains() to filter emails with 'gmail.com'
filtered_df = df[df['Email'].str.contains(r'@gmail\.com', regex=True)]

print(filtered_df)

```

#### **Output:**

``` sql
     Name             Email
0   Alice   alice@gmail.com
2  Charlie  charlie@gmail.com

```
---

### Explanation:

1. **`str.contains(r'@gmail\.com', regex=True)`**: The `regex=True` flag ensures the pattern is treated as a regular expression.
2. **Regular Expression `@gmail\.com`**:
    - `@gmail` matches the literal string `@gmail`.
    - `\.` matches the literal dot (`.`), as `.` is a special regex character.
    - `com` matches the string `com`.

This example filters rows where the **Email** column contains `@gmail.com`.



## **Method: `.str.replace()`**

The `.str.replace()` method is used to replace substrings or patterns in a Series using regular expressions.

#### **Example: Replace phone number format**

Imagine you have a column with phone numbers in different formats, and you want to standardize them.

```python
import pandas as pd

# Sample DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
        'Phone': ['123-456-7890', '(123) 456-7890', '123.456.7890', '1234567890']}
df = pd.DataFrame(data)

# Use .str.replace() to standardize phone numbers to the format '123-456-7890'
df['Phone'] = df['Phone'].str.replace(r'\D', '-', regex=True).str.replace(r'-+', '-', regex=True).str.strip('-')

print(df)

```

#### **Output:**

```sql
      Name           Phone
0    Alice   123-456-7890
1      Bob   123-456-7890
2  Charlie   123-456-7890
3    Diana   123-456-7890

```
---

### Explanation:

1. **Regex Patterns:**
    - `r'\D'`: Matches any non-digit character (e.g., `(`, `)`, `.`).
    - `'-+'`: Matches one or more consecutive dashes (`-`), ensuring no double dashes are left.
2. **Transformations:**
    - `str.replace(r'\D', '-', regex=True)`: Replaces non-digit characters with a dash (`-`).
    - `str.replace(r'-+', '-', regex=True)`: Replaces multiple dashes with a single dash.
    - `str.strip('-')`: Removes leading or trailing dashes.