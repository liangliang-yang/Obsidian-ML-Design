
## str format

```python
template = f"""There are {len(df):,} interactions.
- Unique users: {n_users:,}
- Unique books: {n_books:,}
- Number of reads: {n_reads:,} ({n_reads/len(df)*100:.1f}% of all interactions)
- Number of ratings: {n_rated:,} ({n_rated/len(df)*100:.1f}%)
- Number of reviews: {n_rev:,} ({n_rev/len(df)*100:.1f}%)"""
```

The demo included in the solution cell below should display the following output:

```
There are 12,345 interactions.
- Unique users: 1,766
- Unique books: 9,348
- Number of reads: 6,844 (55.4% of all interactions)
- Number of ratings: 6,389 (51.8%)
- Number of reviews: 744 (6.0%)
```


## str find

```python
text = "Python programming is fun."

# Find the position of the substring 'programming'
index = text.find('programming')
print(index)  # Output: 7

# Find 'Python' starting from index 10 (won't find it, so returns -1)
index = text.find('Python', 10)
print(index)  # Output: -1

# Find 'is' within a specific range
index = text.find('is', 5, 20)
print(index)  # Output: 18

```

The `find()` method in Python is implemented similarly to a sliding window algorithm, scanning the string from left to right. The method compares each position in the string with the start of the target substring. Here’s a breakdown of its implementation and time complexity:

### Simplified Implementation
A typical implementation would look something like this:

```python
def find(substring, string):
    len_sub = len(substring)
    len_str = len(string)

    for i in range(len_str - len_sub + 1):
        # Check if substring matches at position i
        if string[i:i+len_sub] == substring:
            return i
    return -1
```

### Explanation
1. The method iterates through each possible starting position in `string`, from `0` to `len(string) - len(substring) + 1`.
2. For each position, it checks if a slice of `string` matches `substring`.
3. If a match is found, it returns the index; otherwise, it continues.
4. If no match is found by the end, it returns `-1`.

### Time Complexity
The time complexity of `find()` depends on the length of both the `string` \( n \) and the `substring` \( m \):
- **Worst-case complexity**: \( O(n \times m) \), which occurs if the substring is almost as long as the string, and no match exists.
- **Average-case complexity**: \( O(n) \) if the substring is relatively short or there’s an early match.

### Optimized Searching: Knuth-Morris-Pratt (KMP) Algorithm
For longer substrings, a more efficient approach is the **KMP algorithm** with **O(n + m)** complexity. This algorithm preprocesses the substring to create a "partial match" table, allowing the search to skip parts of the text when mismatches are found.

