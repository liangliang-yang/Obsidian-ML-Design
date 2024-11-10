
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


