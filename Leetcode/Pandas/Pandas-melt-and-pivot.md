

Given a data set and a target set of variables, there are at least two common issues that require tidying.
## Melting
First, values often appear as columns. Table 4a is an example. To tidy up, you want to turn columns into rows:

![Gather example](http://r4ds.had.co.nz/images/tidy-9.png)

Because this operation takes columns into rows, making a "fat" table more tall and skinny, it is sometimes called _melting_.



To melt the table, you need to do the following.

1. Extract the _column values_ into a new variable. In this case, columns `"1999"` and `"2000"` of `table4` need to become the values of the variable, `"year"`.
2. Convert the values associated with the column values into a new variable as well. In this case, the values formerly in columns `"1999"` and `"2000"` become the values of the `"cases"` variable.

In the context of a melt, let's also refer to `"year"` as the new _key_ variable and `"cases"` as the new _value_ variable.
**Exercise 5** (4 points). Implement the melt operation as a function,

```python
    def melt(df, col_vals, key, value):
        ...
```

It should take the following arguments:
- `df`: the input data frame, e.g., `table4` in the example above;
- `col_vals`: a list of the column names that will serve as values;  column `1999` & `2000` in example  table
- `key`: name of the new variable, e.g., `year` in the example above;
- `value`: name of the column to hold the values. `cases` in the example above

> You may need to refer to the Pandas documentation to figure out how to create and manipulate tables. The bits related to [indexing](http://pandas.pydata.org/pandas-docs/stable/indexing.html) and [merging](http://pandas.pydata.org/pandas-docs/stable/merging.html) may be especially helpful.


