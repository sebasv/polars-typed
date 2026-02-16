# Polars-typed

Let your static type checker help you remember which columns are present in your dataframe, and which types they (should) have!

## Why polars-typed?

* Static schema validation. No more typos in string column names.
* Autocomplete on the available columns of you dataframe!
* Catch schema errors before your code hits prod. Yes you wrote a unit test but you hardcoded some naive datetimes in the test data, and now it turns out your data lake is serving datetimes with time zones, causing exceptions in production. `polars-typed` forces you to think about this ahead of release, so the problem never reaches prod.

### What about alternatives?

What about `pandera`? Pandera does data validation, but _not_ schema validation. `polars-typed` does schema validation, _not_ data validation.

What about `poldantic`? Poldantic does not integrate with static type checkers. `polars-typed` aims to prevent a whole class of problems from ever reaching the first unit test (or prod..) by letting you validate all your data transforms with a single `pyright` call, or live diagnostics if you have a type checker configured in your IDE.


### statically enforced, runtime-checked schemas for dataframes

Mypy/pyright/ty reminds you to add schema validation, and tells you what schema you can expect from a function.
The actual data validation happens at runtime.

Schemas offer two modes of validation:

`DataFrameSchema.validate` performs strict validation, failing if the schema is not exactly as specified.
`DataFrameSchema.coerce` attempts to coerce before validation, so unnecessary columns are dropped and columns can be reordered. Optionally datatypes are cast (as long as this can be done without loss of information; casting 12345 to `pl.Int8` will fail).

A `DataFrameSchema` _must_ specify each column using `polars.DataType` types; any other type (eg `str` or `int`) will result in a type error.

Both `DataFrame`s and `LazyFrame`s are supported. Note that validation on `LazyFrame`s is potentially expensive.

```python
import polars as pl
from polars_typed import Column, DataFrame, DataFrameSchema

class TestSchema(DataFrameSchema):
    foo = Column(pl.Boolean)
    # Some polars datatypes carry metadata on the object level instead of the type level.
    # For that reason it is necessary to assign (=) the columns, instead of defining them through type annotations (:)
    bar = Column(pl.Datetime(time_unit="us", time_zone=None))

def typed_function(df: DataFrame[TestSchema]) -> DataFrame[TestSchema]:
    df_untyped = df.filter(pl.col("foo"))
    # return df_untyped # mypy/pyright/ty complains
    return TestSchema.validate(df_untyped) # mypy/pyright/ty is happy

untyped_df = pl.DataFrame({"foo":[False], "bar":1})

typed_function(untyped_df) # mypy/pyright/ty complains
typed_df = typed_function(TestSchema.validate(untyped_df)) # mypy/pyright/ty is happy

# the Column wrapper type lets us use the schema as an enum of column identifiers as well
typed_df.select(TestSchema.foo)
  ```
