import polars as pl
import pytest
from polars_typed import Column, DataFrameSchema, data_quality_check, primary_key


class TestSchema(DataFrameSchema):
    foo = Column(pl.Boolean)
    bar = Column(pl.Int8)


class OtherTestSchema(DataFrameSchema):
    baz = Column(pl.String)


class TestInheritedSchema(TestSchema):
    baz = Column(pl.String)


class TestMultipleInheritance(TestSchema, OtherTestSchema): ...


class TestSecondInheritedSchema(TestInheritedSchema):
    boo = Column(pl.Boolean)


def test_columns_are_string() -> None:
    assert TestSchema.foo == "foo"


@pytest.mark.parametrize("schema", [TestInheritedSchema, TestMultipleInheritance])
def test_inheritance(schema: DataFrameSchema) -> None:
    assert schema.foo == "foo"
    assert schema.baz == "baz"


def test_dtype_is_preserved():
    assert TestSchema.foo.dtype == pl.Boolean


def test_col_property_returns_polars_expr():
    """Test that the .col property returns a polars expression."""
    col_expr = TestSchema.foo.col
    assert isinstance(col_expr, pl.Expr)
    assert str(col_expr) == 'col("foo")'


def test_schema_order_is_respected() -> None:
    class ReverseTestSchema(DataFrameSchema):
        bar = Column(pl.Int8)
        foo = Column(pl.Boolean)

    assert ReverseTestSchema.schema() != TestSchema.schema()
    assert list(reversed(ReverseTestSchema.schema().items())) == list(
        TestSchema.schema().items()
    )


def test_schema_exception_message() -> None:
    df = pl.DataFrame(
        {"foo": [True], "bar": ["bar"]}, schema={"foo": pl.Boolean, "bar": pl.String}
    )
    with pytest.raises(TypeError) as e:
        TestSchema.validate(df)

    assert "column bar should be Int8 but is String" in str(e.value)


@pytest.mark.parametrize("lazy", [True, False])
def test_schema_check_invalid_schema_raises(lazy: bool) -> None:
    df: pl.DataFrame | pl.LazyFrame = pl.DataFrame({"foo": [False], "bar": [1]})
    if lazy:
        df = df.lazy()
    with pytest.raises(TypeError):
        TestSchema.validate(df)


@pytest.mark.parametrize("schema", [TestInheritedSchema, TestMultipleInheritance])
@pytest.mark.parametrize("lazy", [True, False])
def test_schema_check_invalid_inherited_schema_raises(
    lazy: bool, schema: DataFrameSchema
) -> None:
    df: pl.DataFrame | pl.LazyFrame = pl.DataFrame({"foo": [False], "bar": [1]})
    if lazy:
        df = df.lazy()
    with pytest.raises(TypeError):
        schema.validate(df)


@pytest.mark.parametrize("lazy", [True, False])
def test_schema_check_valid_schema_passes(lazy: bool) -> None:
    df: pl.DataFrame | pl.LazyFrame = pl.DataFrame(
        {"foo": [False], "bar": [1]}, schema=[("foo", pl.Boolean), ("bar", pl.Int8)]
    )
    if lazy:
        df = df.lazy()
    df = TestSchema.validate(df)

    if lazy:
        df = df.collect()

    assert len(df) == 1


@pytest.mark.parametrize("schema", [TestInheritedSchema, TestMultipleInheritance])
@pytest.mark.parametrize("lazy", [True, False])
def test_schema_check_valid_inherited_schema_passes(
    lazy: bool, schema: DataFrameSchema
) -> None:
    df: pl.DataFrame | pl.LazyFrame = pl.DataFrame(
        {"foo": [False], "bar": [1], "baz": ["i"]},
        schema=[("foo", pl.Boolean), ("bar", pl.Int8), ("baz", pl.String)],
    )
    if lazy:
        df = df.lazy()
    df = schema.validate(df)

    if lazy:
        df = df.collect()

    assert len(df) == 1


@pytest.mark.parametrize("lazy", [True, False])
def test_coercion_valid_schema_passes(lazy: bool) -> None:
    df: pl.DataFrame | pl.LazyFrame = pl.DataFrame(
        {"bar": [1], "foo": [False], "baz": ["a"]},
        schema=[("bar", pl.Int8), ("foo", pl.Boolean), ("baz", pl.String)],
    )
    if lazy:
        df = df.lazy()

    TestSchema.coerce(df)


@pytest.mark.parametrize("schema", [TestInheritedSchema, TestMultipleInheritance])
@pytest.mark.parametrize("lazy", [True, False])
def test_coercion_valid_inherited_schema_passes(
    lazy: bool, schema: DataFrameSchema
) -> None:
    df: pl.DataFrame | pl.LazyFrame = pl.DataFrame(
        {"bar": [1], "foo": [False], "baz": ["a"], "foobar": ["a"]},
        schema=[
            ("bar", pl.Int8),
            ("foo", pl.Boolean),
            ("baz", pl.String),
            ("foobar", pl.String),
        ],
    )
    if lazy:
        df = df.lazy()

    schema.coerce(df)


@pytest.mark.parametrize("lazy", [True, False])
def test_second_inheritance_schema_passes(lazy) -> None:
    df: pl.DataFrame | pl.LazyFrame = pl.DataFrame(
        {"bar": [1], "foo": [False], "baz": ["a"], "boo": [True], "foobar": ["a"]},
        schema=[
            ("bar", pl.Int8),
            ("foo", pl.Boolean),
            ("baz", pl.String),
            ("boo", pl.Boolean),
            ("foobar", pl.String),
        ],
    )

    if lazy:
        df = df.lazy()

    TestSecondInheritedSchema.coerce(df)


def test_cannot_create_schema_of_invalid_type() -> None:
    with pytest.raises(TypeError):

        class BadSchema(DataFrameSchema):
            foo = Column(int)


def test_schema_attribures_must_be_column_type() -> None:
    with pytest.raises(TypeError):

        class BadSchema(DataFrameSchema):
            foo = pl.Int8


def test_schema_disallows_type_annotations() -> None:
    with pytest.raises(TypeError):

        class BadSchema(DataFrameSchema):
            foo: pl.Int8


def test_schema_allow_missing_works() -> None:
    coerced = TestSchema.coerce(pl.DataFrame({"cow": ["baz"]}), allow_missing=True)
    assert coerced.equals(
        pl.DataFrame(
            {TestSchema.foo: [None], TestSchema.bar: [None]}, schema=TestSchema.schema()
        )
    )


class KeyValueSchema(DataFrameSchema):
    key = Column(pl.Int64)
    value = Column(pl.String)

    @data_quality_check
    @classmethod
    def key_is_unique(cls, df: pl.DataFrame) -> None:
        primary_key(df, [cls.key])


def test_data_check_unique_key_passes() -> None:
    df = pl.DataFrame({"key": [1, 2, 3], "value": ["a", "b", "c"]})
    validated = KeyValueSchema.perform_data_quality_checks(df)
    assert len(validated) == 3


def test_data_check_duplicate_key_raises() -> None:
    df = pl.DataFrame({"key": [1, 2, 2], "value": ["a", "b", "c"]})
    with pytest.raises(
        ValueError, match="combination of columns \\(key\\) must be unique"
    ):
        KeyValueSchema.perform_data_quality_checks(df)


class CompositeKeySchema(DataFrameSchema):
    col1 = Column(pl.Int64)
    col2 = Column(pl.String)
    value = Column(pl.String)

    @data_quality_check
    @classmethod
    def composite_key_is_unique(cls, df: pl.DataFrame) -> None:
        primary_key(df, [cls.col1, cls.col2])


def test_composite_primary_key_passes() -> None:
    df = pl.DataFrame(
        {"col1": [1, 1, 2], "col2": ["a", "b", "a"], "value": ["x", "y", "z"]}
    )
    validated = CompositeKeySchema.perform_data_quality_checks(df)
    assert len(validated) == 3


def test_composite_primary_key_duplicate_raises() -> None:
    df = pl.DataFrame(
        {"col1": [1, 1, 2], "col2": ["a", "a", "a"], "value": ["x", "y", "z"]}
    )
    with pytest.raises(
        ValueError, match="combination of columns \\(col1, col2\\) must be unique"
    ):
        CompositeKeySchema.perform_data_quality_checks(df)


class NonClassMethodDataQualityCheckSchema(DataFrameSchema):
    col = Column(pl.Int64)

    @data_quality_check
    def data_quality_check(self, df: pl.DataFrame) -> None:
        return df


def test_non_class_method_data_quality_check_raises() -> None:
    df = pl.DataFrame({"col": [1, 2, 3]})
    with pytest.raises(TypeError):
        NonClassMethodDataQualityCheckSchema.perform_data_quality_checks(df)


def test_upcast_numeric() -> None:
    class Schema(DataFrameSchema):
        icol = Column(pl.Int64)
        fcol = Column(pl.Float64)

    df = pl.DataFrame(
        {"icol": [1, 2, 3], "fcol": [1, 2, 3]},
        schema=[("icol", pl.Int16), ("fcol", pl.Float16)],
    )

    validated = Schema.coerce(df, upcast_numeric=True)
    assert validated["fcol"].dtype == pl.Float64
    assert validated["icol"].dtype == pl.Int64

    with pytest.raises(TypeError):
        validated = Schema.coerce(df, upcast_numeric=False)
