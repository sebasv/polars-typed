from __future__ import annotations

import inspect
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Literal,
    Self,
    Sequence,
    TypeAlias,
    cast,
    overload,
)

if TYPE_CHECKING:
    import numpy as np
    from polars._typing import (
        FrameInitTypes,  # type: ignore[reportUnknownVariableType] # otherwise we need pyarrow stubs
        IntoExpr,
        IntoExprColumn,
        Orientation,
        SchemaDefinition,
        SchemaDict,
    )
    from polars.lazyframe.in_process import InProcessQuery

import polars as pl
from polars.datatypes import N_INFER_DEFAULT
from polars.datatypes.classes import DataTypeClass

PolarsColumnType: TypeAlias = pl.DataType | DataTypeClass


def data_quality_check(func: Callable[..., None]) -> Callable[..., None]:
    func._is_data_quality_check = True  # type: ignore[reportFunctionMemberAccess]
    return func


def primary_key(df: pl.DataFrame, columns: list[str | Column] | str | Column) -> None:
    if isinstance(columns, (str, Column)):
        columns = [columns]
    column_names = [str(col) if isinstance(col, Column) else col for col in columns]
    if df.select(column_names).is_duplicated().any():
        columns_str = ", ".join(column_names)
        raise ValueError(f"combination of columns ({columns_str}) must be unique")


class Column(str):
    def __new__(cls, t: PolarsColumnType, _name: str = "") -> Self:
        return super().__new__(cls, _name)

    def __init__(self, t: PolarsColumnType, _name: str = "") -> None:
        # Test for polars datatype, possibly uninitialized (both `pl.String` and pl.String()` are valid)
        if not isinstance(t, PolarsColumnType):  # type: ignore[reportUnnecessaryIsInstance]
            raise TypeError(
                f"Can only construct Column of `polars.DataType`, not of type {t}"
            )
        self._t = t

    @property
    def dtype(self) -> PolarsColumnType:
        return self._t

    @property
    def col(self) -> pl.Expr:
        """Return a polars column expression for this column."""
        return pl.col(self)


class _Meta(type):
    # A metaclass is needed because we want to rely on type annotations, we don't want to instantiate schema types.
    def __new__(
        cls, name: str, bases: tuple[type, ...], namespace: dict[str, Any]
    ) -> _Meta:
        super_class_schema_items: list[tuple[str, PolarsColumnType]] = []
        super_class_data_quality_checks: list[Callable[..., None]] = []

        if len(bases) == 1 and bases[0] is _DataFrameSchema:
            # Note: this requires a special case because the type `DataFrameSchema` on the next line
            # is not yet instantiated when the class `DataFrameSchema` itself is instantiated.
            pass
        elif any(parent_schemas := [b for b in bases if b is not DataFrameSchema]):
            # We retrieve the polars schemas from the parent classes.
            parent_schemas: list[type[DataFrameSchema]]
            super_class_schemas: list[pl.Schema] = [
                b._schema  # pyright: ignore[reportPrivateUsage]
                for b in parent_schemas
                if hasattr(b, "_schema")
            ]  # type: ignore[reportPrivateUsage]
            super_class_schema_items = [
                (k, v) for schema in super_class_schemas for k, v in schema.items()
            ]
            super_class_data_quality_checks = [
                check
                for b in parent_schemas
                if hasattr(b, "_data_quality_checks")
                for check in b._data_quality_checks  # type: ignore[reportPrivateUsage]
            ]

        # We first use the super class schema items, such that in the final schema the super class column names are put first.
        schema = pl.Schema(
            super_class_schema_items
            + [(k, v._t) for k, v in namespace.items() if isinstance(v, Column)]  # type: ignore[reportPrivateUsage]
        )

        data_quality_checks = super_class_data_quality_checks + [
            check
            for check in namespace.values()
            if hasattr(check, "_is_data_quality_check")
        ]

        if (
            "__annotations__" in namespace
            or "__annotate__" in namespace
            or "__annotate_func__" in namespace
        ):
            raise TypeError(
                f"Found type annotations in schema {name}. Use assignment (=) to specify columns"
            )

        allowed_properties = [
            k
            for k in namespace
            if k.endswith("__")
            or inspect.isfunction(namespace[k])
            or isinstance(namespace[k], (classmethod, staticmethod, property))
        ]
        unexpected_properties = [
            k for k in namespace if k not in allowed_properties and k not in schema
        ]
        if len(unexpected_properties) > 0:
            raise TypeError(
                f"Schema {schema} contains properties without polars datatypes."
                f"Make sure that all columns are specified as `Column(pl.DataType)`: {unexpected_properties}"
            )
        # Set attributes as string. This allows us to use the attributes as column names.
        namespace.setdefault("__annotations__", {})
        for k, v in schema.items():
            namespace[k] = Column(v, k)
        # Save the schema as a separate attribute
        namespace["_schema"] = schema
        namespace["_data_quality_checks"] = data_quality_checks

        return super().__new__(cls, name, bases, namespace)


class _DataFrameSchema:
    _schema: pl.Schema
    _data_quality_checks: list[Callable[..., None]]

    @classmethod
    def schema(cls) -> pl.Schema:
        return cls._schema

    @classmethod
    def data_quality_checks(cls) -> list[Callable[..., None]]:
        return cls._data_quality_checks

    @overload
    @classmethod
    def validate(cls, df: pl.LazyFrame) -> LazyFrame[Self]: ...

    @overload
    @classmethod
    def validate(cls, df: pl.DataFrame) -> DataFrame[Self]: ...

    @classmethod
    def validate(
        cls, df: pl.DataFrame | pl.LazyFrame
    ) -> DataFrame[Self] | LazyFrame[Self]:
        schema = cls._schema
        df_schema = df.collect_schema()
        if schema != df_schema:
            if sorted(schema) == sorted(df_schema) and all(
                df_schema[k] == schema[k] for k in schema
            ):
                raise TypeError(
                    f"Schema validation failed for {cls}. The column order is incorrect. \nExpected: {schema}\nReceived: {df_schema}"
                )
            redundant_columns = set(df_schema) - set(schema)
            missing_columns = set(schema) - set(df_schema)
            columns_with_incorrect_types = [
                f"column {k} should be {v1} but is {v2}"
                for k, v1 in schema.items()
                if k in df_schema and (v2 := df_schema[k]) != v1
            ]
            raise TypeError(
                f"Schema validation failed for {cls}.\n"
                "Redundant columns:\n"
                f"{redundant_columns}\n"
                "Missing columns:\n"
                f"{missing_columns}\n"
                "Columns with incorrect types:\n"
                f"{'\n'.join(columns_with_incorrect_types)}"
            )
        if isinstance(df, pl.DataFrame):
            return cast(DataFrame[Self], df)
        return cast(LazyFrame[Self], df)

    @overload
    @classmethod
    def coerce(
        cls, df: pl.DataFrame, allow_missing: bool = False, upcast_numeric: bool = True
    ) -> DataFrame[Self]: ...

    @overload
    @classmethod
    def coerce(
        cls, df: pl.LazyFrame, allow_missing: bool = False, upcast_numeric: bool = True
    ) -> LazyFrame[Self]: ...

    @classmethod
    def coerce(
        cls,
        df: pl.DataFrame | pl.LazyFrame,
        allow_missing: bool = False,
        upcast_numeric: bool = True,
    ) -> DataFrame[Self] | LazyFrame[Self]:
        if allow_missing:
            df = df.with_columns(
                **{
                    col: pl.lit(None).cast(typ)
                    for col, typ in cls._schema.items()
                    if col not in df
                }
            )
        df = df.select(
            [
                pl.col(c).cast(dtype) if (dtype.is_numeric() and upcast_numeric) else c
                for c, dtype in cls._schema.items()
                if c in df
            ]
        )
        return cls.validate(df)

    @classmethod
    def empty(cls) -> DataFrame[Self]:
        return cast(DataFrame[Self], cls._schema.to_frame())

    @classmethod
    def perform_data_quality_checks(cls, df: pl.DataFrame) -> DataFrame[Self]:
        validated_df = cls.validate(df)
        for check in cls._data_quality_checks:
            if not isinstance(check, classmethod):
                raise TypeError("Every data check should be a classmethod")

            check.__func__(cls, validated_df)  # type: ignore[reportFunctionMemberAccess]
        return validated_df


#  specify metaclass in a child class. If we specify metaclass directly on _DataFrameSchema
# then _DataFrameSchema is treated as a schema itself.
class DataFrameSchema(_DataFrameSchema, metaclass=_Meta): ...


class DataFrame[TDataFrameSchema: _DataFrameSchema](pl.DataFrame):
    def __init__(
        self,
        typed_schema: type[TDataFrameSchema],
        data: FrameInitTypes | None = None,  # type: ignore[reportUnknownParameterType] # otherwise we need pyarrow stubs
        # `schema` and `schema_overrides` parameters are left in order to maintain signature parity with superclass
        # __init__, but should be avoided; instead just provide the `typed_schema`.
        schema: SchemaDefinition | None = None,
        *,
        schema_overrides: SchemaDict | None = None,
        strict: bool = True,
        orient: Orientation | None = None,
        infer_schema_length: int | None = N_INFER_DEFAULT,
        nan_to_null: bool = False,
    ) -> None:
        # If `schema` is provided, pass it and `schema_overrides` into the superclass __init__, then let the
        # validate(...) at the end decide whether the overrides add up to the typed schema.
        if schema is None:
            schema = typed_schema.schema()

        super().__init__(  # type: ignore[reportUnknownMemberType] # otherwise we need pyarrow stubs
            data,
            schema,
            schema_overrides=schema_overrides,
            strict=strict,
            orient=orient,
            infer_schema_length=infer_schema_length,
            nan_to_null=nan_to_null,
        )

        typed_schema.validate(self)

    def lazy(self) -> LazyFrame[TDataFrameSchema]:
        return cast(LazyFrame[TDataFrameSchema], pl.DataFrame.lazy(self))

    def filter(
        self,
        *predicates: (
            IntoExprColumn
            | Iterable[IntoExprColumn]
            | bool
            | list[bool]
            | np.ndarray[Any, Any]
        ),
        **constraints: Any,
    ) -> DataFrame[TDataFrameSchema]:
        return cast(
            DataFrame[TDataFrameSchema],
            pl.DataFrame.filter(self, *predicates, **constraints),
        )

    def sort(
        self,
        by: IntoExpr | Iterable[IntoExpr],
        *more_by: IntoExpr,
        descending: bool | Sequence[bool] = False,
        nulls_last: bool | Sequence[bool] = False,
        multithreaded: bool = True,
        maintain_order: bool = False,
    ) -> DataFrame[TDataFrameSchema]:
        return cast(
            DataFrame[TDataFrameSchema],
            pl.DataFrame.sort(
                self,
                by,
                *more_by,
                descending=descending,
                nulls_last=nulls_last,
                multithreaded=multithreaded,
                maintain_order=maintain_order,
            ),
        )


class LazyFrame[TDataFrameSchema: _DataFrameSchema](pl.LazyFrame):
    @overload
    def collect(
        self, *, background: Literal[True], **kwargs: Any
    ) -> InProcessQuery: ...

    @overload
    def collect(
        self, *, background: Literal[False] = False, **kwargs: Any
    ) -> DataFrame[TDataFrameSchema]: ...

    def collect(
        self, *, background: bool = False, **kwargs: Any
    ) -> DataFrame[TDataFrameSchema] | InProcessQuery:
        if background:
            return pl.LazyFrame.collect(self, background=background, **kwargs)
        return cast(
            DataFrame[TDataFrameSchema],
            pl.LazyFrame.collect(self, background=background, **kwargs),
        )

    def filter(
        self,
        *predicates: (
            IntoExprColumn
            | Iterable[IntoExprColumn]
            | bool
            | list[bool]
            | np.ndarray[Any, Any]
        ),
        **constraints: Any,
    ) -> LazyFrame[TDataFrameSchema]:
        return cast(
            LazyFrame[TDataFrameSchema],
            pl.LazyFrame.filter(self, *predicates, **constraints),
        )

    def sort(
        self,
        by: IntoExpr | Iterable[IntoExpr],
        *more_by: IntoExpr,
        descending: bool | Sequence[bool] = False,
        nulls_last: bool | Sequence[bool] = False,
        maintain_order: bool = False,
        multithreaded: bool = True,
    ) -> LazyFrame[TDataFrameSchema]:
        return cast(
            LazyFrame[TDataFrameSchema],
            pl.LazyFrame.sort(
                self,
                by,
                *more_by,
                descending=descending,
                nulls_last=nulls_last,
                maintain_order=maintain_order,
                multithreaded=multithreaded,
            ),
        )
