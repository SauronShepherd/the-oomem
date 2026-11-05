from pyspark.sql import functions as F

from common import banner, get_spark, measure_optimization, parse_common_args, row_counter


def main() -> None:
    args = parse_common_args('Lab 2 - Nulls and duplicates')
    spark = get_spark(args)
    try:
        banner('Lab 2 — Nulls, duplicates, and the false exorcism')
        master = spark.createDataFrame(
            [(1, None), (2, 'x'), (None, 'null-key')],
            ['k', 'value'],
        )
        incoming = spark.createDataFrame(
            [(1, 'new'), (3, 'a'), (3, 'a'), (None, 'null-key')],
            ['k', 'detail'],
        )
        m, d = master.alias('m'), incoming.alias('d')

        ordinary = d.join(m, F.col('d.k') == F.col('m.k'), 'left_anti')
        null_safe = d.join(m, F.col('d.k').eqNullSafe(F.col('m.k')), 'left_anti')
        measure_optimization(ordinary, 'ordinary left_anti')
        measure_optimization(null_safe, 'null-safe left_anti')
        print('ordinary equality:', ordinary.collect())
        print('null-safe equality:', null_safe.collect())
        assert row_counter(ordinary) == row_counter(
            spark.createDataFrame([(3, 'a'), (3, 'a'), (None, 'null-key')], ['k', 'detail'])
        )
        assert row_counter(null_safe) == row_counter(
            spark.createDataFrame([(3, 'a'), (3, 'a')], ['k', 'detail'])
        )

        null_safe_outer = (m.join(d, F.col('m.k').eqNullSafe(F.col('d.k')), 'right')
                           .filter(F.col('m.k').isNull())
                           .select('d.k', 'd.detail'))
        assert row_counter(null_safe_outer) == row_counter(ordinary)
        assert row_counter(null_safe_outer) != row_counter(null_safe)

        nullable_non_key = (m.join(d, F.col('m.k') == F.col('d.k'), 'right')
                            .filter(F.col('m.k').isNull() | F.col('m.value').isNull())
                            .select('d.k', 'd.detail'))
        measure_optimization(nullable_non_key, 'right_outer with nullable non-key filter')
        print('null-safe outer + null filter:', null_safe_outer.collect())
        print('nullable non-key filter:', nullable_non_key.collect())
        assert row_counter(nullable_non_key) == row_counter(ordinary) + row_counter(
            spark.createDataFrame([(1, 'new')], ['k', 'detail'])
        )
        print('Conclusion: equality choice and nullable filters change the rewrite contract.')
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
