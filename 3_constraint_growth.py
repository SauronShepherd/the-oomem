from pyspark.sql import functions as F

from common import banner, get_spark, measure_optimization, parse_common_args


def main() -> None:
    args = parse_common_args('Lab 3 - Constraint growth')
    spark = get_spark(args)
    try:
        banner('Lab 3 — Alias constraints and the growing plan')
        base = spark.range(64).select(
            (F.col('id') + 1).alias('k1'),
            (F.col('id') + 2).alias('k2'),
        )
        filtered = base.filter((F.col('k1') + F.col('k2')) >= 0)

        print('depth | analyzed constraints | optimized nodes | optimized characters')
        print('-' * 72)
        for depth in range(args.alias_depth + 1):
            frame = filtered
            for index in range(depth):
                frame = frame.select(
                    '*',
                    F.col('k1').alias(f'k1_alias_{index}'),
                    F.col('k2').alias(f'k2_alias_{index}'),
                )
            query = frame._jdf.queryExecution()
            constraints = query.analyzed().constraints().size()
            measured = measure_optimization(frame, f'alias depth {depth} optimizer')
            shape = {
                'nodes': measured['optimized_nodes'],
                'characters': measured['optimized_characters'],
            }
            print(f'{depth:5d} | {constraints:20d} | {shape["nodes"]:16d} | {shape["characters"]:22d}')
            assert 'Project' in query.optimizedPlan().treeString()

        print('Conclusion: aliases expose equivalent forms to Catalyst; this is a bounded inspection, not a leak proof.')
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
