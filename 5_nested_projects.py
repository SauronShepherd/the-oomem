from pyspark.sql import functions as F

from common import banner, get_spark, measure_optimization, parse_common_args


def plan_metrics(label, frame):
    query = frame._jdf.queryExecution()
    analyzed = query.analyzed().treeString()
    measured = measure_optimization(frame, label)
    return {
        'analyzed_nodes': len(analyzed.splitlines()),
        'analyzed_characters': len(analyzed),
        **measured,
    }


def main() -> None:
    args = parse_common_args('Lab 5 - Nested projects')
    spark = get_spark(args)
    try:
        banner('Lab 5 — The withColumn staircase')
        base = spark.range(32)

        chained = base
        for index in range(args.depth):
            chained = chained.withColumn(
                f'value_{index}',
                (F.col('id') + F.lit(index)).cast('long'),
            )

        batched = base.select(
            'id',
            *[(F.col('id') + F.lit(index)).cast('long').alias(f'value_{index}')
              for index in range(args.depth)],
        )

        chained_metrics = plan_metrics('withColumn loop', chained)
        batched_metrics = plan_metrics('one select', batched)
        print(f'withColumn loop ({args.depth} steps): {chained_metrics}')
        print(f'one select ({args.depth} expressions): {batched_metrics}')
        assert chained_metrics['analyzed_nodes'] > batched_metrics['analyzed_nodes']
        print('Conclusion: the public API looks linear, but each chained call adds another logical Project.')
        print('The optimizer may collapse those Projects later; the driver still has to analyze and inspect the chain first.')
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
