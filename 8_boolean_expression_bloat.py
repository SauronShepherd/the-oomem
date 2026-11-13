"""Lab 8: build nested CASE and boolean-OR expression trees.

Expected result: a tiny range produces a deep analyzed expression tree; the
pressure comes from expression shape, not join cardinality or row volume.
"""

from pyspark.sql import functions as F

from common import banner, get_spark, measure_optimization, parse_common_args


def nested_case(depth: int):
    expression = F.lit('leaf')
    for index in range(depth):
        expression = F.when(
            (F.col('id') % (index + 2)) == 0,
            F.lit(f'bucket-{index}'),
        ).otherwise(expression)
    return expression


def nested_boolean(depth: int):
    predicate = F.lit(False)
    for index in range(depth):
        predicate = predicate | ((F.col('id') % (index + 2)) == 0)
    return predicate


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
    args = parse_common_args('Lab 8 - Boolean expression bloat')
    spark = get_spark(args)
    try:
        banner('Lab 8 — The nested predicate staircase')
        frame = spark.range(32).select(
            'id',
            nested_case(args.depth).alias('classification'),
            nested_boolean(args.depth).alias('matches'),
        )
        metrics = plan_metrics('nested CASE + boolean predicate', frame)
        print(f'nested CASE and boolean predicate depth {args.depth}: {metrics}')
        print('Conclusion: expression simplification can reduce work, but it must first parse, resolve, canonicalize, and inspect the entire tree.')
        print('This is expression depth rather than join cardinality: the input is tiny, the compiler tree is not.')
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
