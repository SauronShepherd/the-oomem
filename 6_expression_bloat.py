from pyspark.sql import functions as F

from common import banner, get_spark, measure_optimization, parse_common_args


def deep_expression(depth: int):
    expression = F.col('id').cast('long')
    for index in range(depth):
        expression = (expression + F.lit(index + 1)).cast('long')
    return expression


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
    args = parse_common_args('Lab 6 - Expression bloat')
    spark = get_spark(args)
    try:
        banner('Lab 6 — The expression tree that keeps getting copied')
        compact = spark.range(32).select(deep_expression(args.depth).alias('value'))
        repeated = spark.range(32).select(
            *[deep_expression(args.depth).alias(f'value_{index}')
              for index in range(args.width)]
        )

        print(f'one deep expression (depth {args.depth}): {plan_metrics("one deep expression", compact)}')
        print(f'{args.width} copies of that expression: {plan_metrics("repeated expressions", repeated)}')

        equivalent = spark.range(32).select(deep_expression(args.depth).alias('value'))
        left_plan = compact._jdf.queryExecution().analyzed()
        right_plan = equivalent._jdf.queryExecution().analyzed()
        try:
            semantic_result = left_plan.sameResult(right_plan)
        except Exception as exc:  # Spark does not expose the expression method on every JVM wrapper.
            semantic_result = f'unavailable through this Py4J wrapper ({type(exc).__name__})'
        print('semantic equivalence of equivalent analyzed logical plans:', semantic_result)
        print('canonicalized plan lengths:',
              len(left_plan.canonicalized().toString()),
              len(right_plan.canonicalized().toString()))
        print('Conclusion: row count stays tiny while canonicalization and semantic comparison still see every expression node.')
        print('Repeated deterministic expressions are a plan-shape cost; they are not made free by returning one scalar column per alias.')
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
