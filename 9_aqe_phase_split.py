"""Lab 9: separate AQE's physical wrapper from logical optimization.

Expected result: AQE changes the physical/adaptive phase while the analyzed
and optimized logical shapes remain comparable for the same query.
"""

from pyspark.sql import functions as F

from common import banner, get_spark, measure_optimization, parse_common_args


def run_case(args, adaptive: bool) -> dict[str, object]:
    spark = get_spark(args)
    try:
        spark.conf.set('spark.sql.adaptive.enabled', str(adaptive).lower())
        spark.conf.set('spark.sql.adaptive.coalescePartitions.enabled', 'true')
        left = spark.range(args.rows).withColumn('key', F.col('id') % args.branches)
        right = spark.range(args.rows).withColumn('key', F.col('id') % args.branches)
        frame = left.join(right, 'key').groupBy('key').count()
        query = frame._jdf.queryExecution()

        analyzed = query.analyzed()
        analyzed_tree = analyzed.treeString()
        optimization = measure_optimization(frame, f'AQE={adaptive} logical optimizer')
        initial_physical = query.executedPlan().treeString()
        frame.collect()
        final_physical = query.executedPlan().treeString()
        return {
            'adaptive': adaptive,
            'analyzed_nodes': len(analyzed_tree.splitlines()),
            'analyzed_characters': len(analyzed_tree),
            'optimization_seconds': optimization['optimization_seconds'],
            'optimized_nodes': optimization['optimized_nodes'],
            'optimized_characters': optimization['optimized_characters'],
            'heap_used_after_mib': optimization['heap_used_after_mib'],
            'initial_has_adaptive_wrapper': 'AdaptiveSparkPlan' in initial_physical,
            'final_has_adaptive_wrapper': 'AdaptiveSparkPlan' in final_physical,
            'final_physical_nodes': len(final_physical.splitlines()),
        }
    finally:
        spark.stop()


def main() -> None:
    args = parse_common_args('Lab 9 - AQE phase split')
    banner('Lab 9 — AQE changes the physical phase, not the logical possession')
    aqe_on = run_case(args, True)
    aqe_off = run_case(args, False)
    print('AQE on :', aqe_on)
    print('AQE off:', aqe_off)
    assert aqe_on['optimized_nodes'] == aqe_off['optimized_nodes']
    assert aqe_on['initial_has_adaptive_wrapper'] is True
    assert aqe_on['final_has_adaptive_wrapper'] is True
    assert aqe_off['initial_has_adaptive_wrapper'] is False
    assert aqe_off['final_has_adaptive_wrapper'] is False
    print('Conclusion: compare optimized logical shape first; session-local expression IDs can change text length, while AQE still only adds the physical adaptive wrapper and stage work.')


if __name__ == '__main__':
    main()
