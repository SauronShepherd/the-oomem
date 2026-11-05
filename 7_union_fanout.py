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
    args = parse_common_args('Lab 7 - Union fan-out')
    spark = get_spark(args)
    try:
        banner('Lab 7 — The union fan-out')
        base = spark.range(args.rows).withColumn('bucket', F.col('id') % args.branches)
        branches = [
            base.filter(F.col('bucket') == index)
                .select('id', 'bucket', F.lit(index).alias('branch'))
            for index in range(args.branches)
        ]
        fanout = branches[0]
        for branch in branches[1:]:
            fanout = fanout.unionByName(branch)

        metrics = plan_metrics('union fan-out', fanout)
        print(f'{args.branches} repeated branches over {args.rows} synthetic rows: {metrics}')
        assert metrics['analyzed_nodes'] >= args.branches
        print('Conclusion: no join is required for plan blow-up; duplicated branch trees make every rule walk a wider graph.')
        print('This is a breadth problem: reuse in Python does not deduplicate the logical plan automatically.')
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
