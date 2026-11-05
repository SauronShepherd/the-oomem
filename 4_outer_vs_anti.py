from collections import Counter
from pyspark.sql import functions as F

from common import banner, get_spark, measure_optimization, parse_common_args


def inspect(label, frame):
    metrics = measure_optimization(frame, label)
    tree = frame._jdf.queryExecution().optimizedPlan().treeString()
    shape = {
        'nodes': metrics['optimized_nodes'],
        'characters': metrics['optimized_characters'],
    }
    return metrics['optimization_seconds'], shape, tree


def main() -> None:
    args = parse_common_args('Lab 4 - Outer versus anti planning')
    spark = get_spark(args)
    try:
        banner('Lab 4 — Outer versus anti: a bounded planning comparison')
        master_size = max(1, int(args.rows * 5 / 6))
        master = spark.range(master_size).select('id', (F.col('id') % 5).alias('group'))
        incoming = spark.range(args.rows).select('id', (F.col('id') % 5).alias('group'))
        m, d = master.alias('m'), incoming.alias('d')

        outer = (m.join(d, F.col('m.id') == F.col('d.id'), 'right')
                 .filter(F.col('m.id').isNull())
                 .select('d.id', 'd.group'))
        anti = d.join(m, F.col('d.id') == F.col('m.id'), 'left_anti').select('d.id', 'd.group')

        for name, frame in [('outer', outer), ('anti', anti)]:
            seconds, shape, tree = inspect(name, frame)
            print(f'{name}: first optimizedPlan() {seconds:.4f}s, shape={shape}')
            print(tree[:1000])

        outer_rows, anti_rows = outer.collect(), anti.collect()
        assert Counter(map(tuple, outer_rows)) == Counter(map(tuple, anti_rows))
        assert outer.schema == anti.schema
        print(f'rows: {len(outer_rows)} (outer), {len(anti_rows)} (anti)')
        print('Conclusion: this sample compares logical shapes; one local timing is not a benchmark.')
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
