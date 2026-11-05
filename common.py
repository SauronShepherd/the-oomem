"""Shared helpers for the OOMem Spark labs.

The labs deliberately keep the data small and the measurements local.  Their
question is always the same: how much driver work and heap does Catalyst spend
before an executor can run anything?  The measurements are diagnostics, not a
production benchmark.
"""

import argparse
import os
import sys
import time
from collections import Counter
from typing import Callable

from pyspark.sql import SparkSession, DataFrame


# Spark's Python worker launcher reads these environment variables on Windows
# before it consults the session builder. Pin both sides to this venv.
os.environ.setdefault('PYSPARK_PYTHON', sys.executable)
os.environ.setdefault('PYSPARK_DRIVER_PYTHON', sys.executable)


def parse_common_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--master', default='local[2]', help='Spark master used for the lab')
    parser.add_argument('--shuffle-partitions', type=int, default=2)
    parser.add_argument('--rows', type=int, default=240,
                        help='Synthetic rows for the bounded comparison lab')
    parser.add_argument('--alias-depth', type=int, default=4,
                        help='Maximum projection depth for the constraint lab')
    parser.add_argument('--depth', type=int, default=20,
                        help='Depth for nested-project and expression labs')
    parser.add_argument('--branches', type=int, default=8,
                        help='Number of branches for the union fan-out lab')
    parser.add_argument('--width', type=int, default=8,
                        help='Number of repeated expressions in the expression lab')
    parser.add_argument('--app-name', default=description)
    return parser.parse_args()


def get_spark(args: argparse.Namespace) -> SparkSession:
    return create_spark_session(
        app_name=args.app_name,
        master=args.master,
        shuffle_partitions=args.shuffle_partitions,
    )


def create_spark_session(
    app_name: str,
    master: str = 'local[2]',
    shuffle_partitions: int = 2,
) -> SparkSession:
    """Create the bounded local Spark session shared by scripts and notebooks."""
    spark = (SparkSession.builder
             .master(master)
             .appName(app_name)
             .config('spark.ui.enabled', 'false')
             .config('spark.sql.shuffle.partitions', str(shuffle_partitions))
             # Keep local Python workers on the same interpreter as the lab.
             # This avoids Windows' Microsoft Store `python` alias hijacking
             # the worker when the project is run from its virtualenv.
             .config('spark.pyspark.python', sys.executable)
             .config('spark.pyspark.driver.python', sys.executable)
             .getOrCreate())
    spark.sparkContext.setLogLevel('WARN')
    return spark


def banner(title: str) -> None:
    print('\n' + '=' * 88)
    print(title)
    print('=' * 88)


def timed(label: str, fn: Callable[[], object]):
    start = time.perf_counter()
    value = fn()
    elapsed = time.perf_counter() - start
    print(f'{label}: {elapsed:.4f}s')
    return value, elapsed


def row_counter(frame: DataFrame) -> Counter:
    """Collect a tiny lab frame as a multiset, preserving duplicate semantics."""
    return Counter(map(tuple, frame.collect()))


def assert_same_rows(left: DataFrame, right: DataFrame) -> None:
    assert left.schema == right.schema, (left.schema, right.schema)
    assert row_counter(left) == row_counter(right)


def plan_shape(frame: DataFrame) -> dict[str, int]:
    tree = frame._jdf.queryExecution().optimizedPlan().treeString()
    return {
        'nodes': len(tree.splitlines()),
        'characters': len(tree),
    }


def jvm_heap(frame: DataFrame) -> dict[str, float]:
    """Return the local driver's JVM heap counters without changing Spark config."""
    runtime = frame.sparkSession._jvm.java.lang.Runtime.getRuntime()
    mib = 1024 * 1024
    used = runtime.totalMemory() - runtime.freeMemory()
    return {
        'heap_used_mib': round(used / mib, 1),
        'heap_committed_mib': round(runtime.totalMemory() / mib, 1),
        'heap_max_mib': round(runtime.maxMemory() / mib, 1),
    }


def measure_optimization(frame: DataFrame, label: str) -> dict[str, float | int]:
    """Measure only logical optimization and expose its driver-side footprint."""
    query = frame._jdf.queryExecution()
    before = jvm_heap(frame)
    started = time.perf_counter()
    optimized = query.optimizedPlan()
    tree = optimized.treeString()
    elapsed = time.perf_counter() - started
    after = jvm_heap(frame)
    result: dict[str, float | int] = {
        'optimization_seconds': round(elapsed, 4),
        'optimized_nodes': len(tree.splitlines()),
        'optimized_characters': len(tree),
        'heap_used_before_mib': before['heap_used_mib'],
        'heap_used_after_mib': after['heap_used_mib'],
        'heap_max_mib': after['heap_max_mib'],
    }
    print(f'{label}: {result}')
    return result


def show_plan_shape(label: str, frame: DataFrame) -> dict[str, int]:
    shape = plan_shape(frame)
    print(f'{label}: {shape}')
    return shape
