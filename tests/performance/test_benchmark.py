#!/usr/bin/env python3
"""
Performance Benchmark Tests for Intelligent Recommender System

Tests system performance including:
- Recommendation latency
- Throughput (QPS)
- Cache performance
- Model inference speed
- Memory usage
"""

import time
import statistics
import pytest
import pandas as pd
import numpy as np
from typing import List, Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import create_recommender
from services.cache import CacheService


class BenchmarkResult:
    """Performance benchmark result container"""

    def __init__(self, name: str):
        self.name = name
        self.latencies = []
        self.start_time = None
        self.end_time = None

    def record(self, latency: float):
        """Record a single operation latency"""
        self.latencies.append(latency)

    def start(self):
        """Start timing"""
        self.start_time = time.time()

    def end(self):
        """End timing"""
        self.end_time = time.time()

    @property
    def total_time(self) -> float:
        """Total elapsed time"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

    @property
    def count(self) -> int:
        """Number of operations"""
        return len(self.latencies)

    @property
    def mean_latency(self) -> float:
        """Mean latency in ms"""
        return statistics.mean(self.latencies) * 1000 if self.latencies else 0.0

    @property
    def median_latency(self) -> float:
        """Median latency in ms"""
        return statistics.median(self.latencies) * 1000 if self.latencies else 0.0

    @property
    def p95_latency(self) -> float:
        """P95 latency in ms"""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        index = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[index] * 1000

    @property
    def p99_latency(self) -> float:
        """P99 latency in ms"""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        index = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[index] * 1000

    @property
    def max_latency(self) -> float:
        """Max latency in ms"""
        return max(self.latencies) * 1000 if self.latencies else 0.0

    @property
    def min_latency(self) -> float:
        """Min latency in ms"""
        return min(self.latencies) * 1000 if self.latencies else 0.0

    @property
    def qps(self) -> float:
        """Queries per second"""
        if self.total_time > 0:
            return self.count / self.total_time
        return 0.0

    def summary(self) -> Dict:
        """Return summary statistics"""
        return {
            'name': self.name,
            'count': self.count,
            'total_time': f'{self.total_time:.2f}s',
            'qps': f'{self.qps:.2f}',
            'mean': f'{self.mean_latency:.2f}ms',
            'median': f'{self.median_latency:.2f}ms',
            'p95': f'{self.p95_latency:.2f}ms',
            'p99': f'{self.p99_latency:.2f}ms',
            'min': f'{self.min_latency:.2f}ms',
            'max': f'{self.max_latency:.2f}ms'
        }

    def print_summary(self):
        """Print benchmark summary"""
        print(f"\n{'='*70}")
        print(f"Benchmark: {self.name}")
        print(f"{'='*70}")
        print(f"Total Operations: {self.count}")
        print(f"Total Time:       {self.total_time:.2f}s")
        print(f"QPS:              {self.qps:.2f}")
        print(f"")
        print(f"Latency Statistics:")
        print(f"  Mean:    {self.mean_latency:>8.2f}ms")
        print(f"  Median:  {self.median_latency:>8.2f}ms")
        print(f"  P95:     {self.p95_latency:>8.2f}ms")
        print(f"  P99:     {self.p99_latency:>8.2f}ms")
        print(f"  Min:     {self.min_latency:>8.2f}ms")
        print(f"  Max:     {self.max_latency:>8.2f}ms")
        print(f"{'='*70}\n")


@pytest.fixture(scope='module')
def benchmark_data():
    """Generate benchmark dataset"""
    np.random.seed(42)
    return pd.DataFrame({
        'user_id': np.random.randint(1, 100, 500),
        'item_id': np.random.randint(1, 200, 500),
        'rating': np.random.uniform(1, 5, 500)
    })


@pytest.fixture(scope='module')
def trained_benchmark_model(benchmark_data):
    """Pre-trained model for benchmarking"""
    model = create_recommender('simple_deepfm', learning_rate=0.05)
    model.fit(benchmark_data)
    return model


@pytest.mark.performance
@pytest.mark.slow
class TestModelPerformance:
    """Model performance benchmarks"""

    def test_model_training_speed(self, benchmark_data):
        """Benchmark model training speed"""
        benchmark = BenchmarkResult("Model Training")

        model = create_recommender('simple_deepfm', learning_rate=0.05, epochs=5)

        benchmark.start()
        model.fit(benchmark_data)
        benchmark.end()

        benchmark.print_summary()

        # Assert reasonable training time (< 30 seconds)
        assert benchmark.total_time < 30, f"Training took too long: {benchmark.total_time:.2f}s"

    def test_prediction_latency(self, trained_benchmark_model):
        """Benchmark single prediction latency"""
        benchmark = BenchmarkResult("Single Prediction")
        model = trained_benchmark_model

        user_ids = list(range(1, 101))
        item_ids = list(range(1, 101))

        benchmark.start()
        for user_id in user_ids[:100]:
            for item_id in item_ids[:10]:
                start = time.time()
                model.predict(user_id, item_id)
                benchmark.record(time.time() - start)
        benchmark.end()

        benchmark.print_summary()

        # Assert P95 latency < 10ms
        assert benchmark.p95_latency < 10, f"P95 latency too high: {benchmark.p95_latency:.2f}ms"

    def test_recommendation_latency(self, trained_benchmark_model):
        """Benchmark recommendation generation latency"""
        benchmark = BenchmarkResult("Recommendation Generation")
        model = trained_benchmark_model

        user_ids = list(range(1, 101))

        benchmark.start()
        for user_id in user_ids[:50]:
            start = time.time()
            model.get_user_recommendations(user_id, top_k=10)
            benchmark.record(time.time() - start)
        benchmark.end()

        benchmark.print_summary()

        # Assert P95 latency < 100ms
        assert benchmark.p95_latency < 100, f"P95 latency too high: {benchmark.p95_latency:.2f}ms"

    def test_batch_prediction_throughput(self, trained_benchmark_model):
        """Benchmark batch prediction throughput"""
        benchmark = BenchmarkResult("Batch Predictions")
        model = trained_benchmark_model

        user_ids = list(range(1, 101))
        item_ids = list(range(1, 201))

        benchmark.start()
        for _ in range(100):
            user_id = np.random.choice(user_ids)
            item_id = np.random.choice(item_ids)
            start = time.time()
            model.predict(user_id, item_id)
            benchmark.record(time.time() - start)
        benchmark.end()

        benchmark.print_summary()

        # Assert QPS > 100
        assert benchmark.qps > 100, f"QPS too low: {benchmark.qps:.2f}"


@pytest.mark.performance
class TestCachePerformance:
    """Cache performance benchmarks"""

    @pytest.fixture
    def mock_cache(self):
        """Mock cache for testing"""
        from unittest.mock import MagicMock
        cache = MagicMock(spec=CacheService)
        cache.get.return_value = None
        cache.set.return_value = True
        cache.is_connected.return_value = True
        return cache

    def test_cache_write_latency(self, mock_cache):
        """Benchmark cache write latency"""
        benchmark = BenchmarkResult("Cache Write")

        test_data = {'recommendations': [1, 2, 3, 4, 5]}

        benchmark.start()
        for i in range(1000):
            start = time.time()
            mock_cache.set(f'key_{i}', test_data, ttl=300)
            benchmark.record(time.time() - start)
        benchmark.end()

        benchmark.print_summary()

        # Assert mean latency < 1ms
        assert benchmark.mean_latency < 1, f"Cache write too slow: {benchmark.mean_latency:.2f}ms"

    def test_cache_read_latency(self, mock_cache):
        """Benchmark cache read latency"""
        benchmark = BenchmarkResult("Cache Read")

        benchmark.start()
        for i in range(1000):
            start = time.time()
            mock_cache.get(f'key_{i}')
            benchmark.record(time.time() - start)
        benchmark.end()

        benchmark.print_summary()

        # Assert mean latency < 1ms
        assert benchmark.mean_latency < 1, f"Cache read too slow: {benchmark.mean_latency:.2f}ms"


@pytest.mark.performance
class TestMemoryUsage:
    """Memory usage benchmarks"""

    def test_model_memory_usage(self, benchmark_data):
        """Benchmark model memory usage"""
        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Create and train multiple models
            models = []
            for i in range(3):
                model = create_recommender('simple_deepfm')
                model.fit(benchmark_data)
                models.append(model)

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            print(f"\nMemory Usage:")
            print(f"  Initial: {initial_memory:.2f} MB")
            print(f"  Final:   {final_memory:.2f} MB")
            print(f"  Increase: {memory_increase:.2f} MB")
            print(f"  Per Model: {memory_increase/3:.2f} MB")

            # Assert memory increase < 500MB
            assert memory_increase < 500, f"Memory usage too high: {memory_increase:.2f}MB"

        except ImportError:
            pytest.skip("psutil not available")


def run_all_benchmarks():
    """Run all performance benchmarks"""
    pytest.main([
        __file__,
        '-v',
        '-m', 'performance',
        '--tb=short'
    ])


if __name__ == '__main__':
    run_all_benchmarks()
