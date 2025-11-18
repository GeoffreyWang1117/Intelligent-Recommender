#!/usr/bin/env python3
"""
Pytest configuration and fixtures for Intelligent Recommender System tests.

This module provides shared fixtures and configuration for all test modules.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope='session')
def test_data_small():
    """Small test dataset for quick tests"""
    np.random.seed(42)
    return pd.DataFrame({
        'user_id': np.random.randint(1, 10, 50),
        'item_id': np.random.randint(1, 15, 50),
        'rating': np.random.uniform(1, 5, 50)
    })


@pytest.fixture(scope='session')
def test_data_medium():
    """Medium test dataset for standard tests"""
    np.random.seed(42)
    return pd.DataFrame({
        'user_id': np.random.randint(1, 50, 200),
        'item_id': np.random.randint(1, 30, 200),
        'rating': np.random.uniform(1, 5, 200)
    })


@pytest.fixture(scope='session')
def test_data_large():
    """Large test dataset for performance tests"""
    np.random.seed(42)
    return pd.DataFrame({
        'user_id': np.random.randint(1, 100, 1000),
        'item_id': np.random.randint(1, 200, 1000),
        'rating': np.random.uniform(1, 5, 1000)
    })


@pytest.fixture
def temp_file():
    """Temporary file for testing file operations"""
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
        temp_path = tmp.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_dir():
    """Temporary directory for testing"""
    temp_path = tempfile.mkdtemp()
    yield temp_path

    # Cleanup
    import shutil
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)


@pytest.fixture(scope='module')
def trained_svd_model(test_data_medium):
    """Pre-trained SVD model for tests"""
    from models import create_recommender

    model = create_recommender('svd', n_components=10)
    model.fit(test_data_medium)
    return model


@pytest.fixture(scope='module')
def trained_deepfm_model(test_data_medium):
    """Pre-trained DeepFM model for tests"""
    from models import create_recommender

    model = create_recommender('simple_deepfm', learning_rate=0.05)
    model.fit(test_data_medium)
    return model


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing cache without real Redis"""
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    mock_client.ping.return_value = True
    mock_client.get.return_value = None
    mock_client.set.return_value = True
    mock_client.setex.return_value = True
    mock_client.delete.return_value = 1
    mock_client.exists.return_value = 0
    mock_client.keys.return_value = []

    return mock_client


@pytest.fixture
def cache_service_mock(mock_redis):
    """Mocked cache service for testing"""
    from services.cache import CacheService
    from unittest.mock import patch

    with patch('services.cache.redis.Redis', return_value=mock_redis):
        cache = CacheService()
        cache.redis_client = mock_redis
        yield cache


@pytest.fixture(scope='session')
def sample_recommendations():
    """Sample recommendation results for testing"""
    return [
        {'item_id': 1, 'score': 4.5, 'rank': 1},
        {'item_id': 5, 'score': 4.2, 'rank': 2},
        {'item_id': 10, 'score': 3.8, 'rank': 3},
        {'item_id': 3, 'score': 3.5, 'rank': 4},
        {'item_id': 7, 'score': 3.2, 'rank': 5},
    ]


@pytest.fixture
def algorithm_list():
    """List of available algorithms for testing"""
    return ['svd', 'simple_deepfm', 'simple_autoint', 'simple_din']


# Markers configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "cache: mark test as requiring Redis cache"
    )
    config.addinivalue_line(
        "markers", "model: mark test as model training/prediction test"
    )
    config.addinivalue_line(
        "markers", "api: mark test as API endpoint test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance benchmarking"
    )


# Collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Auto-mark tests based on file names
        if "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "test_cache" in item.nodeid:
            item.add_marker(pytest.mark.cache)
        elif "test_api" in item.nodeid:
            item.add_marker(pytest.mark.api)
        elif "test_algorithms" in item.nodeid:
            item.add_marker(pytest.mark.model)

        # Auto-mark slow tests
        if "slow" in item.name or "performance" in item.name:
            item.add_marker(pytest.mark.slow)


# Session hooks
@pytest.fixture(scope='session', autouse=True)
def setup_test_environment():
    """Setup test environment before running tests"""
    print("\n" + "="*70)
    print("Setting up test environment for Intelligent Recommender System")
    print("="*70)

    # Set random seeds for reproducibility
    np.random.seed(42)

    # Disable warnings during tests
    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)

    yield

    print("\n" + "="*70)
    print("Test environment cleanup complete")
    print("="*70)


# Pytest plugins
pytest_plugins = []
