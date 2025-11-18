#!/usr/bin/env python3
"""
Test runner script for Intelligent Recommender System

This script provides a convenient way to run tests with different configurations.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_pytest(args_list):
    """Run pytest with given arguments"""
    cmd = ['pytest'] + args_list
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=Path(__file__).parent).returncode


def run_all_tests(verbose=True):
    """Run all tests"""
    args = ['tests/']
    if verbose:
        args.append('-v')
    return run_pytest(args)


def run_unit_tests(verbose=True):
    """Run only unit tests"""
    args = ['tests/', '-m', 'unit']
    if verbose:
        args.append('-v')
    return run_pytest(args)


def run_integration_tests(verbose=True):
    """Run only integration tests"""
    args = ['tests/', '-m', 'integration']
    if verbose:
        args.append('-v')
    return run_pytest(args)


def run_with_coverage(html=True, fail_under=50):
    """Run tests with coverage report"""
    args = [
        'tests/',
        f'--cov=models',
        f'--cov=services',
        f'--cov=utils',
        f'--cov=app',
        '--cov-report=term-missing',
        f'--cov-fail-under={fail_under}',
        '-v'
    ]

    if html:
        args.append('--cov-report=html')

    return run_pytest(args)


def run_quick_tests():
    """Run quick tests only (skip slow tests)"""
    args = ['tests/', '-m', 'not slow', '-v']
    return run_pytest(args)


def run_specific_file(file_path, verbose=True):
    """Run tests from a specific file"""
    args = [file_path]
    if verbose:
        args.append('-v')
    return run_pytest(args)


def run_with_markers(markers, verbose=True):
    """Run tests with specific markers"""
    args = ['tests/', '-m', markers]
    if verbose:
        args.append('-v')
    return run_pytest(args)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Test runner for Intelligent Recommender System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python run_tests.py --all

  # Run with coverage
  python run_tests.py --coverage

  # Run only unit tests
  python run_tests.py --unit

  # Run only integration tests
  python run_tests.py --integration

  # Run quick tests (skip slow ones)
  python run_tests.py --quick

  # Run specific test file
  python run_tests.py --file tests/test_cache.py

  # Run tests with specific marker
  python run_tests.py --marker cache
        """
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all tests'
    )
    parser.add_argument(
        '--unit',
        action='store_true',
        help='Run only unit tests'
    )
    parser.add_argument(
        '--integration',
        action='store_true',
        help='Run only integration tests'
    )
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Run tests with coverage report'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick tests only (skip slow tests)'
    )
    parser.add_argument(
        '--file',
        type=str,
        help='Run tests from specific file'
    )
    parser.add_argument(
        '--marker',
        type=str,
        help='Run tests with specific marker'
    )
    parser.add_argument(
        '--fail-under',
        type=int,
        default=50,
        help='Minimum coverage percentage required (default: 50)'
    )
    parser.add_argument(
        '--no-html',
        action='store_true',
        help='Disable HTML coverage report'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    verbose = args.verbose

    # Run tests based on arguments
    if args.all:
        return run_all_tests(verbose=verbose)

    elif args.unit:
        return run_unit_tests(verbose=verbose)

    elif args.integration:
        return run_integration_tests(verbose=verbose)

    elif args.coverage:
        return run_with_coverage(html=not args.no_html, fail_under=args.fail_under)

    elif args.quick:
        return run_quick_tests()

    elif args.file:
        return run_specific_file(args.file, verbose=verbose)

    elif args.marker:
        return run_with_markers(args.marker, verbose=verbose)

    else:
        # Default: run all tests
        return run_all_tests(verbose=verbose)


if __name__ == '__main__':
    sys.exit(main())
