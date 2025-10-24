#!/usr/bin/env python3
"""
Test runner script for RiskX backend comprehensive testing suite.
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print('='*60)
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ SUCCESS: {description}")
        if result.stdout:
            print("STDOUT:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ FAILED: {description}")
        print("STDERR:", e.stderr)
        if e.stdout:
            print("STDOUT:", e.stdout)
        return False


def run_unit_tests(verbose=False, coverage=False):
    """Run unit tests."""
    cmd = "python -m pytest tests/unit/"
    
    if verbose:
        cmd += " -v"
    
    if coverage:
        cmd += " --cov=src --cov-report=term-missing --cov-report=html"
    
    cmd += " --tb=short"
    
    return run_command(cmd, "Unit Tests")


def run_integration_tests(verbose=False):
    """Run integration tests."""
    cmd = "python -m pytest tests/integration/"
    
    if verbose:
        cmd += " -v"
    
    cmd += " --tb=short"
    
    return run_command(cmd, "Integration Tests")


def run_e2e_tests(verbose=False):
    """Run end-to-end tests."""
    cmd = "python -m pytest tests/e2e/"
    
    if verbose:
        cmd += " -v"
    
    cmd += " --tb=short"
    
    return run_command(cmd, "End-to-End Tests")


def run_performance_tests(verbose=False):
    """Run performance tests."""
    cmd = "python -m pytest tests/ -m slow"
    
    if verbose:
        cmd += " -v"
    
    cmd += " --tb=short"
    
    return run_command(cmd, "Performance Tests")


def run_linting():
    """Run code linting."""
    commands = [
        ("python -m flake8 src/ --max-line-length=100 --ignore=E203,W503", "Flake8 Linting"),
        ("python -m black --check src/", "Black Code Formatting Check"),
        ("python -m isort --check-only src/", "Import Sorting Check"),
        ("python -m mypy src/ --ignore-missing-imports", "Type Checking")
    ]
    
    results = []
    for cmd, desc in commands:
        try:
            result = run_command(cmd, desc)
            results.append(result)
        except Exception:
            # Some tools might not be installed, continue anyway
            results.append(False)
    
    return all(results)


def run_security_checks():
    """Run security checks."""
    commands = [
        ("python -m bandit -r src/ -f json", "Security Vulnerability Scan"),
        ("python -m safety check", "Dependency Security Check")
    ]
    
    results = []
    for cmd, desc in commands:
        try:
            result = run_command(cmd, desc)
            results.append(result)
        except Exception:
            # Security tools might not be installed
            print(f"⚠️  SKIPPED: {desc} (tool not installed)")
            results.append(True)  # Don't fail if security tools aren't available
    
    return all(results)


def check_test_environment():
    """Check if test environment is properly set up."""
    print("🔍 Checking test environment...")
    
    # Check if pytest is installed
    try:
        import pytest
        print(f"✅ pytest {pytest.__version__} is installed")
    except ImportError:
        print("❌ pytest is not installed. Run: pip install pytest")
        return False
    
    # Check if test database can be created
    try:
        import sqlite3
        print("✅ SQLite available for test database")
    except ImportError:
        print("❌ SQLite not available")
        return False
    
    # Check if async support is available
    try:
        import pytest_asyncio
        print("✅ pytest-asyncio available for async tests")
    except ImportError:
        print("⚠️  pytest-asyncio not installed (install with: pip install pytest-asyncio)")
    
    # Check if coverage tools are available
    try:
        import coverage
        print("✅ coverage.py available for code coverage")
    except ImportError:
        print("⚠️  coverage.py not installed (install with: pip install coverage)")
    
    print("✅ Test environment check completed")
    return True


def generate_test_report():
    """Generate comprehensive test report."""
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE TEST REPORT")
    print("="*80)
    
    # Run all tests with coverage
    cmd = """python -m pytest tests/ \
        --cov=src \
        --cov-report=html:htmlcov \
        --cov-report=xml \
        --cov-report=term \
        --junit-xml=test-results.xml \
        --html=test-report.html \
        --self-contained-html \
        -v"""
    
    return run_command(cmd, "Comprehensive Test Report Generation")


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="RiskX Backend Test Runner")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--e2e", action="store_true", help="Run end-to-end tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--lint", action="store_true", help="Run linting checks only")
    parser.add_argument("--security", action="store_true", help="Run security checks only")
    parser.add_argument("--all", action="store_true", help="Run all tests and checks")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--report", action="store_true", help="Generate comprehensive test report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")
    
    args = parser.parse_args()
    
    # Change to project directory
    os.chdir(project_root)
    
    print("🚀 RiskX Backend Test Suite")
    print(f"📁 Project Root: {project_root}")
    
    # Check environment
    if not check_test_environment():
        print("❌ Test environment check failed")
        sys.exit(1)
    
    # Determine what to run
    run_all = args.all or not any([
        args.unit, args.integration, args.e2e, args.performance, 
        args.lint, args.security, args.report
    ])
    
    results = []
    
    if args.report:
        results.append(generate_test_report())
    
    elif run_all or args.unit:
        results.append(run_unit_tests(args.verbose, args.coverage))
    
    if run_all or args.integration:
        results.append(run_integration_tests(args.verbose))
    
    if run_all or args.e2e:
        results.append(run_e2e_tests(args.verbose))
    
    if (run_all or args.performance) and not args.fast:
        results.append(run_performance_tests(args.verbose))
    
    if run_all or args.lint:
        results.append(run_linting())
    
    if run_all or args.security:
        results.append(run_security_checks())
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    if all(results):
        print("🎉 ALL TESTS PASSED!")
        exit_code = 0
    else:
        print("❌ SOME TESTS FAILED!")
        exit_code = 1
    
    passed = sum(results)
    total = len(results)
    print(f"📊 Results: {passed}/{total} test suites passed")
    
    if args.coverage or args.report:
        print("\n📋 Coverage Report:")
        print("  - HTML report: htmlcov/index.html")
        print("  - XML report: coverage.xml")
    
    if args.report:
        print("\n📋 Test Report:")
        print("  - HTML report: test-report.html")
        print("  - JUnit XML: test-results.xml")
    
    print("\n💡 Quick commands:")
    print("  Run all tests:        python tests/run_tests.py --all")
    print("  Run with coverage:    python tests/run_tests.py --coverage")
    print("  Generate report:      python tests/run_tests.py --report")
    print("  Run unit tests only:  python tests/run_tests.py --unit")
    print("  Run fast tests:       python tests/run_tests.py --fast")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()