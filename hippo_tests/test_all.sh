#!/bin/bash
#
# test_all.sh - Run all dispatch functionality tests
#
# This script runs both CPU and AVX2 dispatch tests to ensure they compile
# and execute without crashing. Used for CI/testing automation.
#
# Usage: ./test_all.sh

set -e  # Exit on any error

echo "=== Running All Dispatch Tests ==="
echo

# Change to the script directory
cd "$(dirname "$0")"

echo "Running CPU dispatch test..."
if nim r dispatch_cpu.nim; then
    echo "✓ CPU dispatch test PASSED"
else
    echo "✗ CPU dispatch test FAILED"
    exit 1
fi

echo
echo "Running AVX2 dispatch test..."
if nim r dispatch_cpu_avx2.nim; then
    echo "✓ AVX2 dispatch test PASSED"
else
    echo "✗ AVX2 dispatch test FAILED"
    exit 1
fi

echo
echo "=== All tests completed successfully! ==="
