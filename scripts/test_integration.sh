#!/bin/bash
# Test integration and end-to-end workflows

set -e

echo "🔗 Running integration tests..."
python -m pytest tests/integration/ -v

echo "🌐 Running end-to-end tests..."
python -m pytest tests/e2e/ -v

echo "✅ Integration tests completed!"
