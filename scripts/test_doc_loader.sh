#!/bin/bash
# Quick test script for Document Loader agent only

set -e

echo "📄 Testing Document Loader Agent..."
cd agents/document_loader
echo "Running unit tests..."
python -m pytest tests/ -v -x
echo "✅ Document Loader tests completed!"
