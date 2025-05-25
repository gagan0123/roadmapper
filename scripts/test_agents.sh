#!/bin/bash
# Test individual agents

set -e

echo "🧪 Testing individual agents..."

echo "📄 Testing Document Loader Agent..."
cd agents/document_loader && python -m pytest tests/ -v
cd ../..

echo "🎯 Testing Strategy Aligner Agent..."
cd agents/strategy_aligner && python -m pytest tests/ -v
cd ../..

echo "💡 Testing Ideation Agent..."
cd agents/ideation && python -m pytest tests/ -v
cd ../..

echo "🗺️ Testing Roadmap Synthesis Agent..."
cd agents/roadmap_synthesis && python -m pytest tests/ -v
cd ../..

echo "✅ All agent tests completed!"
