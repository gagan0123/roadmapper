#!/bin/bash
# Run examples for all agents

set -e

echo "🎮 Running agent examples..."

echo "📄 Document Loader Example..."
cd agents/document_loader && python examples/vector_search_example.py
cd ../..

echo "🗺️ Full Roadmap Generation Example..."
python examples/roadmap_generation_example.py --full-demo

echo "✅ All examples completed!"
