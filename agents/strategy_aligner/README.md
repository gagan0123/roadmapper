# Strategy Aligner Agent

The Strategy Aligner Agent analyzes organizational strategies and aligns them with business objectives to provide strategic context for roadmap generation.

## Purpose

This agent processes strategic documents, business plans, and organizational goals to:
- Extract key strategic themes and objectives
- Identify strategic priorities and constraints
- Analyze alignment between different strategic initiatives
- Provide strategic context for idea generation and roadmap synthesis

## Features

- **Strategic Document Analysis**: Parse and understand strategic documents
- **Objective Extraction**: Identify and categorize business objectives
- **Alignment Assessment**: Evaluate alignment between initiatives and goals
- **Strategic Context Generation**: Provide strategic insights for other agents

## Development

This agent can be developed and tested independently:

### Setup
```bash
# From the agent directory
cd agents/strategy_aligner

# Install dependencies
pip install -r ../../requirements.txt

# Set up environment variables
export GOOGLE_CLOUD_PROJECT="your-project-id"
gcloud auth application-default login
```

### Testing
```bash
# Run all tests for this agent
python -m pytest tests/ -v

# Run specific components
python -m pytest tests/test_strategy_analyzer.py
python -m pytest tests/test_alignment_engine.py
```

### Examples
```bash
# Run strategy alignment examples
python examples/strategy_alignment_example.py

# Or from root directory
python -m agents.strategy_aligner.examples.strategy_alignment_example
```

## Architecture

```
src/
├── analyzers/          # Strategic analysis components
├── alignment/          # Alignment assessment logic
├── extractors/         # Strategic information extraction
├── models/            # Strategy-specific data models
└── utils/             # Strategy-related utilities
```

## Integration

- **Input**: Documents from Document Loader Agent
- **Output**: Strategic context and alignment insights
- **Consumers**: Ideation Agent and Roadmap Synthesis Agent

## Development Status

🚧 **This agent is in development phase**

### TODO
- [ ] Implement strategic document parsing
- [ ] Create alignment assessment algorithms
- [ ] Add strategic objective extraction
- [ ] Build strategy validation framework
- [ ] Create integration with Document Loader Agent 