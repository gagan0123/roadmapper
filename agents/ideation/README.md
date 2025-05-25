# Ideation Agent

The Ideation Agent generates innovative ideas and solutions based on analyzed content and strategic context for roadmap development.

## Purpose

This agent uses AI-powered ideation to:
- Generate creative solutions based on strategic context
- Synthesize insights from documents into actionable ideas
- Evaluate and rank ideas based on strategic alignment
- Provide diverse perspectives and innovative approaches

## Features

- **AI-Powered Idea Generation**: Generate creative solutions using LLMs
- **Context-Aware Ideation**: Use strategic and document context for relevant ideas
- **Idea Evaluation**: Assess ideas against strategic objectives
- **Innovation Frameworks**: Apply structured innovation methodologies
- **Collaborative Filtering**: Rank and prioritize ideas

## Development

This agent can be developed and tested independently:

### Setup
```bash
# From the agent directory
cd agents/ideation

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
python -m pytest tests/test_idea_generator.py
python -m pytest tests/test_innovation_engine.py
python -m pytest tests/test_idea_evaluator.py
```

### Examples
```bash
# Run ideation examples
python examples/idea_generation_example.py

# Or from root directory
python -m agents.ideation.examples.idea_generation_example
```

## Architecture

```
src/
├── generators/         # Idea generation engines
├── evaluators/         # Idea assessment and ranking
├── frameworks/         # Innovation methodologies
├── models/            # Ideation-specific data models
└── utils/             # Ideation utilities
```

## Integration

- **Input**: Strategic context from Strategy Aligner Agent and documents from Document Loader Agent
- **Output**: Ranked ideas and innovative solutions
- **Consumers**: Roadmap Synthesis Agent

## Development Status

🚧 **This agent is in development phase**

### TODO
- [ ] Implement LLM-based idea generation
- [ ] Create idea evaluation algorithms
- [ ] Add innovation framework support
- [ ] Build strategic alignment scoring
- [ ] Create integration with Strategy Aligner Agent
- [ ] Add collaborative filtering capabilities 