# Roadmap Generation Agentic System

This repository implements a comprehensive roadmap generation system using Google's Agent Development Kit (ADK) and GenKit. The system consists of multiple specialized AI agents that collaborate to analyze documents, align strategies, generate ideas, and synthesize comprehensive roadmaps.

## System Architecture

The system is built as a monorepo containing multiple specialized agents that work together through an agentic flow orchestration layer:

### Agent Components

- **Document Loader Agent** (`agents/document_loader/`): Processes documents from multiple sources (Google Drive, GitHub, Web URLs), generates embeddings, and creates vector search indexes
- **Strategy Aligner Agent** (`agents/strategy_aligner/`): Analyzes organizational strategies and aligns them with business objectives  
- **Ideation Agent** (`agents/ideation/`): Generates innovative ideas and solutions based on analyzed content and strategic context
- **Roadmap Synthesis Agent** (`agents/roadmap_synthesis/`): Synthesizes information from all agents to create comprehensive, actionable roadmaps

### Agentic Flow Orchestration

- **ADK Integration** (`agentic_flow/`): Multi-agent orchestration using Google's Agent Development Kit
- **Shared Components** (`shared/`): Common models, utilities, and configurations used across all agents
- **Deployment** (`deployment/`): Infrastructure and deployment configurations for various environments

## Features

- **Multi-Source Document Processing**: Ingest from Google Drive, GitHub repositories, and web URLs
- **Vector Search & RAG**: Advanced document retrieval using Vertex AI Vector Search
- **Strategic Alignment**: AI-powered strategy analysis and alignment
- **Collaborative Ideation**: Context-aware idea generation
- **Comprehensive Roadmapping**: End-to-end roadmap synthesis and generation
- **Scalable Architecture**: Microservices-based agent design with shared infrastructure
- **Production Ready**: Built with Google Cloud services and enterprise-grade reliability

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Google Cloud credentials:**
   ```bash
   # Set your project ID
   export GOOGLE_CLOUD_PROJECT="your-project-id"
   
   # Authenticate with Google Cloud
   gcloud auth application-default login
   ```

3. **Run individual agents:**
   ```bash
   # Test Document Loader agent
   cd agents/document_loader
   python examples/vector_search_example.py
   
   # Test other agents independently
   cd agents/strategy_aligner
   python examples/strategy_alignment_example.py
   ```

4. **Run the full agentic flow:**
   ```bash
   # Install ADK
   pip install google-adk
   
   # Run the orchestrated system
   adk run agentic_flow.flows.roadmap_generation_flow
   ```

## Development

Each agent can be developed and tested independently:

```bash
# Test specific agent
python -m pytest agents/document_loader/tests/
python -m pytest agents/strategy_aligner/tests/

# Test integration
python -m pytest tests/integration/

# Run examples
python -m agents.document_loader.examples.vector_search_example
```

## Project Structure

```
├── agents/                          # Individual agent components
│   ├── document_loader/            # Document processing & vector search
│   ├── strategy_aligner/           # Strategy analysis & alignment  
│   ├── ideation/                   # Idea generation & innovation
│   └── roadmap_synthesis/          # Roadmap creation & synthesis
├── agentic_flow/                   # ADK orchestration layer
│   ├── flows/                      # Multi-agent workflows
│   ├── agents/                     # ADK agent definitions
│   └── tools/                      # Shared integration tools
├── shared/                         # Common utilities & models
├── deployment/                     # Infrastructure & deployment
├── tests/                          # Integration & E2E tests
└── docs/                          # Documentation
```

## Requirements

- Python 3.9+
- Google Cloud project with Vertex AI API enabled
- Google Cloud authentication configured
- GitHub API token (for repository access)

## Contributing

Each agent component can be developed independently. See individual agent READMEs for specific development instructions. 
