# Document Loader Agent

The Document Loader Agent is responsible for processing documents from multiple sources, generating embeddings, and creating vector search indexes for the roadmap generation system.

## Features

- **Multi-Source Document Ingestion**:
  - Google Drive (documents and slide decks)
  - GitHub repositories  
  - Web URLs
- **Text Extraction and Processing**
- **Embedding Generation** using Vertex AI
- **Vector Search Index Creation** and management
- **Data Validation** using Pydantic

## Development

This agent can be developed and tested completely independently:

### Setup
```bash
# From the agent directory
cd agents/document_loader

# Install dependencies (if developing independently)
pip install -r ../../requirements.txt

# Set up environment variables
export GOOGLE_CLOUD_PROJECT="your-project-id"
gcloud auth application-default login
```

### Testing
```bash
# Run all tests for this agent
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_vector_search_index_manager.py
python -m pytest tests/test_embedding_processor.py
python -m pytest tests/test_drive_loader.py
```

### Examples
```bash
# Run the vector search example
python examples/vector_search_example.py

# Or from the root directory
python -m agents.document_loader.examples.vector_search_example
```

## Architecture

```
src/
├── models/              # Pydantic data models
├── loaders/            # Document loaders for different sources
│   ├── drive_loader.py
│   ├── github_loader.py
│   └── web_loader.py
├── processors/         # Text processing and embedding generation
├── vector_search/      # Vector search index management
└── utils/             # Utilities and helpers
```

## Configuration

The agent uses configuration files in the `config/` directory and environment variables for authentication and project settings.

## Integration with Other Agents

This agent provides document processing capabilities that other agents can use:
- Strategy Aligner: Uses processed documents for strategy analysis
- Ideation: Leverages document content for idea generation  
- Roadmap Synthesis: Incorporates document insights into roadmaps

## API

Key classes and functions:
- `DataSourceLoader`: Main orchestrator for document loading
- `VectorSearchIndexManager`: Manages vector search operations
- `EmbeddingProcessor`: Handles embedding generation
- Document loaders for specific sources (`DriveLoader`, `GitHubLoader`, `WebLoader`) 