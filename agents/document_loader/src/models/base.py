from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

class Document(BaseModel):
    """Base document model for all ingested content."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    content: str
    source_type: str  # "drive", "github", or "web"
    source_url: str
    category: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    images: List[bytes] = Field(default_factory=list)  # Multimodal support: list of image bytes
    embedding_model_type: Optional[str] = "multimodal" # Default model type

class Embedding(BaseModel):
    """Model for document embeddings."""
    document_id: str
    embedding: List[float]
    model_name: str = "textembedding-gecko@latest"
    category: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class VectorSearchConfig(BaseModel):
    """Configuration for Vector Search index."""
    project_id: str
    location: str = "us-central1"
    index_name: str
    dimensions: int = 768  # Default for textembedding-gecko
    approximate_neighbors_count: int = 150
    distance_measure_type: str = "DOT_PRODUCT_DISTANCE"
    algorithm_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "treeAhConfig": {
                "leafNodeEmbeddingCount": 500,
                "leafNodesToSearchPercent": 10
            }
        }
    ) 