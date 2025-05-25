#!/usr/bin/env python3
"""
Example script demonstrating the usage of VectorSearchIndexManager
with the Document Loader module.

This script shows how to:
1. Create a vector search index
2. Upload embeddings to Cloud Storage
3. Deploy the index to an endpoint
4. Perform vector searches
5. Clean up resources
"""

import asyncio
import os
import sys
from typing import List
import logging

# Add the root directory to the Python path
project_root = os.path.join(os.path.dirname(__file__), '..')
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

# Set PYTHONPATH environment variable
if 'PYTHONPATH' in os.environ:
    os.environ['PYTHONPATH'] = f"{project_root}:{src_path}:{os.environ['PYTHONPATH']}"
else:
    os.environ['PYTHONPATH'] = f"{project_root}:{src_path}"

from agents.document_loader.src.models.base import VectorSearchConfig, Embedding, Document
from agents.document_loader.src.vector_search.index_manager import VectorSearchIndexManager
from agents.document_loader.src.processors.embedding_processor import EmbeddingProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_documents() -> List[Document]:
    """Create sample documents for demonstration."""
    documents = [
        Document(
            title="Introduction to Machine Learning",
            content="Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
            source_type="web",
            source_url="https://example.com/ml-intro",
            category="technology",
            metadata={"author": "John Doe", "topic": "AI/ML"}
        ),
        Document(
            title="Vector Databases Explained",
            content="Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently, commonly used in AI applications for similarity search.",
            source_type="web", 
            source_url="https://example.com/vector-db",
            category="technology",
            metadata={"author": "Jane Smith", "topic": "databases"}
        ),
        Document(
            title="Natural Language Processing",
            content="Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret and manipulate human language.",
            source_type="web",
            source_url="https://example.com/nlp",
            category="technology", 
            metadata={"author": "Bob Johnson", "topic": "AI/ML"}
        ),
        Document(
            title="Cloud Computing Best Practices",
            content="Cloud computing provides on-demand delivery of IT resources over the internet with pay-as-you-go pricing. Best practices include security, cost optimization, and scalability.",
            source_type="web",
            source_url="https://example.com/cloud",
            category="technology",
            metadata={"author": "Alice Brown", "topic": "cloud"}
        ),
        Document(
            title="Data Science Fundamentals",
            content="Data science combines statistics, mathematics, programming, and domain expertise to extract insights from structured and unstructured data.",
            source_type="web",
            source_url="https://example.com/data-science",
            category="technology",
            metadata={"author": "Charlie Wilson", "topic": "data"}
        )
    ]
    return documents


async def generate_sample_embeddings(documents: List[Document], project_id: str) -> List[Embedding]:
    """Generate embeddings for sample documents."""
    logger.info(f"Generating embeddings for {len(documents)} documents...")
    
    try:
        # Initialize embedding processor
        embedding_processor = EmbeddingProcessor(
            project_id=project_id,
            location="us-central1",
            batch_size=2,
            verbose=True
        )
        
        # Generate embeddings
        embeddings = await embedding_processor.generate_embeddings(documents)
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        # For demonstration purposes, create mock embeddings if real ones fail
        logger.info("Creating mock embeddings for demonstration...")
        mock_embeddings = []
        for doc in documents:
            # Create mock embedding vector (768 dimensions)
            mock_vector = [0.1 + (i * 0.001) for i in range(768)]
            embedding = Embedding(
                document_id=doc.id,
                embedding=mock_vector,
                model_name="mock-model",
                category=doc.category,
                metadata={
                    "title": doc.title,
                    "source_type": doc.source_type,
                    "source_url": doc.source_url
                }
            )
            mock_embeddings.append(embedding)
        return mock_embeddings


async def demonstrate_vector_search_workflow():
    """Demonstrate the complete vector search workflow."""
    
    # Configuration
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        logger.error("GOOGLE_CLOUD_PROJECT environment variable not set")
        return
    
    bucket_name = f"{project_id}-vector-search-demo"
    
    # Create vector search configuration
    config = VectorSearchConfig(
        project_id=project_id,
        location="us-central1",
        index_name="demo-document-search-index",
        dimensions=768,
        approximate_neighbors_count=150,
        distance_measure_type="COSINE",
        algorithm_config={
            "treeAhConfig": {
                "leafNodeEmbeddingCount": 500,
                "leafNodesToSearchPercent": 10
            }
        }
    )
    
    # Initialize the vector search manager
    logger.info("Initializing Vector Search Index Manager...")
    index_manager = VectorSearchIndexManager(config)
    
    try:
        # Step 1: Create sample documents and embeddings
        logger.info("Step 1: Creating sample documents and embeddings...")
        documents = create_sample_documents()
        embeddings = await generate_sample_embeddings(documents, project_id)
        
        # Step 2: Create vector search index
        logger.info("Step 2: Creating vector search index...")
        index_resource_name = index_manager.create_index(
            metadata={"environment": "demo", "purpose": "document-search"}
        )
        logger.info(f"✓ Index created: {index_resource_name}")
        
        # Step 3: Upload embeddings to Cloud Storage
        logger.info("Step 3: Uploading embeddings to Cloud Storage...")
        gcs_uri = index_manager.upload_embeddings(
            embeddings, 
            bucket_name, 
            file_prefix="demo_embeddings"
        )
        logger.info(f"✓ Embeddings uploaded to: {gcs_uri}")
        
        # Step 4: Deploy index to endpoint
        logger.info("Step 4: Deploying index to endpoint...")
        logger.info("This may take several minutes...")
        endpoint_resource_name = index_manager.deploy_index(gcs_uri)
        logger.info(f"✓ Index deployed to endpoint: {endpoint_resource_name}")
        
        # Step 5: Get index statistics
        logger.info("Step 5: Getting index statistics...")
        stats = index_manager.get_index_stats()
        logger.info(f"✓ Index stats: {stats}")
        
        # Step 6: Perform vector searches
        logger.info("Step 6: Performing vector searches...")
        
        # Search for documents similar to "machine learning"
        query_embedding = embeddings[0].embedding  # Use first document's embedding as query
        search_results = index_manager.search(
            query_embedding=query_embedding,
            num_neighbors=3
        )
        
        logger.info("Search results for 'machine learning' topic:")
        for i, result in enumerate(search_results, 1):
            logger.info(f"  {i}. Document ID: {result['id']}, Score: {result['score']:.3f}")
        
        # Batch search example
        query_embeddings = [emb.embedding for emb in embeddings[:2]]
        batch_results = index_manager.batch_search(
            query_embeddings=query_embeddings,
            num_neighbors=2
        )
        
        logger.info(f"Batch search results: {len(batch_results)} queries processed")
        for i, results in enumerate(batch_results):
            logger.info(f"  Query {i+1}: {len(results)} results")
        
        # Step 7: Demonstrate search with filters (if supported)
        logger.info("Step 7: Demonstrating filtered search...")
        try:
            filtered_results = index_manager.search(
                query_embedding=query_embedding,
                num_neighbors=2,
                filter_criteria={"category": "technology"}
            )
            logger.info(f"✓ Filtered search returned {len(filtered_results)} results")
        except Exception as e:
            logger.warning(f"Filtered search not supported or failed: {e}")
        
        logger.info("\n" + "="*60)
        logger.info("DEMO COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        
        # Ask user if they want to clean up
        cleanup_choice = input("\nDo you want to clean up resources? (y/n): ").lower().strip()
        
        if cleanup_choice == 'y':
            logger.info("Step 8: Cleaning up resources...")
            index_manager.cleanup(delete_index=True, delete_endpoint=True)
            logger.info("✓ Resources cleaned up successfully")
        else:
            logger.info("Resources left for manual cleanup. Remember to clean up to avoid charges!")
            logger.info(f"Index: {index_resource_name}")
            logger.info(f"Endpoint: {endpoint_resource_name}")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        logger.error("Attempting cleanup...")
        try:
            index_manager.cleanup(delete_index=True, delete_endpoint=True)
            logger.info("✓ Cleanup completed")
        except Exception as cleanup_error:
            logger.error(f"Cleanup failed: {cleanup_error}")
        raise


def list_existing_indexes():
    """List existing indexes for inspection."""
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        logger.error("GOOGLE_CLOUD_PROJECT environment variable not set")
        return
    
    config = VectorSearchConfig(
        project_id=project_id,
        location="us-central1",
        index_name="demo-document-search-index",
        dimensions=768
    )
    
    manager = VectorSearchIndexManager(config)
    indexes = manager.list_indexes()
    
    logger.info(f"Found {len(indexes)} indexes:")
    for index in indexes:
        logger.info(f"  - {index.display_name} ({index.resource_name})")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Vector Search Demo")
    parser.add_argument("--list-indexes", action="store_true", 
                       help="List existing indexes")
    parser.add_argument("--run-demo", action="store_true", 
                       help="Run the complete demo workflow")
    
    args = parser.parse_args()
    
    if args.list_indexes:
        list_existing_indexes()
    elif args.run_demo:
        asyncio.run(demonstrate_vector_search_workflow())
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 