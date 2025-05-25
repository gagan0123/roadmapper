import asyncio
from src.loaders.drive_loader import DriveLoader
from src.loaders.github_loader import GitHubLoader
from src.loaders.web_loader import WebLoader
from src.processors.embedding_processor import EmbeddingProcessor as MultimodalEmbeddingProcessor
from src.processors.text_embedding_processor import TextEmbeddingProcessor
from src.vector_search.index_manager import VectorSearchIndexManager
from src.data_source_loader import DataSourceLoader
from src.config import Config
import os
from pydantic import ValidationError
from src.models.base import VectorSearchConfig, Document
import argparse
import logging
from typing import List

# Define model IDs, could also come from config
TEXT_EMBEDDING_MODEL_ID = "text-embedding-004"
MULTIMODAL_EMBEDDING_MODEL_ID = "multimodalembedding@001"

async def main(verbose=False, cleanup=False):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    index_manager = None
    try:
        # Load configuration
        config = Config()
    except ValidationError as e:
        print("Configuration error:")
        for error in e.errors():
            print(f"- {error['loc'][0]}: {error['msg']}")
        return
    
    try:
        # Initialize data source loader
        data_source_loader = DataSourceLoader()
        
        # Load data sources - now returns (item, category, model_type)
        drive_sources = data_source_loader.load_drive_folders()
        github_sources = data_source_loader.load_github_repos()
        web_sources = data_source_loader.load_web_urls()
        
        if not any([drive_sources, github_sources, web_sources]):
            print("No data sources found. Please add sources to the data/sources directory.")
            return
        
        # Initialize processors
        multimodal_embedding_processor = MultimodalEmbeddingProcessor(
            project_id=config.google_cloud_project,
            location="us-central1",
            model_id=MULTIMODAL_EMBEDDING_MODEL_ID, # Specify model ID
            verbose=verbose,
            batch_size=5 # Multimodal typically has smaller batch size
        )
        text_embedding_processor = TextEmbeddingProcessor(
            project_id=config.google_cloud_project,
            location="us-central1",
            model_id=TEXT_EMBEDDING_MODEL_ID, # Specify model ID
            verbose=verbose,
            batch_size=250 # Text models can handle larger batches
        )
        
        # Initialize vector search index manager
        vector_search_config = VectorSearchConfig(
            project_id=config.google_cloud_project,
            location="us-central1",
            index_name=config.index_name,
            dimensions=768,  # Both models produce 768-dim embeddings
            approximate_neighbors_count=150,
            distance_measure_type="COSINE", # COSINE is good for semantic similarity
            algorithm_config={
                "treeAhConfig": {
                    "leafNodeEmbeddingCount": 500,
                    "leafNodesToSearchPercent": 10
                }
            }
        )
        index_manager = VectorSearchIndexManager(vector_search_config)
        
        all_loaded_documents: List[Document] = [] # Explicitly typed
        
        # Process Google Drive folders
        for folder_id, category, model_type_preference in drive_sources:
            print(f"\nProcessing Google Drive folder: {folder_id} (Category: {category or 'N/A'}, Model: {model_type_preference})")
            drive_loader = DriveLoader(item_id=folder_id, category=category, embedding_model_type=model_type_preference)
            async for doc in drive_loader.load(): # Document objects now have embedding_model_type set by loader
                all_loaded_documents.append(doc)
        
        # Process GitHub repositories
        for repo_name, category, model_type_preference in github_sources:
            print(f"\nProcessing GitHub repository: {repo_name} (Category: {category or 'N/A'}, Model: {model_type_preference})")
            github_loader = GitHubLoader(repo_name=repo_name, category=category, embedding_model_type=model_type_preference)
            async for doc in github_loader.load():
                all_loaded_documents.append(doc)
        
        # Process web URLs
        for url, category, model_type_preference in web_sources:
            print(f"\nProcessing Web URL: {url} (Category: {category or 'N/A'}, Model: {model_type_preference})")
            web_loader = WebLoader(urls=[url], category=category, embedding_model_type=model_type_preference)
            async for doc in web_loader.load():
                all_loaded_documents.append(doc)
        
        if not all_loaded_documents:
            print("No documents were loaded from any source.")
            return
        
        print(f"\nLoaded {len(all_loaded_documents)} documents in total")
        
        # Separate documents by preferred embedding model type
        text_model_documents: List[Document] = []
        multimodal_model_documents: List[Document] = []

        for doc in all_loaded_documents:
            if doc.embedding_model_type == "text":
                text_model_documents.append(doc)
            else: # Default to multimodal if not "text" (includes "multimodal" or any other unspecified)
                multimodal_model_documents.append(doc)

        print(f"\nProcessing {len(text_model_documents)} documents with text embedding model ({TEXT_EMBEDDING_MODEL_ID})...")
        print(f"Processing {len(multimodal_model_documents)} documents with multimodal embedding model ({MULTIMODAL_EMBEDDING_MODEL_ID})...")

        all_embeddings = []
        
        # Generate embeddings using TextEmbeddingProcessor
        if text_model_documents:
            text_embeddings = await text_embedding_processor.generate_embeddings(text_model_documents)
            all_embeddings.extend(text_embeddings)
            print(f"Generated {len(text_embeddings)} embeddings using {TEXT_EMBEDDING_MODEL_ID}")

        # Generate embeddings using MultimodalEmbeddingProcessor
        if multimodal_model_documents:
            multimodal_embeddings = await multimodal_embedding_processor.generate_embeddings(multimodal_model_documents)
            all_embeddings.extend(multimodal_embeddings)
            print(f"Generated {len(multimodal_embeddings)} embeddings using {MULTIMODAL_EMBEDDING_MODEL_ID}")
        
        if not all_embeddings:
            print("No embeddings were generated. Exiting.")
            return

        print(f"\nTotal generated {len(all_embeddings)} embeddings from all processors.")
        
        if verbose:
            logger.debug(f"First 3 embedding IDs: {[e.document_id for e in all_embeddings[:3]] if all_embeddings else 'N/A'}")
            logger.debug(f"Sample embedding model names: {[e.model_name for e in all_embeddings[:3]] if all_embeddings else 'N/A'}")
            if len(all_loaded_documents) != len(all_embeddings) and not multimodal_model_documents: # Simple check for text only path
                 # For multimodal, one doc can produce multiple embeddings, so this check is not direct.
                 # For text-only, one doc produces one embedding. 
                 # This condition is tricky because a mix of text and multimodal is expected.
                 pass # A more sophisticated check might be needed depending on how docs map to embeddings.

        # Create and deploy index
        print("\nCreating vector search index...")
        # Check if index exists, use it or create new one.
        # For simplicity, the current IndexManager might just create or fail if it exists.
        # Consider adding get_or_create logic in IndexManager if needed.
        index_resource_name = index_manager.create_index(
            metadata={"created_by": "document-loader-pipeline", "environment": "production"}
        )
        print(f"✓ Index target: {index_resource_name}") # Name vs Resource Name
        
        # Upload embeddings to Cloud Storage
        print("\nUploading embeddings to Cloud Storage...")
        bucket_name = f"{config.google_cloud_project}-embeddings"
        gcs_uri = index_manager.upload_embeddings(
            all_embeddings, 
            bucket_name,
            file_prefix="hybrid_document_embeddings"
        )
        print(f"✓ Uploaded embeddings to: {gcs_uri}")
        
        # Deploy index
        print("\nDeploying index (this may take several minutes)...")
        # The deploy_index method in the original code implies it handles deploying to an endpoint.
        # It might internally create an endpoint if one is not specified or found.
        index_endpoint_resource_name = index_manager.deploy_index(gcs_uri)
        print(f"✓ Index deployed at endpoint: {index_endpoint_resource_name}")
        
        # Get index statistics
        print("\nGetting index statistics...")
        stats = index_manager.get_index_stats()
        if stats:
            print("Index Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        
        # Perform a sample search
        if all_embeddings:
            print("\nPerforming sample search...")
            try:
                sample_results = index_manager.search(
                    query_embedding=all_embeddings[0].embedding,
                    num_neighbors=min(5, len(all_embeddings))
                )
                print(f"Sample search results ({len(sample_results)} found for doc ID {all_embeddings[0].document_id}):")
                for i, result in enumerate(sample_results, 1):
                    print(f"  {i}. Document ID: {result['id']}, Score: {result['score']:.3f}")
            except Exception as search_error:
                logger.warning(f"Sample search failed: {search_error}")
        
        # Batch search demonstration
        if len(all_embeddings) >= 2:
            print("\nPerforming batch search demonstration...")
            try:
                batch_queries = [emb.embedding for emb in all_embeddings[:min(2, len(all_embeddings))]]
                batch_results = index_manager.batch_search(batch_queries, num_neighbors=3)
                print(f"Batch search results: {len(batch_results)} queries processed")
                for i, results in enumerate(batch_results):
                    print(f"  Query {i+1} (for doc ID {all_embeddings[i].document_id}): {len(results)} results")
            except Exception as batch_error:
                logger.warning(f"Batch search failed: {batch_error}")
        
        print("\n" + "="*60)
        print("DOCUMENT PROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Index Resource Name: {index_resource_name}")
        print(f"Endpoint Resource Name: {index_endpoint_resource_name}")
        print(f"Embeddings GCS URI: {gcs_uri}")
        print(f"Total Documents Loaded: {len(all_loaded_documents)}")
        print(f"Total Embeddings Generated: {len(all_embeddings)}")
        
        if cleanup:
            print("\nCleaning up resources...")
            index_manager.cleanup(delete_index=True, delete_endpoint=True)
            print("✓ Resources cleaned up successfully")
        else:
            print("\nNote: Vector search resources remain active.")
            print("Use --cleanup flag to automatically clean up resources after processing.")
            print("Or manually clean up using the Google Cloud Console to avoid ongoing charges.")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True if verbose else False)
        # if verbose:
        #     import traceback
        #     traceback.print_exc()
        
        # Attempt cleanup on failure
        if index_manager:
            logger.info("Attempting cleanup due to failure...")
            try:
                index_manager.cleanup(delete_index=True, delete_endpoint=True)
                logger.info("✓ Cleanup completed")
            except Exception as cleanup_error:
                logger.error(f"Cleanup failed: {cleanup_error}")
        
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document Loading and Embedding Pipeline.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--cleanup", action="store_true", help="Clean up Vertex AI resources after processing.")
    args = parser.parse_args()
    
    asyncio.run(main(verbose=args.verbose, cleanup=args.cleanup)) 