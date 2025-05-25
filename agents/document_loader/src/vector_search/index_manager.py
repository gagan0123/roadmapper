from typing import List, Optional, Dict, Any, Tuple
from google.cloud import aiplatform
from google.cloud import storage
from google.cloud.aiplatform import MatchingEngineIndex, MatchingEngineIndexEndpoint
from google.cloud.aiplatform_v1 import PredictionServiceClient
from google.cloud.aiplatform_v1.types import PredictRequest
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import json
import os
import logging
import time
from ..models.base import Embedding, VectorSearchConfig
from datetime import datetime

logger = logging.getLogger(__name__)

class VectorSearchIndexManager:
    """Manages Vertex AI Vector Search indices and endpoints."""
    
    def __init__(self, config: VectorSearchConfig):
        self.config = config
        self.index = None
        self.endpoint = None
        self.deployed_index_id = None
        self.storage_client = None
        self.prediction_client = None
        self.initialize_vertex_ai()

    def initialize_vertex_ai(self):
        """Initialize Vertex AI clients."""
        try:
            aiplatform.init(project=self.config.project_id, location=self.config.location)
            self.storage_client = storage.Client(project=self.config.project_id)
            
            # Initialize prediction client for searching
            client_options = {"api_endpoint": f"{self.config.location}-aiplatform.googleapis.com"}
            self.prediction_client = PredictionServiceClient(client_options=client_options)
            
            logger.info(f"Initialized Vertex AI for project {self.config.project_id} in {self.config.location}")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")
            raise

    def create_index(self, metadata: Optional[Dict[str, str]] = None) -> str:
        """Create a new vector search index."""
        try:
            logger.info(f"Creating index '{self.config.index_name}' with {self.config.dimensions} dimensions")
            
            # Create index using tree-AH algorithm
            self.index = MatchingEngineIndex.create_tree_ah_index(
                display_name=self.config.index_name,
                dimensions=self.config.dimensions,
                approximate_neighbors_count=self.config.approximate_neighbors_count,
                distance_measure_type=self.config.distance_measure_type,
                leaf_node_embedding_count=self.config.algorithm_config.get("treeAhConfig", {}).get("leafNodeEmbeddingCount", 500),
                leaf_nodes_to_search_percent=self.config.algorithm_config.get("treeAhConfig", {}).get("leafNodesToSearchPercent", 10),
                labels=metadata or {}
            )
            
            logger.info(f"Index created successfully: {self.index.resource_name}")
            return self.index.resource_name
            
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise

    def get_index(self, index_resource_name: str) -> Optional[MatchingEngineIndex]:
        """Get an existing index by resource name."""
        try:
            self.index = MatchingEngineIndex(index_resource_name)
            logger.info(f"Retrieved index: {index_resource_name}")
            return self.index
        except Exception as e:
            logger.error(f"Error retrieving index {index_resource_name}: {e}")
            return None

    def list_indexes(self) -> List[MatchingEngineIndex]:
        """List all indexes in the project."""
        try:
            indexes = MatchingEngineIndex.list()
            logger.info(f"Found {len(indexes)} indexes")
            return indexes
        except Exception as e:
            logger.error(f"Error listing indexes: {e}")
            return []

    def delete_index(self, force: bool = False) -> bool:
        """Delete the current index."""
        try:
            if not self.index:
                logger.warning("No index to delete")
                return False
            
            self.index.delete(force=force)
            logger.info(f"Index {self.index.resource_name} deleted successfully")
            self.index = None
            return True
            
        except Exception as e:
            logger.error(f"Error deleting index: {e}")
            return False

    def upload_embeddings(self, embeddings: List[Embedding], bucket_name: str, 
                         file_prefix: Optional[str] = None) -> str:
        """Upload embeddings to GCS and update the index."""
        try:
            # Ensure bucket exists
            self._ensure_bucket_exists(bucket_name)
            
            # Prepare embeddings data in JSONL format
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Create a unique subdirectory for this batch
            batch_subdirectory = f"batch_{timestamp}"
            gcs_embeddings_directory_path = f"embeddings/{batch_subdirectory}"
            
            filename = f"embeddings.json" # Changed from .jsonl to .json
            local_file_path = f"/tmp/{filename}" # Temporary local file
            
            logger.info(f"Writing {len(embeddings)} embeddings to local temp file: {local_file_path}")
            with open(local_file_path, 'w') as f:
                for embedding in embeddings:
                    embedding_data = {
                        "id": embedding.document_id,
                        "embedding": embedding.embedding,
                        "restricts": [
                            {"namespace": "category", "allow": [embedding.category or "default"]},
                            {"namespace": "source_type", "allow": [embedding.metadata.get("source_type", "unknown")]}
                        ]
                    }
                    f.write(json.dumps(embedding_data) + '\n')
            logger.info(f"Successfully wrote embeddings to local file: {local_file_path}")
            
            # Upload to GCS into the unique subdirectory
            if not self.storage_client:
                logger.error("GCS Storage client not initialized!")
                raise ConnectionError("GCS Storage client not initialized in VectorSearchIndexManager")
                
            bucket = self.storage_client.bucket(bucket_name)
            gcs_file_path_in_bucket = f"{gcs_embeddings_directory_path}/{filename}"
            logger.info(f"Attempting to upload {local_file_path} to GCS path: gs://{bucket_name}/{gcs_file_path_in_bucket}")
            blob = bucket.blob(gcs_file_path_in_bucket)
            
            try:
                blob.upload_from_filename(local_file_path)
                logger.info(f"Successfully uploaded to GCS: gs://{bucket_name}/{gcs_file_path_in_bucket}")
            except Exception as gcs_upload_error:
                logger.error(f"GCS UPLOAD FAILED for {local_file_path} to gs://{bucket_name}/{gcs_file_path_in_bucket}: {gcs_upload_error}", exc_info=True)
                raise # Re-raise the GCS upload error to be caught by the outer try-except
            
            # Clean up local file
            try:
                os.remove(local_file_path)
                logger.info(f"Successfully removed local temp file: {local_file_path}")
            except Exception as local_remove_error:
                logger.warning(f"Could not remove local temp file {local_file_path}: {local_remove_error}")
            
            gcs_file_uri = f"gs://{bucket_name}/{gcs_file_path_in_bucket}"
            logger.info(f"Uploaded {len(embeddings)} embeddings to {gcs_file_uri}")
            
            # The update_embeddings method expects a GCS *directory* URI.
            gcs_batch_directory_uri = f"gs://{bucket_name}/{gcs_embeddings_directory_path}/"
            # Ensure trailing slash for a directory URI

            # Update index with new embeddings
            if self.index:
                logger.info(f"Updating index with embeddings from GCS directory: {gcs_batch_directory_uri}")
                update_lro = self.index.update_embeddings(contents_delta_uri=gcs_batch_directory_uri)
                logger.info(f"Index update LRO initiated: {update_lro.operation.name}. Waiting for completion...")
                
                # Wait for the LRO to complete
                update_lro.result() # This will block until the LRO is done
                
                # Check LRO status (though result() would raise an exception on failure)
                if update_lro.done() and not update_lro.cancelled() and not update_lro.exception():
                    logger.info(f"Index update process completed successfully for GCS directory: {gcs_batch_directory_uri}")
                else:
                    if update_lro.exception():
                        logger.error(f"Index update LRO failed with exception: {update_lro.exception()}")
                        raise update_lro.exception() # Re-raise the LRO exception
                    else:
                        logger.warning(f"Index update LRO finished but may not have succeeded (cancelled: {update_lro.cancelled()}). Operation: {update_lro.operation.name}")
            
            return gcs_file_uri # Return the specific file URI as it's still useful info
            
        except Exception as e:
            logger.error(f"Error uploading embeddings: {e}")
            raise

    def _ensure_bucket_exists(self, bucket_name: str) -> None:
        """Ensure the GCS bucket exists, create if it doesn't."""
        try:
            bucket = self.storage_client.bucket(bucket_name)
            bucket.reload()
            logger.info(f"Bucket {bucket_name} exists")
        except Exception:
            # Bucket doesn't exist, create it
            bucket = self.storage_client.create_bucket(bucket_name, location=self.config.location)
            logger.info(f"Created bucket {bucket_name}")

    def create_endpoint(self, endpoint_name: Optional[str] = None) -> str:
        """Create a new endpoint for serving the index."""
        try:
            display_name = endpoint_name or f"{self.config.index_name}-endpoint"
            
            self.endpoint = MatchingEngineIndexEndpoint.create(
                display_name=display_name,
                public_endpoint_enabled=True
            )
            
            logger.info(f"Endpoint created successfully: {self.endpoint.resource_name}")
            return self.endpoint.resource_name
            
        except Exception as e:
            logger.error(f"Error creating endpoint: {e}")
            raise

    def get_endpoint(self, endpoint_resource_name: str) -> Optional[MatchingEngineIndexEndpoint]:
        """Get an existing endpoint by resource name."""
        try:
            self.endpoint = MatchingEngineIndexEndpoint(endpoint_resource_name)
            logger.info(f"Retrieved endpoint: {endpoint_resource_name}")
            return self.endpoint
        except Exception as e:
            logger.error(f"Error retrieving endpoint {endpoint_resource_name}: {e}")
            return None

    def deploy_index(self, gcs_uri: str, endpoint_resource_name: Optional[str] = None, 
                    min_replica_count: int = 1, max_replica_count: int = 1) -> str:
        """Deploy the index to an endpoint."""
        try:
            if not self.index:
                raise Exception("No index available. Create or load an index first.")
            
            # Get or create endpoint
            if endpoint_resource_name:
                self.get_endpoint(endpoint_resource_name)
            elif not self.endpoint:
                self.create_endpoint()
            
            # Generate unique deployed index ID
            self.deployed_index_id = f"deployed_{self.config.index_name}_{int(time.time())}"
            
            # Deploy index to endpoint
            self.endpoint.deploy_index(
                index=self.index,
                deployed_index_id=self.deployed_index_id,
                display_name=self.deployed_index_id,
                min_replica_count=min_replica_count,
                max_replica_count=max_replica_count
            )
            
            logger.info(f"Index deployed successfully with ID: {self.deployed_index_id}")
            return self.endpoint.resource_name
            
        except Exception as e:
            logger.error(f"Error deploying index: {e}")
            raise

    def _wait_for_index_ready(self, timeout: int = 3600, check_interval: int = 30) -> None:
        """Wait for index to be ready after embedding update."""
        logger.info("Waiting for index to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Refresh index state
                self.index = self.index._gca_resource
                state = self.index.index_stats.vectors_count if hasattr(self.index, 'index_stats') else 0
                
                if state > 0:
                    logger.info(f"Index is ready with {state} vectors")
                    return
                    
                logger.info(f"Index not ready yet, waiting {check_interval} seconds...")
                time.sleep(check_interval)
                
            except Exception as e:
                logger.warning(f"Error checking index status: {e}")
                time.sleep(check_interval)
        
        raise TimeoutError(f"Index not ready after {timeout} seconds")

    def search(self, query_embedding: List[float], num_neighbors: int = 5, 
              filter_criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search the index for similar vectors."""
        try:
            if not self.endpoint or not self.deployed_index_id:
                raise Exception("Index not deployed. Call deploy_index() first.")
            
            # Prepare the query instance for Vertex AI Vector Search
            query_instance = {
                "deployed_index_id": self.deployed_index_id,
                "queries": [{
                    "embedding": query_embedding,
                    "neighbor_count": num_neighbors
                }]
            }
            
            if filter_criteria:
                query_instance["queries"][0]["filters"] = filter_criteria
            
            # Call predict directly with endpoint and instances
            response = self.prediction_client.predict(
                endpoint=self.endpoint.resource_name,
                instances=[query_instance]
            )
            
            # Parse results
            results = []
            if response.predictions:
                for prediction in response.predictions:
                    # Parse the prediction response format
                    if isinstance(prediction, Value):
                        prediction_dict = json_format.MessageToDict(prediction)
                    else:
                        prediction_dict = prediction
                    
                    neighbors = prediction_dict.get("neighbors", [])
                    for neighbor in neighbors:
                        results.append({
                            "id": neighbor.get("id"),
                            "distance": neighbor.get("distance"),
                            "score": 1 - neighbor.get("distance", 1)  # Convert distance to similarity score
                        })
            
            logger.info(f"Search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error searching index: {e}")
            raise

    def batch_search(self, query_embeddings: List[List[float]], num_neighbors: int = 5) -> List[List[Dict[str, Any]]]:
        """Perform batch search with multiple query embeddings."""
        try:
            if not self.endpoint or not self.deployed_index_id:
                raise Exception("Index not deployed. Call deploy_index() first.")
            
            # Prepare batch queries
            queries = []
            for embedding in query_embeddings:
                queries.append({
                    "embedding": embedding,
                    "neighbor_count": num_neighbors
                })
            
            query_instance = {
                "deployed_index_id": self.deployed_index_id,
                "queries": queries
            }
            
            # Call predict directly with endpoint and instances
            response = self.prediction_client.predict(
                endpoint=self.endpoint.resource_name,
                instances=[query_instance]
            )
            
            # Parse results for each query
            batch_results = []
            if response.predictions:
                for prediction in response.predictions:
                    if isinstance(prediction, Value):
                        prediction_dict = json_format.MessageToDict(prediction)
                    else:
                        prediction_dict = prediction
                    
                    # Each prediction should contain results for all queries
                    if "neighbors" in prediction_dict:
                        # Single query result format
                        query_results = []
                        neighbors = prediction_dict.get("neighbors", [])
                        for neighbor in neighbors:
                            query_results.append({
                                "id": neighbor.get("id"),
                                "distance": neighbor.get("distance"),
                                "score": 1 - neighbor.get("distance", 1)
                            })
                        batch_results.append(query_results)
                    elif "results" in prediction_dict:
                        # Batch query result format
                        for result in prediction_dict["results"]:
                            query_results = []
                            neighbors = result.get("neighbors", [])
                            for neighbor in neighbors:
                                query_results.append({
                                    "id": neighbor.get("id"),
                                    "distance": neighbor.get("distance"),
                                    "score": 1 - neighbor.get("distance", 1)
                                })
                            batch_results.append(query_results)
            
            logger.info(f"Batch search processed {len(query_embeddings)} queries")
            return batch_results
            
        except Exception as e:
            logger.error(f"Error in batch search: {e}")
            raise

    def undeploy_index(self) -> bool:
        """Undeploy the index from the endpoint."""
        try:
            if not self.endpoint or not self.deployed_index_id:
                logger.warning("No deployed index to undeploy")
                return False
            
            self.endpoint.undeploy_index(deployed_index_id=self.deployed_index_id)
            logger.info(f"Index {self.deployed_index_id} undeployed successfully")
            self.deployed_index_id = None
            return True
            
        except Exception as e:
            logger.error(f"Error undeploying index: {e}")
            return False

    def delete_endpoint(self, force: bool = False) -> bool:
        """Delete the endpoint."""
        try:
            if not self.endpoint:
                logger.warning("No endpoint to delete")
                return False
            
            self.endpoint.delete(force=force)
            logger.info(f"Endpoint {self.endpoint.resource_name} deleted successfully")
            self.endpoint = None
            return True
            
        except Exception as e:
            logger.error(f"Error deleting endpoint: {e}")
            return False

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index."""
        try:
            if not self.index:
                return {}
            
            # Refresh index to get latest stats
            self.index = self.index._gca_resource
            
            stats = {
                "index_name": self.index.display_name,
                "resource_name": self.index.name,
                "dimensions": self.config.dimensions,
                "distance_measure": self.config.distance_measure_type,
                "created_time": str(self.index.create_time),
                "updated_time": str(self.index.update_time)
            }
            
            if hasattr(self.index, 'index_stats') and self.index.index_stats:
                stats.update({
                    "vectors_count": self.index.index_stats.vectors_count,
                    "shards_count": self.index.index_stats.shards_count
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {}

    def cleanup(self, delete_index: bool = False, delete_endpoint: bool = False) -> None:
        """Clean up resources."""
        try:
            if self.deployed_index_id:
                self.undeploy_index()
            
            if delete_endpoint and self.endpoint:
                self.delete_endpoint(force=True)
            
            if delete_index and self.index:
                self.delete_index(force=True)
                
            logger.info("Cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            raise 