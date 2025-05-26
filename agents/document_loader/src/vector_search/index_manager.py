from typing import List, Optional, Dict, Any, Tuple
from google.cloud import aiplatform
from google.cloud import storage
from google.cloud.aiplatform import MatchingEngineIndex, MatchingEngineIndexEndpoint
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import MatchNeighbor, Namespace
import json
import os
import logging
import time
from ..models.base import Embedding, VectorSearchConfig
from datetime import datetime
from google.cloud.aiplatform_v1 import PredictionServiceClient
from google.cloud.aiplatform_v1.types import PredictRequest
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

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
            if file_prefix:
                batch_subdirectory = f"{file_prefix}_batch_{timestamp}"
            else:
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
                            {"namespace": "category", "allow_list": [embedding.category or "default"]},
                            {"namespace": "source_type", "allow_list": [embedding.metadata.get("source_type", "unknown")]}
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
                logger.info(f"Index update LRO initiated: {update_lro.name}. Waiting for completion...")
                
                # Wait for the LRO to complete
                start_time_lro_wait = time.time()
                logger.info(f"Waiting for LRO {update_lro.name} to complete (blocking call to .result())...")
                update_lro.result() # This will block until the LRO is done
                end_time_lro_wait = time.time()
                logger.info(f"LRO {update_lro.name} .result() call finished. Duration: {end_time_lro_wait - start_time_lro_wait:.2f} seconds.")
                
                # Check LRO status (though result() would raise an exception on failure)
                if update_lro.done() and not update_lro.cancelled() and not update_lro.exception():
                    logger.info(f"Index update process completed successfully for GCS directory: {gcs_batch_directory_uri}")
                else:
                    if update_lro.exception():
                        lro_exception = update_lro.exception()
                        logger.error(f"Index update LRO failed with exception: {lro_exception}")
                        # Ensure we are raising a proper exception instance
                        if isinstance(lro_exception, BaseException):
                            raise lro_exception
                        else:
                            # If the mock (or actual object) isn't a proper exception, wrap it
                            raise RuntimeError(f"LRO failed with non-exception object: {lro_exception}")
                    else:
                        logger.warning(f"Index update LRO finished but may not have succeeded (cancelled: {update_lro.cancelled()}). Operation: {update_lro.name}")
            
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

    def ensure_index_deployed(self, endpoint_display_name: str,
                              min_replica_count: int = 1, max_replica_count: int = 1) -> bool:
        """Ensures the current self.index is deployed to an endpoint with the given display name.
           Creates the endpoint if it doesn't exist. Deploys the index if not already deployed there.
           Returns True if successfully deployed (or already deployed), False otherwise.
        """
        if not self.index:
            logger.error("No index loaded (self.index is None). Cannot ensure deployment.")
            return False

        found_existing_deployment = False
        active_endpoint_obj = None  # To store the endpoint object if found or created

        try:
            endpoints = MatchingEngineIndexEndpoint.list()
            for ep in endpoints:
                logger.debug(f"Checking endpoint: DisplayName='{ep.display_name}', Name='{ep.name}'")
                if ep.display_name == endpoint_display_name:
                    logger.info(f"Found existing endpoint '{endpoint_display_name}' with resource name: {ep.name}")
                    active_endpoint_obj = ep
                    
                    logger.debug(f"Endpoint {ep.name} has {len(ep.deployed_indexes)} deployed index(es).")
                    if not self.index:  # Should not happen
                        logger.error("self.index became None unexpectedly during endpoint check.")
                        return False
                    logger.debug(f"Target index for deployment: self.index.name='{self.index.name}', self.index.display_name='{self.index.display_name}'")
                    logger.debug(f"  (Target attributes: ProjID='{self.index.project}', Loc='{self.index.location}', IndexID='{self.index.name.split('/')[-1]}')")

                    for deployed_index_obj in ep.deployed_indexes:
                        # deployed_index_obj.index is like "projects/PROJECT_NUMBER/locations/LOCATION/indexes/INDEX_ID"
                        
                        expected_location = self.index.location
                        expected_index_id = self.index.name.split('/')[-1]

                        deployed_name_parts = deployed_index_obj.index.split('/')
                        # actual_project_identifier = deployed_name_parts[1] # This is the project number or ID
                        actual_location_str = deployed_name_parts[3]
                        actual_index_id_str = deployed_name_parts[5]

                        logger.debug(f"  Inspecting deployed index on endpoint: DeployedIndex.id='{deployed_index_obj.id}', DeployedIndex.index (resource_name)='{deployed_index_obj.index}', DeployedIndex.display_name='{deployed_index_obj.display_name}'")
                        logger.debug(f"    Parsed from DeployedIndex.index: Loc='{actual_location_str}', IndexID='{actual_index_id_str}'")
                        
                        # Match if location and the final index ID are the same.
                        # The project part of deployed_index_obj.index uses Project Number, while self.index.project is Project ID.
                        # As long as the endpoint 'ep' was found correctly (assumed to be in our target project by its display_name),
                        # matching location and index ID should be sufficient.
                        match = (
                            actual_location_str == expected_location and
                            actual_index_id_str == expected_index_id
                        )

                        if match:
                            logger.info(f"SUCCESS: Index (Loc:{expected_location}, ID:{expected_index_id}) is ALREADY DEPLOYED to endpoint {ep.name} with DeployedIndex.id: {deployed_index_obj.id}")
                            self.endpoint = active_endpoint_obj
                            self.deployed_index_id = deployed_index_obj.id
                            found_existing_deployment = True
                            return True # Critical: Return True as soon as existing deployment is confirmed
                        else:
                            logger.debug(f"    MISMATCH with target. LocMatch={actual_location_str == expected_location}, IndexIDMatch={actual_index_id_str == expected_index_id}")
                    
                    if not found_existing_deployment:
                         logger.info(f"Index {self.index.name} (DisplayName: {self.index.display_name}) was NOT FOUND on existing endpoint {ep.name} (DisplayName: {ep.display_name}). Will proceed to deploy to this endpoint.")
                    # Important: if endpoint is found, we break out of the ENDPOINT loop to use this active_endpoint_obj
                    break 
            
            if not active_endpoint_obj:
                logger.info(f"Endpoint '{endpoint_display_name}' not found. Creating it...")
                created_endpoint_name = self.create_endpoint(endpoint_name=endpoint_display_name)
                if not self.endpoint or not created_endpoint_name:
                    logger.error(f"Failed to create or retrieve endpoint '{endpoint_display_name}'. self.endpoint is {self.endpoint}")
                    return False
                active_endpoint_obj = self.endpoint
                logger.info(f"Successfully created endpoint: {active_endpoint_obj.name} (DisplayName: {active_endpoint_obj.display_name})")
            
            # If found_existing_deployment is True here, it means we returned from inside the loop.
            # This part of the code should only be reached if a deployment is needed.
            if found_existing_deployment:
                 # This state should ideally not be reached if return True was effective.
                 logger.warning("Reached post-loop check with found_existing_deployment=True. This might indicate a logic flow issue if not intended.")
                 if self.endpoint and self.deployed_index_id:
                     return True # If state is consistent, still okay.
                 else:
                     logger.error("Inconsistent state: found_existing_deployment is true, but endpoint/deployed_index_id might be missing.")
                     return False

            logger.info(f"Proceeding to deploy index {self.index.name} (DisplayName: {self.index.display_name}) to endpoint {active_endpoint_obj.name} (DisplayName: {active_endpoint_obj.display_name})...")
            self.endpoint = active_endpoint_obj # Ensure self.endpoint is the one we want to deploy to

            index_name_part = self.index.name.split('/')[-1][:30].replace('-', '_').replace('.', '_') # Sanitize further for ID
            deployment_id = f"dp_{index_name_part}_{int(time.time())}"
            logger.info(f"Generated deployment_id for Vertex AI: {deployment_id}")

            deploy_lro = self.endpoint.deploy_index(
                index=self.index,
                deployed_index_id=deployment_id,
                display_name=f"Deployment of {self.index.display_name[:30]} at {int(time.time())}",
                min_replica_count=min_replica_count,
                max_replica_count=max_replica_count,
            )
            logger.info(f"Deploy index LRO initiated: {deploy_lro.operation.name}. Waiting for completion (this may take 20-60 minutes for new deployments)...")
            deploy_lro.result() 
            logger.info(f"Index {self.index.name} successfully deployed to endpoint {self.endpoint.name}. Target DeployedIndex ID was: {deployment_id}")
            
            # After deployment, confirm the deployed_index_id by fetching a fresh state of the endpoint.
            refreshed_deployed_id_from_new_deploy = None
            try:
                freshly_retrieved_endpoint = MatchingEngineIndexEndpoint(self.endpoint.name)
                for dep_idx in freshly_retrieved_endpoint.deployed_indexes:
                    # We are looking for the specific deployment we just made.
                    if dep_idx.index == self.index.name and dep_idx.id == deployment_id:
                        refreshed_deployed_id_from_new_deploy = dep_idx.id
                        break
            except Exception as ex_refresh:
                logger.warning(f"Could not refresh endpoint state after deployment to confirm deployed_index_id: {ex_refresh}")
            
            if refreshed_deployed_id_from_new_deploy:
                self.deployed_index_id = refreshed_deployed_id_from_new_deploy
                logger.info(f"Confirmed DeployedIndex ID after new deployment: {self.deployed_index_id}")
            else:
                self.deployed_index_id = deployment_id # Fallback to the ID we generated
                logger.warning(f"Could not confirm DeployedIndex ID by refreshing endpoint after new deployment. Using generated ID: {self.deployed_index_id}. Manual check advised.")
            return True

        except Exception as ex: # Changed 'e' to 'ex'
            logger.error(f"Error in ensure_index_deployed for index '{self.index.name if self.index else 'Unknown Index'}' to endpoint '{endpoint_display_name}': {ex}", exc_info=True)
            if self.index:
                logger.error(f"Context: Index DisplayName='{self.index.display_name}'")
            return False

    def upsert_datapoints(self, embeddings: List[Embedding]) -> None:
        """Upsert datapoints (embeddings) to the index using streaming. 
           Assumes self.index is already set and the index is deployed to an endpoint.
        """
        if not self.index:
            logger.error("No index available. Cannot upsert datapoints.")
            raise ValueError("Index not initialized. Call get_index() first.")

        if not self.endpoint or not self.deployed_index_id:
            # This check might be too strict if upsert_datapoints can work on an index
            # not yet deployed for querying, but typically an endpoint is needed for writes too.
            # For now, let's assume it requires a deployed index as per search().
            # The SDK might allow upserting to an index resource directly without an endpoint, 
            # but then searching that index would require deploying it.
            # Let's assume for consistency with search, an endpoint context is good.
            # UPDATE: According to Vertex AI docs, upsert_datapoints is a method of MatchingEngineIndex,
            # not MatchingEngineIndexEndpoint. So, we only need self.index.
            # We still need the index to be *configured* for streaming updates.
            pass # Endpoint not strictly needed for index.upsert_datapoints

        datapoints_to_upsert = []
        for embedding in embeddings:
            # Ensure embedding vector is a list of floats as expected by the SDK
            # The model produces numpy arrays or lists, ensure it's the latter for JSON serialization if needed by underlying calls,
            # though upsert_datapoints might handle numpy arrays directly.
            # For safety, let's assume list of floats.
            embedding_vector = [float(val) for val in embedding.embedding]
            
            datapoint = {
                "datapoint_id": embedding.document_id,
                "feature_vector": embedding_vector,
                "restricts": [
                    {"namespace": "category", "allow_list": [embedding.category or "default"]},
                    # Convert source_type from metadata to a restrict
                    {"namespace": "source_type", "allow_list": [embedding.metadata.get("source_type", "unknown")]}
                ]
                # "crowding_tag" can be added if needed
            }
            datapoints_to_upsert.append(datapoint)
        
        if not datapoints_to_upsert:
            logger.info("No datapoints to upsert.")
            return

        try:
            logger.info(f"Upserting {len(datapoints_to_upsert)} datapoints to index {self.index.name} via streaming...")
            # The upsert_datapoints method is on the MatchingEngineIndex object itself.
            self.index.upsert_datapoints(datapoints=datapoints_to_upsert)
            logger.info(f"Successfully initiated upsert for {len(datapoints_to_upsert)} datapoints to index {self.index.name}.")
            # Streaming upserts are generally fast to initiate. 
            # The actual indexing might take a short while in the background.
        except Exception as e:
            logger.error(f"Error upserting datapoints to index {self.index.name}: {e}", exc_info=True)
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
               filter_criteria: Optional[List[Dict[str, List[str]]]] = None) -> List[Dict[str, Any]]:
        """Search the index for similar vectors using the endpoint's find_neighbors method."""
        if not self.endpoint or not self.deployed_index_id:
            logger.error("Cannot search: Index not deployed or endpoint not set.")
            raise Exception("Index not deployed. Ensure deployment first.")

        # Ensure self.endpoint is a full object, not just from a list() call
        try:
            logger.debug(f"Refreshing endpoint object for: {self.endpoint.name}")
            self.endpoint = MatchingEngineIndexEndpoint(self.endpoint.name)
            logger.debug(f"Endpoint object refreshed. Public match client should exist: {hasattr(self.endpoint, '_public_match_client')}")
        except Exception as e_refresh:
            logger.error(f"Failed to refresh endpoint object {self.endpoint.name}: {e_refresh}")
            # Decide if we should raise here or try to proceed with the existing self.endpoint object
            raise # Re-raise for now, as a stale object is likely the cause of the error

        # Convert filter_criteria to List[Namespace] if provided
        sdk_filters: Optional[List[Namespace]] = None
        if filter_criteria:
            sdk_filters = []
            for fc_item in filter_criteria:
                # fc_item is e.g. {"namespace": "category", "allow_list": ["X"]}
                # Note: current Document structure uses "allow_list". Namespace uses "allow_tokens".
                ns_name = fc_item.get("namespace")
                allow_tokens_list = fc_item.get("allow_list") 
                deny_tokens_list = fc_item.get("deny_list")

                if ns_name and (allow_tokens_list or deny_tokens_list) :
                    sdk_filters.append(Namespace(
                        name=ns_name, 
                        allow_tokens=allow_tokens_list or [],
                        deny_tokens=deny_tokens_list or []
                    ))
            if not sdk_filters:
                logger.warning(f"filter_criteria provided but resulted in empty sdk_filters: {filter_criteria}")


        try:
            logger.debug(
                f"Performing find_neighbors on endpoint: {self.endpoint.resource_name}, "
                f"deployed_index_id: {self.deployed_index_id}, num_neighbors: {num_neighbors}"
            )
            if sdk_filters:
                logger.debug(f"  with filters: {[f.name for f in sdk_filters]}")

            # find_neighbors expects a list of queries. We are sending one.
            response: List[List[MatchNeighbor]] = self.endpoint.find_neighbors(
                queries=[query_embedding],
                deployed_index_id=self.deployed_index_id,
                num_neighbors=num_neighbors,
                filter=sdk_filters, # Pass the converted filters
                return_full_datapoint=False # Optional, False by default
            )

            results = []
            # Response is List[List[MatchNeighbor]]. Outer list for queries, inner for neighbors.
            if response and response[0]:
                for neighbor in response[0]:
                    score = None
                    if neighbor.distance is not None:
                        # Assuming smaller distance is better, convert to a similarity score.
                        # For cosine distance (0 to 2), 1 - (distance/2) could map to [0,1] score.
                        # Or simply 1 - distance if distance is already in a good range for this.
                        # Let's use a simple 1.0 - distance for now, and adjust if needed based on typical distance values.
                        score = 1.0 - neighbor.distance
                    results.append({
                        "id": neighbor.id, 
                        "distance": neighbor.distance,
                        "score": score 
                    })
            
            logger.info(f"Search (find_neighbors) returned {len(results)} results for the query.")
            return results

        except Exception as e:
            logger.error(f"Error searching index with find_neighbors: {e}", exc_info=True)
            raise

    def batch_search(self, query_embeddings: List[List[float]], num_neighbors: int = 5) -> List[List[Dict[str, Any]]]:
        """Perform batch search with multiple query embeddings."""
        # TODO: Refactor this to use self.endpoint.find_neighbors if single search works.
        # For now, it still uses the old PredictionServiceClient method.
        try:
            if not self.endpoint or not self.deployed_index_id:
                raise Exception("Index not deployed. Call deploy_index() first.")
            
            if not self.prediction_client: # Prediction client is still used by batch_search
                 logger.error("Prediction client not initialized. Cannot perform batch_search.")
                 raise Exception("Prediction client not initialized.")

            # Prepare batch queries
            queries = []
            for embedding in query_embeddings:
                queries.append({
                    "embedding": embedding,       # Changed to "embedding"
                    "neighborCount": num_neighbors # Changed to "neighborCount"
                })
            
            instances_list = queries # The list of query instances
            instance_values = [json_format.ParseDict(inst, Value()) for inst in instances_list]
            
            predict_parameters = {
                "deployedIndexId": self.deployed_index_id # Camel case
            }
            parameters_value = json_format.ParseDict(predict_parameters, Value())
            
            # Call predict directly with endpoint and instances
            predict_response = self.prediction_client.predict(
                endpoint=self.endpoint.resource_name,
                instances=instance_values,
                parameters=parameters_value
            )
            
            # Parse results for each query
            batch_results = []
            if predict_response.predictions:
                for prediction_value in predict_response.predictions: # This is a list of Value
                    prediction_dict = json_format.MessageToDict(prediction_value)
                    # Each prediction_dict should contain results for one query from the batch.
                    # The structure within prediction_dict is expected to be {"neighbors": [...]}
                    query_results = []
                    neighbors_data = prediction_dict.get("neighbors", [])
                    for neighbor_item in neighbors_data:
                        query_results.append({
                            "id": neighbor_item.get("id") or neighbor_item.get("datapoint", {}).get("datapointId"),
                            "distance": neighbor_item.get("distance")
                        })
                    batch_results.append(query_results)
            
            logger.info(f"Batch search processed {len(query_embeddings)} queries")
            return batch_results
            
        except Exception as e:
            logger.error(f"Error in batch search (still uses PredictionServiceClient): {e}", exc_info=True)
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