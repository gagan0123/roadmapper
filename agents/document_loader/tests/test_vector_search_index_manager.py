import pytest
import json
import os
import time
from unittest.mock import Mock, patch, MagicMock, call
from typing import List, Dict, Any
import tempfile
from datetime import datetime
from google.protobuf.struct_pb2 import Value
from google.protobuf import json_format
from google.cloud import aiplatform
from google.api_core import operations_v1
from google.longrunning import operations_pb2
from google.api_core.operation import Operation as ApiCoreOperation

from agents.document_loader.src.vector_search.index_manager import VectorSearchIndexManager
from agents.document_loader.src.models.base import VectorSearchConfig, Embedding
from google.cloud.aiplatform_v1.types import Index as AiplatformIndex
from google.cloud.aiplatform_v1.types import IndexEndpoint as AiplatformIndexEndpoint


@pytest.fixture
def vector_search_config():
    """Create a test configuration for vector search."""
    return VectorSearchConfig(
        project_id="test-project",
        location="us-central1",
        index_name="test-index",
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


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    return [
        Embedding(
            document_id="doc1",
            embedding=[0.1] * 768,
            model_name="textembedding-gecko@latest",
            category="category1",
            metadata={"source": "test"}
        ),
        Embedding(
            document_id="doc2",
            embedding=[0.2] * 768,
            model_name="textembedding-gecko@latest",
            category="category2",
            metadata={"source": "test"}
        ),
        Embedding(
            document_id="doc3",
            embedding=[0.3] * 768,
            model_name="textembedding-gecko@latest",
            category="category1",
            metadata={"source": "test"}
        )
    ]


@pytest.fixture
def mock_index():
    """Create a mock index object."""
    mock = Mock()
    mock.resource_name = "projects/test-project/locations/us-central1/indexes/12345"
    mock.display_name = "test-index"
    mock.name = "projects/test-project/locations/us-central1/indexes/12345"
    mock.create_time = "2023-01-01T00:00:00Z"
    mock.update_time = "2023-01-01T00:00:00Z"
    mock.index_stats.vectors_count = 1000
    mock.index_stats.shards_count = 1
    mock._gca_resource = mock
    return mock


@pytest.fixture
def mock_endpoint():
    """Create a mock endpoint object."""
    mock = Mock()
    mock.resource_name = "projects/test-project/locations/us-central1/indexEndpoints/67890"
    mock.display_name = "test-index_endpoint"
    mock.name = "projects/test-project/locations/us-central1/indexEndpoints/67890"
    mock._is_complete_object = True
    return mock


class TestVectorSearchIndexManagerInitialization:
    """Test initialization and setup of VectorSearchIndexManager."""
    
    @patch('agents.document_loader.src.vector_search.index_manager.aiplatform')
    @patch('agents.document_loader.src.vector_search.index_manager.storage.Client')
    @patch('agents.document_loader.src.vector_search.index_manager.PredictionServiceClient')
    def test_successful_initialization(self, mock_prediction_client, mock_storage_client, mock_aiplatform, vector_search_config):
        """Test successful initialization of VectorSearchIndexManager."""
        manager = VectorSearchIndexManager(vector_search_config)
        
        # Verify initialization calls
        mock_aiplatform.init.assert_called_once_with(
            project="test-project", 
            location="us-central1"
        )
        mock_storage_client.assert_called_once_with(project="test-project")
        mock_prediction_client.assert_called_once()
        
        # Verify attributes
        assert manager.config == vector_search_config
        assert manager.index is None
        assert manager.endpoint is None
        assert manager.deployed_index_id is None
    
    @patch('agents.document_loader.src.vector_search.index_manager.aiplatform')
    def test_initialization_failure(self, mock_aiplatform, vector_search_config):
        """Test initialization failure handling."""
        mock_aiplatform.init.side_effect = Exception("Initialization failed")
        
        with pytest.raises(Exception, match="Initialization failed"):
            VectorSearchIndexManager(vector_search_config)


class TestIndexManagement:
    """Test index creation, retrieval, and management operations."""
    
    @patch('agents.document_loader.src.vector_search.index_manager.MatchingEngineIndex')
    @patch('agents.document_loader.src.vector_search.index_manager.aiplatform')
    @patch('agents.document_loader.src.vector_search.index_manager.storage.Client')
    @patch('agents.document_loader.src.vector_search.index_manager.PredictionServiceClient')
    def test_create_index_success(self, mock_prediction_client, mock_storage_client, 
                                  mock_aiplatform, mock_matching_engine_index, 
                                  vector_search_config, mock_index):
        """Test successful index creation."""
        mock_matching_engine_index.create_tree_ah_index.return_value = mock_index
        
        manager = VectorSearchIndexManager(vector_search_config)
        result = manager.create_index()
        
        # Verify index creation call
        mock_matching_engine_index.create_tree_ah_index.assert_called_once_with(
            display_name="test-index",
            dimensions=768,
            approximate_neighbors_count=150,
            distance_measure_type="COSINE",
            leaf_node_embedding_count=500,
            leaf_nodes_to_search_percent=10,
            labels={}
        )
        
        assert result == mock_index.resource_name
        assert manager.index == mock_index
    
    @patch('agents.document_loader.src.vector_search.index_manager.MatchingEngineIndex')
    @patch('agents.document_loader.src.vector_search.index_manager.aiplatform')
    @patch('agents.document_loader.src.vector_search.index_manager.storage.Client')
    @patch('agents.document_loader.src.vector_search.index_manager.PredictionServiceClient')
    def test_create_index_with_metadata(self, mock_prediction_client, mock_storage_client,
                                       mock_aiplatform, mock_matching_engine_index,
                                       vector_search_config, mock_index):
        """Test index creation with metadata."""
        mock_matching_engine_index.create_tree_ah_index.return_value = mock_index
        metadata = {"environment": "test", "version": "1.0"}
        
        manager = VectorSearchIndexManager(vector_search_config)
        result = manager.create_index(metadata=metadata)
        
        # Verify metadata is passed
        call_kwargs = mock_matching_engine_index.create_tree_ah_index.call_args[1]
        assert call_kwargs["labels"] == metadata
    
    @patch('agents.document_loader.src.vector_search.index_manager.MatchingEngineIndex')
    @patch('agents.document_loader.src.vector_search.index_manager.aiplatform')
    @patch('agents.document_loader.src.vector_search.index_manager.storage.Client')
    @patch('agents.document_loader.src.vector_search.index_manager.PredictionServiceClient')
    def test_create_index_failure(self, mock_prediction_client, mock_storage_client,
                                  mock_aiplatform, mock_matching_engine_index,
                                  vector_search_config):
        """Test index creation failure."""
        mock_matching_engine_index.create_tree_ah_index.side_effect = Exception("Creation failed")
        
        manager = VectorSearchIndexManager(vector_search_config)
        
        with pytest.raises(Exception, match="Creation failed"):
            manager.create_index()
    
    @patch('agents.document_loader.src.vector_search.index_manager.MatchingEngineIndex')
    @patch('agents.document_loader.src.vector_search.index_manager.aiplatform')
    @patch('agents.document_loader.src.vector_search.index_manager.storage.Client')
    @patch('agents.document_loader.src.vector_search.index_manager.PredictionServiceClient')
    def test_get_index_success(self, mock_prediction_client, mock_storage_client,
                              mock_aiplatform, mock_matching_engine_index,
                              vector_search_config, mock_index):
        """Test successful index retrieval."""
        mock_matching_engine_index.return_value = mock_index
        
        manager = VectorSearchIndexManager(vector_search_config)
        result = manager.get_index("test-resource-name")
        
        mock_matching_engine_index.assert_called_once_with("test-resource-name")
        assert result == mock_index
        assert manager.index == mock_index
    
    @patch('agents.document_loader.src.vector_search.index_manager.MatchingEngineIndex')
    @patch('agents.document_loader.src.vector_search.index_manager.aiplatform')
    @patch('agents.document_loader.src.vector_search.index_manager.storage.Client')
    @patch('agents.document_loader.src.vector_search.index_manager.PredictionServiceClient')
    def test_get_index_failure(self, mock_prediction_client, mock_storage_client,
                              mock_aiplatform, mock_matching_engine_index,
                              vector_search_config):
        """Test index retrieval failure."""
        mock_matching_engine_index.side_effect = Exception("Index not found")
        
        manager = VectorSearchIndexManager(vector_search_config)
        result = manager.get_index("invalid-resource-name")
        
        assert result is None
    
    @patch('agents.document_loader.src.vector_search.index_manager.MatchingEngineIndex')
    @patch('agents.document_loader.src.vector_search.index_manager.aiplatform')
    @patch('agents.document_loader.src.vector_search.index_manager.storage.Client')
    @patch('agents.document_loader.src.vector_search.index_manager.PredictionServiceClient')
    def test_list_indexes(self, mock_prediction_client, mock_storage_client,
                         mock_aiplatform, mock_matching_engine_index,
                         vector_search_config, mock_index):
        """Test listing indexes."""
        mock_matching_engine_index.list.return_value = [mock_index]
        
        manager = VectorSearchIndexManager(vector_search_config)
        result = manager.list_indexes()
        
        mock_matching_engine_index.list.assert_called_once()
        assert result == [mock_index]
    
    @patch('agents.document_loader.src.vector_search.index_manager.aiplatform')
    @patch('agents.document_loader.src.vector_search.index_manager.storage.Client')
    @patch('agents.document_loader.src.vector_search.index_manager.PredictionServiceClient')
    def test_delete_index_success(self, mock_prediction_client, mock_storage_client,
                                  mock_aiplatform, vector_search_config, mock_index):
        """Test successful index deletion."""
        manager = VectorSearchIndexManager(vector_search_config)
        manager.index = mock_index
        
        result = manager.delete_index(force=True)
        
        mock_index.delete.assert_called_once_with(force=True)
        assert result is True
        assert manager.index is None
    
    @patch('agents.document_loader.src.vector_search.index_manager.aiplatform')
    @patch('agents.document_loader.src.vector_search.index_manager.storage.Client')
    @patch('agents.document_loader.src.vector_search.index_manager.PredictionServiceClient')
    def test_delete_index_no_index(self, mock_prediction_client, mock_storage_client,
                                   mock_aiplatform, vector_search_config):
        """Test deleting when no index exists."""
        manager = VectorSearchIndexManager(vector_search_config)
        
        result = manager.delete_index()
        
        assert result is False


class TestEmbeddingOperations:
    """Test embedding upload and management operations."""
    
    @patch('agents.document_loader.src.vector_search.index_manager.aiplatform')
    @patch('agents.document_loader.src.vector_search.index_manager.storage.Client')
    @patch('agents.document_loader.src.vector_search.index_manager.PredictionServiceClient')
    @patch('builtins.open', create=True)
    @patch('os.remove')
    def test_upload_embeddings_success(self, mock_remove, mock_open, mock_prediction_client,
                                      mock_storage_client, mock_aiplatform,
                                      vector_search_config, sample_embeddings):
        """Test successful embedding upload."""
        # Mock storage components
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_storage_instance = Mock()
        mock_storage_instance.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_bucket.exists.return_value = True
        mock_storage_client.return_value = mock_storage_instance
        
        # Mock file operations
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        manager = VectorSearchIndexManager(vector_search_config)
        result = manager.upload_embeddings(sample_embeddings, "test-bucket")
        
        # Verify file operations
        assert mock_open.called
        assert mock_file.write.call_count == len(sample_embeddings)
        
        # Verify cloud storage operations
        mock_blob.upload_from_filename.assert_called_once()
        mock_remove.assert_called_once()
        
        # Verify return value
        assert result.startswith("gs://test-bucket/embeddings/")
    
    @patch('agents.document_loader.src.vector_search.index_manager.aiplatform')
    @patch('agents.document_loader.src.vector_search.index_manager.storage.Client')
    @patch('agents.document_loader.src.vector_search.index_manager.PredictionServiceClient')
    def test_upload_embeddings_with_prefix(self, mock_prediction_client, mock_storage_client,
                                          mock_aiplatform, vector_search_config, sample_embeddings):
        """Test embedding upload with custom prefix."""
        # Mock storage components
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_storage_instance = Mock()
        mock_storage_instance.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_bucket.exists.return_value = True
        mock_storage_client.return_value = mock_storage_instance
        
        with patch('builtins.open', create=True), patch('os.remove'):
            manager = VectorSearchIndexManager(vector_search_config)
            result = manager.upload_embeddings(sample_embeddings, "test-bucket", file_prefix="custom")
            
            assert "custom_" in result
    
    @patch('agents.document_loader.src.vector_search.index_manager.aiplatform')
    @patch('agents.document_loader.src.vector_search.index_manager.storage.Client')
    @patch('agents.document_loader.src.vector_search.index_manager.PredictionServiceClient')
    def test_ensure_bucket_exists_create_new(self, mock_prediction_client, mock_storage_client,
                                            mock_aiplatform, vector_search_config):
        """Test bucket creation when bucket doesn't exist."""
        mock_bucket = Mock()
        mock_bucket.reload.side_effect = Exception("Bucket not found")
        mock_storage_instance = Mock()
        mock_storage_instance.bucket.return_value = mock_bucket
        mock_storage_instance.create_bucket.return_value = mock_bucket
        mock_storage_client.return_value = mock_storage_instance
        
        manager = VectorSearchIndexManager(vector_search_config)
        manager._ensure_bucket_exists("new-bucket")
        
        mock_storage_instance.create_bucket.assert_called_once_with("new-bucket", location="us-central1")


class TestEndpointManagement:
    """Test endpoint creation and management operations."""
    
    @patch('agents.document_loader.src.vector_search.index_manager.MatchingEngineIndexEndpoint')
    @patch('agents.document_loader.src.vector_search.index_manager.aiplatform')
    @patch('agents.document_loader.src.vector_search.index_manager.storage.Client')
    @patch('agents.document_loader.src.vector_search.index_manager.PredictionServiceClient')
    def test_create_endpoint_success(self, mock_prediction_client, mock_storage_client,
                                    mock_aiplatform, mock_matching_engine_endpoint,
                                    vector_search_config, mock_endpoint):
        """Test successful endpoint creation."""
        mock_matching_engine_endpoint.create.return_value = mock_endpoint
        
        manager = VectorSearchIndexManager(vector_search_config)
        result = manager.create_endpoint()
        
        mock_matching_engine_endpoint.create.assert_called_once_with(
            display_name="test-index-endpoint",
            public_endpoint_enabled=True
        )
        
        assert result == mock_endpoint.resource_name
        assert manager.endpoint == mock_endpoint
    
    @patch('agents.document_loader.src.vector_search.index_manager.MatchingEngineIndexEndpoint')
    @patch('agents.document_loader.src.vector_search.index_manager.aiplatform')
    @patch('agents.document_loader.src.vector_search.index_manager.storage.Client')
    @patch('agents.document_loader.src.vector_search.index_manager.PredictionServiceClient')
    def test_create_endpoint_custom_name(self, mock_prediction_client, mock_storage_client,
                                        mock_aiplatform, mock_matching_engine_endpoint,
                                        vector_search_config, mock_endpoint):
        """Test endpoint creation with custom name."""
        mock_matching_engine_endpoint.create.return_value = mock_endpoint
        
        manager = VectorSearchIndexManager(vector_search_config)
        result = manager.create_endpoint("custom-endpoint")
        
        call_kwargs = mock_matching_engine_endpoint.create.call_args[1]
        assert call_kwargs["display_name"] == "custom-endpoint"
    
    @patch('agents.document_loader.src.vector_search.index_manager.MatchingEngineIndexEndpoint')
    @patch('agents.document_loader.src.vector_search.index_manager.aiplatform')
    @patch('agents.document_loader.src.vector_search.index_manager.storage.Client')
    @patch('agents.document_loader.src.vector_search.index_manager.PredictionServiceClient')
    def test_get_endpoint_success(self, mock_prediction_client, mock_storage_client,
                                 mock_aiplatform, mock_matching_engine_endpoint,
                                 vector_search_config, mock_endpoint):
        """Test successful endpoint retrieval."""
        mock_matching_engine_endpoint.return_value = mock_endpoint
        
        manager = VectorSearchIndexManager(vector_search_config)
        result = manager.get_endpoint("test-endpoint-resource")
        
        mock_matching_engine_endpoint.assert_called_once_with("test-endpoint-resource")
        assert result == mock_endpoint
        assert manager.endpoint == mock_endpoint


class TestDeploymentOperations:
    """Test index deployment and undeployment operations."""
    
    @patch('agents.document_loader.src.vector_search.index_manager.time.time')
    @patch('agents.document_loader.src.vector_search.index_manager.time.sleep')
    @patch('agents.document_loader.src.vector_search.index_manager.aiplatform')
    @patch('agents.document_loader.src.vector_search.index_manager.storage.Client')
    @patch('agents.document_loader.src.vector_search.index_manager.PredictionServiceClient')
    def test_deploy_index_success(self, mock_prediction_client, mock_storage_client,
                                  mock_aiplatform, mock_sleep, mock_time,
                                  vector_search_config, mock_index, mock_endpoint):
        """Test successful index deployment."""
        mock_time.return_value = 1234567890
        mock_index.index_stats.vectors_count = 100
        
        manager = VectorSearchIndexManager(vector_search_config)
        manager.index = mock_index
        manager.endpoint = mock_endpoint
        
        result = manager.deploy_index("gs://test-bucket/embeddings.jsonl")
        
        # Verify deployment
        mock_endpoint.deploy_index.assert_called_once()
        deploy_call = mock_endpoint.deploy_index.call_args[1]
        assert deploy_call["index"] == mock_index
        assert deploy_call["deployed_index_id"] == "deployed_test-index_1234567890"
        assert deploy_call["min_replica_count"] == 1
        assert deploy_call["max_replica_count"] == 1
        
        assert result == mock_endpoint.resource_name
        assert manager.deployed_index_id == "deployed_test-index_1234567890"
    
    @patch('agents.document_loader.src.vector_search.index_manager.aiplatform')
    @patch('agents.document_loader.src.vector_search.index_manager.storage.Client')
    @patch('agents.document_loader.src.vector_search.index_manager.PredictionServiceClient')
    def test_deploy_index_no_index(self, mock_prediction_client, mock_storage_client,
                                   mock_aiplatform, vector_search_config):
        """Test deployment failure when no index exists."""
        manager = VectorSearchIndexManager(vector_search_config)
        
        with pytest.raises(Exception, match="No index available"):
            manager.deploy_index("gs://test-bucket/embeddings.jsonl")
    
    @patch('agents.document_loader.src.vector_search.index_manager.aiplatform')
    @patch('agents.document_loader.src.vector_search.index_manager.storage.Client')
    @patch('agents.document_loader.src.vector_search.index_manager.PredictionServiceClient')
    def test_undeploy_index_success(self, mock_prediction_client, mock_storage_client,
                                   mock_aiplatform, vector_search_config, mock_endpoint):
        """Test successful index undeployment."""
        manager = VectorSearchIndexManager(vector_search_config)
        manager.endpoint = mock_endpoint
        manager.deployed_index_id = "test-deployed-index"
        
        result = manager.undeploy_index()
        
        mock_endpoint.undeploy_index.assert_called_once_with(deployed_index_id="test-deployed-index")
        assert result is True
        assert manager.deployed_index_id is None
    
    @patch('agents.document_loader.src.vector_search.index_manager.aiplatform')
    @patch('agents.document_loader.src.vector_search.index_manager.storage.Client')
    @patch('agents.document_loader.src.vector_search.index_manager.PredictionServiceClient')
    def test_undeploy_index_not_deployed(self, mock_prediction_client, mock_storage_client,
                                        mock_aiplatform, vector_search_config):
        """Test undeployment when no index is deployed."""
        manager = VectorSearchIndexManager(vector_search_config)
        
        result = manager.undeploy_index()
        
        assert result is False


class TestSearchOperations:
    """Test vector search operations."""
    
    @patch('agents.document_loader.src.vector_search.index_manager.aiplatform')
    @patch('agents.document_loader.src.vector_search.index_manager.storage.Client')
    @patch('agents.document_loader.src.vector_search.index_manager.PredictionServiceClient')
    def test_search_success(self, mock_prediction_client_cls, mock_storage_client,
                           mock_aiplatform, vector_search_config, mock_endpoint):
        """Test successful vector search."""
        # Mock the find_neighbors method of the endpoint
        mock_endpoint.find_neighbors.return_value = [ # This is a List[List[MatchNeighbor]]
            [ # Results for the first (and only) query
                aiplatform.matching_engine.matching_engine_index_endpoint.MatchNeighbor(id="doc1", distance=0.1),
                aiplatform.matching_engine.matching_engine_index_endpoint.MatchNeighbor(id="doc2", distance=0.2)
            ]
        ]
        
        manager = VectorSearchIndexManager(vector_search_config)
        manager.endpoint = mock_endpoint # mock_endpoint already has _is_complete_object = True
        manager.deployed_index_id = "test-deployed-index"
        
        query_embedding = [0.1] * 768
        num_neighbors_requested = 2
        result = manager.search(query_embedding, num_neighbors=num_neighbors_requested)
        
        # Verify find_neighbors call
        mock_endpoint.find_neighbors.assert_called_once_with(
            queries=[query_embedding],
            deployed_index_id="test-deployed-index",
            num_neighbors=num_neighbors_requested,
            filter=None, # No filter passed in this test
            return_full_datapoint=False 
        )
        
        # Verify results (search method converts MatchNeighbor to dict)
        assert len(result) == 2
        assert result[0]["id"] == "doc1"
        assert result[0]["distance"] == 0.1
        assert result[0]["score"] == 0.9  # 1 - 0.1 for COSINE
        assert result[1]["id"] == "doc2"
        assert result[1]["distance"] == 0.2
        assert result[1]["score"] == 0.8  # 1 - 0.2 for COSINE
    
    @patch('agents.document_loader.src.vector_search.index_manager.aiplatform')
    @patch('agents.document_loader.src.vector_search.index_manager.storage.Client')
    @patch('agents.document_loader.src.vector_search.index_manager.PredictionServiceClient')
    def test_search_not_deployed(self, mock_prediction_client, mock_storage_client,
                                 mock_aiplatform, vector_search_config):
        """Test search failure when index not deployed."""
        manager = VectorSearchIndexManager(vector_search_config)
        
        with pytest.raises(Exception, match="Index not deployed"):
            manager.search([0.1] * 768)
    
    @patch('agents.document_loader.src.vector_search.index_manager.aiplatform')
    @patch('agents.document_loader.src.vector_search.index_manager.storage.Client')
    @patch('agents.document_loader.src.vector_search.index_manager.PredictionServiceClient')
    def test_search_with_filters(self, mock_prediction_client_cls, mock_storage_client,
                                mock_aiplatform, vector_search_config, mock_endpoint):
        """Test search with filter criteria."""
        # Mock the find_neighbors method of the endpoint
        # For this test, let's assume the filter results in one match
        mock_endpoint.find_neighbors.return_value = [
            [aiplatform.matching_engine.matching_engine_index_endpoint.MatchNeighbor(id="doc_filtered", distance=0.3)]
        ]
        
        manager = VectorSearchIndexManager(vector_search_config)
        manager.endpoint = mock_endpoint # mock_endpoint already has _is_complete_object = True
        manager.deployed_index_id = "test-deployed-index"
        
        query_embedding = [0.1] * 768
        num_neighbors_requested = 1
        # Corrected filter_criteria format
        filter_criteria_input = [{"namespace": "category", "allow_list": ["test"]}]
        
        result = manager.search(
            query_embedding,
            num_neighbors=num_neighbors_requested,
            filter_criteria=filter_criteria_input
        )
        
        # Verify find_neighbors call and that the filter was converted correctly
        assert mock_endpoint.find_neighbors.call_count == 1
        call_args, call_kwargs = mock_endpoint.find_neighbors.call_args
        
        assert call_kwargs['queries'] == [query_embedding]
        assert call_kwargs['deployed_index_id'] == "test-deployed-index"
        assert call_kwargs['num_neighbors'] == num_neighbors_requested
        
        # Check the passed filter argument carefully
        passed_filter_arg = call_kwargs['filter'] # This should be a list of Namespace objects
        assert isinstance(passed_filter_arg, list)
        assert len(passed_filter_arg) == 1
        namespace_filter = passed_filter_arg[0]
        assert isinstance(namespace_filter, aiplatform.matching_engine.matching_engine_index_endpoint.Namespace)
        assert namespace_filter.name == "category"
        assert namespace_filter.allow_tokens == ["test"]
        assert namespace_filter.deny_tokens == [] # Ensure deny_tokens is empty as it was not provided
        assert call_kwargs['return_full_datapoint'] == False
        
        # Verify results
        assert len(result) == 1
        assert result[0]["id"] == "doc_filtered"
        assert result[0]["distance"] == 0.3
        assert result[0]["score"] == 0.7 # 1 - 0.3 for COSINE
    
    @patch('agents.document_loader.src.vector_search.index_manager.aiplatform')
    @patch('agents.document_loader.src.vector_search.index_manager.storage.Client')
    @patch('agents.document_loader.src.vector_search.index_manager.PredictionServiceClient')
    def test_batch_search_success(self, mock_prediction_client_cls, mock_storage_client,
                                 mock_aiplatform, vector_search_config, mock_endpoint):
        """Test successful batch search."""
        mock_prediction_client = Mock()
        mock_prediction_client_cls.return_value = mock_prediction_client
        
        # Original problematic mock:
        # mock_response = Mock()
        # mock_response.predictions = [
        #     {
        #         "results": [
        #             {"neighbors": [{"id": "doc1", "distance": 0.1}]},
        #             {"neighbors": [{"id": "doc2", "distance": 0.2}]}
        #         ]
        #     }
        # ]

        # Corrected mock: response.predictions should be a list of protobuf messages
        # The batch_search method iterates through response.predictions, and each
        # element (prediction_value) is passed to json_format.MessageToDict().
        # So, each element must be a protobuf message, not a raw dict.

        pred_val_1_dict = {"neighbors": [{"id": "doc1", "distance": 0.1}]}
        pred_val_1_pb = json_format.ParseDict(pred_val_1_dict, Value())

        pred_val_2_dict = {"neighbors": [{"id": "doc2", "distance": 0.2}]}
        pred_val_2_pb = json_format.ParseDict(pred_val_2_dict, Value())

        mock_response = Mock()
        mock_response.predictions = [pred_val_1_pb, pred_val_2_pb]
        
        mock_prediction_client.predict.return_value = mock_response
        
        manager = VectorSearchIndexManager(vector_search_config)
        manager.endpoint = mock_endpoint
        manager.deployed_index_id = "test-deployed-index"
        
        query_embeddings = [[0.1] * 768, [0.2] * 768]
        result = manager.batch_search(query_embeddings, num_neighbors=1)
        
        # Verify the request format
        call_args = mock_prediction_client.predict.call_args
        # instances_pb will be a list of protobuf Value objects, one for each query
        instances_pb = call_args[1]["instances"]
        parameters_pb = call_args[1]["parameters"]

        assert len(instances_pb) == len(query_embeddings) # Should be 2 in this test

        # Verify each instance
        for i, query_embedding in enumerate(query_embeddings):
            instance_data_dict = json_format.MessageToDict(instances_pb[i])
            assert instance_data_dict["embedding"] == query_embedding
            assert instance_data_dict["neighborCount"] == 1 # num_neighbors is 1 in this test

        # Verify parameters
        parameters_dict = json_format.MessageToDict(parameters_pb)
        assert parameters_dict["deployedIndexId"] == "test-deployed-index"
        
        assert len(result) == 2
        assert len(result[0]) == 1
        assert len(result[1]) == 1
        assert result[0][0]["id"] == "doc1"
        assert result[1][0]["id"] == "doc2"


class TestUtilityOperations:
    """Test utility and management operations."""
    
    @patch('agents.document_loader.src.vector_search.index_manager.aiplatform')
    @patch('agents.document_loader.src.vector_search.index_manager.storage.Client')
    @patch('agents.document_loader.src.vector_search.index_manager.PredictionServiceClient')
    def test_get_index_stats_success(self, mock_prediction_client, mock_storage_client,
                                    mock_aiplatform, vector_search_config, mock_index):
        """Test getting index statistics."""
        manager = VectorSearchIndexManager(vector_search_config)
        manager.index = mock_index
        
        result = manager.get_index_stats()
        
        expected_stats = {
            "index_name": "test-index",
            "resource_name": mock_index.name,
            "dimensions": 768,
            "distance_measure": "COSINE",
            "created_time": "2023-01-01T00:00:00Z",
            "updated_time": "2023-01-01T00:00:00Z",
            "vectors_count": 1000,
            "shards_count": 1
        }
        
        assert result == expected_stats
    
    @patch('agents.document_loader.src.vector_search.index_manager.aiplatform')
    @patch('agents.document_loader.src.vector_search.index_manager.storage.Client')
    @patch('agents.document_loader.src.vector_search.index_manager.PredictionServiceClient')
    def test_get_index_stats_no_index(self, mock_prediction_client, mock_storage_client,
                                     mock_aiplatform, vector_search_config):
        """Test getting stats when no index exists."""
        manager = VectorSearchIndexManager(vector_search_config)
        
        result = manager.get_index_stats()
        
        assert result == {}
    
    @patch('agents.document_loader.src.vector_search.index_manager.aiplatform')
    @patch('agents.document_loader.src.vector_search.index_manager.storage.Client')
    @patch('agents.document_loader.src.vector_search.index_manager.PredictionServiceClient')
    def test_cleanup_full(self, mock_prediction_client, mock_storage_client,
                         mock_aiplatform, vector_search_config, mock_index, mock_endpoint):
        """Test full cleanup operation."""
        manager = VectorSearchIndexManager(vector_search_config)
        manager.index = mock_index
        manager.endpoint = mock_endpoint
        manager.deployed_index_id = "test-deployed-index"
        
        manager.cleanup(delete_index=True, delete_endpoint=True)
        
        # Verify cleanup calls
        mock_endpoint.undeploy_index.assert_called_once_with(deployed_index_id="test-deployed-index")
        mock_endpoint.delete.assert_called_once_with(force=True)
        mock_index.delete.assert_called_once_with(force=True)
    
    @patch('agents.document_loader.src.vector_search.index_manager.aiplatform')
    @patch('agents.document_loader.src.vector_search.index_manager.storage.Client')
    @patch('agents.document_loader.src.vector_search.index_manager.PredictionServiceClient')
    def test_cleanup_undeploy_only(self, mock_prediction_client, mock_storage_client,
                                  mock_aiplatform, vector_search_config, mock_endpoint):
        """Test cleanup with undeploy only."""
        manager = VectorSearchIndexManager(vector_search_config)
        manager.endpoint = mock_endpoint
        manager.deployed_index_id = "test-deployed-index"
        
        manager.cleanup(delete_index=False, delete_endpoint=False)
        
        # Verify only undeploy is called
        mock_endpoint.undeploy_index.assert_called_once_with(deployed_index_id="test-deployed-index")
        mock_endpoint.delete.assert_not_called()


class TestIntegrationScenarios:
    """Test complete integration scenarios."""
    
    @patch('agents.document_loader.src.vector_search.index_manager.time.time')
    @patch('agents.document_loader.src.vector_search.index_manager.time.sleep')
    @patch('agents.document_loader.src.vector_search.index_manager.MatchingEngineIndex')
    @patch('agents.document_loader.src.vector_search.index_manager.MatchingEngineIndexEndpoint')
    @patch('agents.document_loader.src.vector_search.index_manager.aiplatform')
    @patch('agents.document_loader.src.vector_search.index_manager.storage.Client')
    @patch('agents.document_loader.src.vector_search.index_manager.PredictionServiceClient')
    @patch('builtins.open', create=True)
    @patch('os.remove')
    def test_complete_workflow(self, mock_remove, mock_open, mock_prediction_client_cls,
                              mock_storage_client, mock_aiplatform, mock_matching_engine_endpoint,
                              mock_matching_engine_index, mock_sleep, mock_time,
                              vector_search_config, sample_embeddings, mock_index, mock_endpoint):
        """Test complete workflow from index creation to search."""
        # Setup mocks
        mock_time.return_value = 1234567890
        
        # Configure the main index mock for create_tree_ah_index
        mock_created_index_instance = Mock(spec=aiplatform.MatchingEngineIndex)
        mock_created_index_instance.name = "projects/test-project/locations/us-central1/indexes/test-index-123"
        mock_created_index_instance.resource_name = "projects/test-project/locations/us-central1/indexes/test-index-123"
        mock_created_index_instance.index_stats = Mock()
        mock_created_index_instance.index_stats.vectors_count = 100
        mock_matching_engine_index.create_tree_ah_index.return_value = mock_created_index_instance

        # Configure the LRO mock for update_embeddings
        mock_lro_update_embeddings = Mock(spec=ApiCoreOperation)
        mock_lro_update_embeddings.done.return_value = True
        mock_lro_update_embeddings.cancelled.return_value = False
        mock_lro_update_embeddings.exception.return_value = None
        mock_lro_update_embeddings.result.return_value = None
        mock_lro_update_embeddings.operation = Mock()
        mock_lro_update_embeddings.operation.name = "projects/test-project/locations/us-central1/operations/update-op-123"
        mock_created_index_instance.update_embeddings.return_value = mock_lro_update_embeddings 

        # Configure the endpoint mock
        mock_created_endpoint_instance = Mock(spec=aiplatform.MatchingEngineIndexEndpoint)
        mock_created_endpoint_instance.name = "projects/test-project/locations/us-central1/indexEndpoints/test-endpoint-456"
        mock_created_endpoint_instance.resource_name = "projects/test-project/locations/us-central1/indexEndpoints/test-endpoint-456"
        mock_created_endpoint_instance._is_complete_object = True 
        mock_matching_engine_endpoint.create.return_value = mock_created_endpoint_instance
        
        # Mock the find_neighbors call for the search operation
        mock_created_endpoint_instance.find_neighbors.return_value = [
            [aiplatform.matching_engine.matching_engine_index_endpoint.MatchNeighbor(id="doc1", distance=0.1)] # Corresponds to num_neighbors=1
        ]

        # Storage mocks
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_storage_instance = Mock()
        mock_storage_instance.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_bucket.exists.return_value = True
        mock_storage_client.return_value = mock_storage_instance
        
        # Prediction client mock (still needed for other parts, or can be removed if truly unused)
        mock_prediction_client = Mock()
        mock_prediction_client_cls.return_value = mock_prediction_client
        # Remove the mock_response for predict, as search now uses find_neighbors
        # mock_response = Mock()
        # mock_response.predictions = [{"neighbors": [{"id": "doc1", "distance": 0.1}]}]
        # mock_prediction_client.predict.return_value = mock_response
        
        # Execute workflow
        manager = VectorSearchIndexManager(vector_search_config)
        manager.aiplatform = mock_aiplatform
        
        # Create index
        index_resource_name_returned = manager.create_index()
        assert index_resource_name_returned == mock_created_index_instance.resource_name
        assert manager.index is mock_created_index_instance
        
        # Upload embeddings
        gcs_uri = manager.upload_embeddings(sample_embeddings, "test-bucket")
        assert gcs_uri.startswith("gs://test-bucket/embeddings/")
        
        # Deploy index
        endpoint_name = manager.deploy_index(gcs_uri)
        assert endpoint_name == mock_created_endpoint_instance.resource_name
        
        # Search
        results = manager.search([0.1] * 768, num_neighbors=1)
        assert len(results) == 1
        assert results[0]["id"] == "doc1"
        
        # Cleanup
        manager.cleanup(delete_index=True, delete_endpoint=True)


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @patch('agents.document_loader.src.vector_search.index_manager.aiplatform')
    @patch('agents.document_loader.src.vector_search.index_manager.storage.Client')
    @patch('agents.document_loader.src.vector_search.index_manager.PredictionServiceClient')
    @patch('builtins.open', create=True)
    def test_upload_embeddings_file_error(self, mock_open, mock_prediction_client,
                                         mock_storage_client, mock_aiplatform,
                                         vector_search_config, sample_embeddings):
        """Test embedding upload with file error."""
        mock_open.side_effect = IOError("File write error")
        
        manager = VectorSearchIndexManager(vector_search_config)
        
        with pytest.raises(IOError, match="File write error"):
            manager.upload_embeddings(sample_embeddings, "test-bucket")
    
    @patch('agents.document_loader.src.vector_search.index_manager.time.sleep')
    @patch('agents.document_loader.src.vector_search.index_manager.aiplatform')
    @patch('agents.document_loader.src.vector_search.index_manager.storage.Client')
    @patch('agents.document_loader.src.vector_search.index_manager.PredictionServiceClient')
    def test_wait_for_index_timeout(self, mock_prediction_client, mock_storage_client,
                                   mock_aiplatform, mock_sleep, vector_search_config, mock_index):
        """Test index ready timeout."""
        mock_index.index_stats.vectors_count = 0  # Never becomes ready
        
        manager = VectorSearchIndexManager(vector_search_config)
        manager.index = mock_index
        
        with pytest.raises(TimeoutError, match="Index not ready after"):
            manager._wait_for_index_ready(timeout=1, check_interval=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 