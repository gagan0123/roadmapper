from typing import List, Optional, Dict, Any, Tuple
from google.cloud import aiplatform_v1
from google.api_core import exceptions
from ..models.base import Document, Embedding
import asyncio
from tqdm import tqdm
import time
import logging
import random
import tiktoken
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Character limit from the Vertex AI multimodal embedding model error
TEXT_CHAR_LIMIT_FOR_MULTIMODAL = 1023 # Keep it just under 1024 to be safe
DEFAULT_MULTIMODAL_MODEL_ID = "multimodalembedding@001"

class EmbeddingProcessor:
    def __init__(self, project_id: str, location: str = "us-central1", model_id: str = DEFAULT_MULTIMODAL_MODEL_ID, batch_size: int = 5, max_retries: int = 3, verbose: bool = False):
        self.project_id = project_id
        self.location = location
        self.model_id = model_id
        self.api_batch_size = min(batch_size, 5) # Max instances per API call for multimodal (conservative start)
        self.max_retries = max_retries
        self.client = None
        self.endpoint = None
        self.verbose = verbose
        self.max_tokens_per_text_instance = 8192  # For text-only instances (model might still shorten)
        self.max_tokens_for_multimodal_text = 30  # Max tokens for text paired with an image (strict limit)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        # Official limit for 'multimodalembedding@001' is 20MB for the decoded image.
        self.max_image_bytes = 20 * 1024 * 1024  # 20MB
        self.initialize_vertex_ai()

    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text."""
        return len(self.tokenizer.encode(text))

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to a specific maximum token limit."""
        tokens = self.tokenizer.encode(text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            return self.tokenizer.decode(tokens)
        return text

    def initialize_vertex_ai(self) -> bool:
        """Initialize Vertex AI PredictionServiceClient for Multimodal embeddings."""
        try:
            # Ensure aiplatform is initialized for the project and location if not done elsewhere
            # from google.cloud import aiplatform as vertex_ai_platform # Potentially needed
            # vertex_ai_platform.init(project=self.project_id, location=self.location) # Potentially needed
            
            # For multimodal embeddings, the client setup might be slightly different or use specific options
            # For now, assume PredictionServiceClient is still appropriate
            # Credentials should be handled by the environment (GOOGLE_APPLICATION_CREDENTIALS)
            client_options = {"api_endpoint": f"{self.location}-aiplatform.googleapis.com"}
            self.client = aiplatform_v1.PredictionServiceClient(client_options=client_options)
            self.endpoint = f"projects/{self.project_id}/locations/{self.location}/publishers/google/models/{self.model_id}"
            
            # Test the model with a simple multimodal prediction (e.g., text only, as image bytes can be tricky for a simple init test)
            # The actual input structure for multimodalembedding@001 requires careful formatting.
            # A simple text instance:
            test_instance = [{"text": "test document"}]
            # A more complex one might involve image bytes, but let's keep the init test simple.
            # Ensure this matches what the model expects. For `multimodalembedding@001`, it expects a list of instances.

            # The predict call itself might need to be asynchronous if the client supports it, 
            # or wrapped if it's blocking and called from async code later.
            # For initialization, a synchronous call is usually fine.
            response = self.client.predict(endpoint=self.endpoint, instances=test_instance)
            
            if not response or not response.predictions:
                logger.error(f"Failed to get a valid response from Vertex AI multimodal embedding model: {self.endpoint}")
                # Log the full response for debugging if it's not too large
                # logger.error(f"Full response: {response}")
                raise ValueError(f"Failed to get embeddings from Vertex AI multimodal model {self.model_id}")
            
            # Check the structure of the prediction
            # Example: ensure it contains an 'embeddings' field or similar based on model docs
            # if not isinstance(response.predictions[0], dict) or "embedding" not in response.predictions[0]:
            #    logger.error(f"Unexpected prediction format: {response.predictions[0]}")
            #    raise ValueError("Unexpected prediction format from multimodal model")

            logger.info(f"Successfully initialized Vertex AI multimodal embedding model: {self.model_id}")
            return True
        except exceptions.PermissionDenied as e:
            logger.error(f"Permission denied when initializing Vertex AI: {e}")
            logger.error("Please ensure the service account has the following roles:")
            logger.error("- roles/aiplatform.user")
            logger.error("- roles/aiplatform.serviceAgent")
            raise
        except exceptions.NotFound as e:
            logger.error(f"Vertex AI API not enabled or model not found: {e}")
            logger.error("Please enable the Vertex AI API in your Google Cloud project")
            raise
        except Exception as e:
            logger.error(f"Error initializing Vertex AI: {e}")
            raise

    async def _retry_with_backoff(self, func, *args, **kwargs) -> Optional[List]:
        """Execute a function with exponential backoff retry logic."""
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except exceptions.ResourceExhausted as e:
                if attempt == self.max_retries - 1:
                    raise
                wait_time = (2 ** attempt) + (random.random() * 0.1)
                logger.warning(f"Rate limit exceeded, retrying in {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                wait_time = (2 ** attempt) + (random.random() * 0.1)
                logger.warning(f"Error occurred, retrying in {wait_time:.2f} seconds... Error: {e}")
                await asyncio.sleep(wait_time)
        return None

    async def _process_batch(self, prepared_instances: List[Dict[str, Any]], source_metadata_for_batch: List[Dict[str, Any]]) -> List[Embedding]:
        """Process a single batch of prepared instances using the multimodal embedding model (async)."""
        if not prepared_instances:
            return []

        loop = asyncio.get_event_loop()
        try:
            # Run the blocking predict call in a thread pool executor
            # Ensure arguments are passed as keyword arguments to the predict method
            response = await loop.run_in_executor(
                None, # Uses default ThreadPoolExecutor
                lambda: self.client.predict(endpoint=self.endpoint, instances=prepared_instances)
            )
        except Exception as e:
            logger.error(f"API call to Vertex AI predict failed: {e}")
            # Depending on policy, could re-raise or return empty to let _retry_with_backoff handle it
            raise # Re-raise to allow _retry_with_backoff to see the original error type

        raw_embeddings = response.predictions
        
        final_embeddings = []
        if len(raw_embeddings) != len(prepared_instances):
            logger.error(f"Mismatch between number of instances sent ({len(prepared_instances)}) and embeddings received ({len(raw_embeddings)}). Skipping batch.")
            # Potentially log more details about the request and response here
            return []

        for i, raw_embedding_result in enumerate(raw_embeddings):
            # Multimodal embedding models usually return a dict like {"embedding": [values...]} or similar
            # We need to confirm the exact key for the vector list from documentation
            # Assuming it might be just a list of floats directly in the prediction object, or under a specific key.
            # Let's assume the prediction object *is* the list of floats for now, or it's in prediction.values
            # This part WILL LIKELY NEED ADJUSTMENT based on actual API response structure.
            vector = None
            if isinstance(raw_embedding_result, list): # Direct list of floats
                vector = raw_embedding_result
            elif hasattr(raw_embedding_result, 'values') and isinstance(raw_embedding_result.values, list): # Protobuf-like .values
                vector = raw_embedding_result.values
            elif isinstance(raw_embedding_result, dict) and 'embedding' in raw_embedding_result and isinstance(raw_embedding_result['embedding'], list):
                vector = raw_embedding_result['embedding'] # Common pattern
            elif isinstance(raw_embedding_result, dict) and 'values' in raw_embedding_result and isinstance(raw_embedding_result['values'], list):
                 vector = raw_embedding_result['values'] # As seen with gemini-embedding-001 text model

            if not vector or not isinstance(vector, list):
                logger.error(f"Skipping instance {i}: No valid embedding vector found in result: {raw_embedding_result}")
                continue
            
            source_info = source_metadata_for_batch[i]
            doc_embedding = Embedding(
                document_id=source_info['unique_instance_id'], 
                embedding=vector,
                model_name=self.model_id,
                category=source_info.get('category'), # Assign category to Embedding object
                metadata={
                    'original_doc_id': source_info['original_doc_id'], 
                    'source_type': source_info['source_type'],
                    'source_url': source_info['source_url'],
                    'original_title': source_info['original_doc_title'],
                    'category': source_info.get('category'), # Also include category in metadata for completeness
                    'instance_type': source_info['instance_type'], 
                    'image_index': source_info.get('image_index'), 
                    'original_image_index_in_document': source_info.get('original_image_index_in_document'),
                    'embedding_model_used': self.model_id
                }
            )
            final_embeddings.append(doc_embedding)
            
        if self.verbose and final_embeddings:
            logger.debug(f"Processed batch. Generated {len(final_embeddings)} embeddings. Sample: {final_embeddings[0].vector[:5]}")
        elif not final_embeddings and prepared_instances:
            logger.warning(f"Processed batch but generated 0 embeddings from {len(prepared_instances)} instances.")
            
        return final_embeddings

    def _prepare_batch(self, documents_queue: List[Document]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
        """
        Prepare a batch of API instances from a queue of documents for multimodal embedding.
        Returns a tuple: (api_instances_batch, source_metadata_batch, documents_consumed_count)
        Each document can yield multiple API instances: one for its text (if any), and one for each image (if any).
        Stops when api_batch_size is reached or documents_queue is empty.
        """
        api_instances_batch: List[Dict[str, Any]] = []
        source_metadata_batch: List[Dict[str, Any]] = []
        docs_consumed_from_queue = 0
        
        # Use a copy of the queue head to peek and potentially requeue if a doc's parts exceed batch
        # This is simpler than full rollback logic if we commit parts of a doc to the batch.
        # We consume from documents_queue directly once we decide to process a doc.
        
        processed_docs_this_batch_ids = set()

        while documents_queue and len(api_instances_batch) < self.api_batch_size:
            doc = documents_queue[0] # Peek at the first document

            # Skip document if it has no text content (after stripping) and no images,
            # or if it has already been processed in this batch (should not happen with peek and pop).
            if (not doc.content.strip() and not doc.images) or doc.id in processed_docs_this_batch_ids:
                logger.warning(f"Document {doc.id} has no content/images or already processed, skipping. This doc will be removed from queue.")
                documents_queue.pop(0) # Consume it from the main queue
                docs_consumed_from_queue += 1 # Count as consumed for progress
                continue

            # Store instances and metadata for the current document temporarily
            # before adding to the main batch, to handle cases where a doc might overflow the batch.
            current_doc_instances = []
            current_doc_source_metadata = []

            # 1. Process Text Content
            if doc.content.strip():
                text_for_text_only_instance = self._truncate_text(doc.content, self.max_tokens_per_text_instance)
                # Apply character limit as well
                if len(text_for_text_only_instance) > TEXT_CHAR_LIMIT_FOR_MULTIMODAL:
                    text_for_text_only_instance = text_for_text_only_instance[:TEXT_CHAR_LIMIT_FOR_MULTIMODAL]
                    logger.debug(f"Truncated text_only instance for doc {doc.id} to {TEXT_CHAR_LIMIT_FOR_MULTIMODAL} characters.")

                if text_for_text_only_instance: # Ensure text remains after truncation
                    current_doc_instances.append({"text": text_for_text_only_instance})
                    current_doc_source_metadata.append({
                        'unique_instance_id': f"{doc.id}::text",
                        'original_doc_id': doc.id,
                        'original_doc_title': doc.title,
                        'source_type': doc.source_type,
                        'source_url': doc.source_url,
                        'category': doc.category, # Propagate category from Document
                        'instance_type': 'text_only',
                        'image_index': None,
                        'original_image_index_in_document': None
                    })

            # 2. Process Images
            if doc.images:
                text_for_multimodal_instance = self._truncate_text(doc.content, self.max_tokens_for_multimodal_text) if doc.content.strip() else ""
                # Apply character limit as well
                if len(text_for_multimodal_instance) > TEXT_CHAR_LIMIT_FOR_MULTIMODAL:
                    text_for_multimodal_instance = text_for_multimodal_instance[:TEXT_CHAR_LIMIT_FOR_MULTIMODAL]
                    logger.debug(f"Truncated text_for_multimodal (with image) for doc {doc.id} to {TEXT_CHAR_LIMIT_FOR_MULTIMODAL} characters.")

                for original_image_idx, image_bytes in enumerate(doc.images):
                    if len(image_bytes) > self.max_image_bytes:
                        logger.warning(
                            f"Image {original_image_idx} for document {doc.id} "
                            f"exceeds size limit of {self.max_image_bytes} bytes ({len(image_bytes)} bytes) and will be skipped."
                        )
                        continue

                    instance: Dict[str, Any] = {}
                    instance_type_suffix = f"image_{original_image_idx}"
                    
                    if text_for_multimodal_instance:
                        instance["text"] = text_for_multimodal_instance
                        instance_type = f"text_image_{original_image_idx}"
                    else:
                        instance_type = f"image_only_{original_image_idx}"
                    
                    instance["image"] = {"bytesBase64Encoded": base64.b64encode(image_bytes).decode('utf-8')}
                    
                    current_doc_instances.append(instance)
                    current_doc_source_metadata.append({
                        'unique_instance_id': f"{doc.id}::{instance_type_suffix}",
                        'original_doc_id': doc.id,
                        'original_doc_title': doc.title,
                        'source_type': doc.source_type,
                        'source_url': doc.source_url,
                        'category': doc.category, # Propagate category from Document
                        'instance_type': instance_type,
                        'image_index': original_image_idx,
                        'original_image_index_in_document': original_image_idx
                    })
            
            # If the current document generated no instances (e.g. empty doc, or text only doc with empty text after strip, or only oversized images)
            if not current_doc_instances:
                logger.warning(f"Document {doc.id} yielded no processable instances and will be skipped.")
                documents_queue.pop(0) # Consume from main queue
                docs_consumed_from_queue += 1 # Count as consumed
                processed_docs_this_batch_ids.add(doc.id) # Mark as processed for this batch cycle
                continue

            # Check if adding these instances would overflow the API batch size
            if len(api_instances_batch) + len(current_doc_instances) <= self.api_batch_size:
                api_instances_batch.extend(current_doc_instances)
                source_metadata_batch.extend(current_doc_source_metadata)
                documents_queue.pop(0) # Consume the document from the main queue
                docs_consumed_from_queue += 1
                processed_docs_this_batch_ids.add(doc.id)
            else:
                # Not enough space in the current API batch for all parts of this document.
                # If the batch is currently empty, it means this single document (with its parts)
                # is too large for the api_batch_size. We must process what fits if any.
                if not api_instances_batch:
                    can_fit_count = self.api_batch_size
                    api_instances_batch.extend(current_doc_instances[:can_fit_count])
                    source_metadata_batch.extend(current_doc_source_metadata[:can_fit_count])
                    logger.warning(f"Document {doc.id} with {len(current_doc_instances)} parts exceeds api_batch_size of {self.api_batch_size}. "
                                   f"Processing only the first {can_fit_count} parts. The rest will be dropped for this document "
                                   f"as partial processing of a document across batches is not supported by current _prepare_batch logic.")
                    # This means parts of the doc are dropped if it alone is too big.
                    # A more complex implementation would re-queue the remainder of current_doc_instances.
                    # For now, we consume the doc and only take what fits if it's the first doc for the batch.
                    documents_queue.pop(0) # Consume the document
                    docs_consumed_from_queue += 1
                    processed_docs_this_batch_ids.add(doc.id)
                    # The while loop condition (len(api_instances_batch) < self.api_batch_size) will now be false.
                else:
                    # There are already other docs' instances in the batch.
                    # This current document cannot fit. Break the loop, it will be processed in the next batch.
                    break 
        
        return api_instances_batch, source_metadata_batch, docs_consumed_from_queue

    async def generate_embeddings(self, documents: List[Document]) -> List[Embedding]:
        """
        Generate multimodal embeddings for a list of documents.
        Each document can result in multiple embeddings (text part, image parts).
        """
        if not documents:
            return []

        all_embeddings: List[Embedding] = []
        # Create a queue from the documents list to allow efficient removal from the front
        documents_queue = list(documents)
        
        overall_progress_bar = None
        if self.verbose:
            overall_progress_bar = tqdm(total=len(documents_queue), desc="Processing Documents", unit="doc")

        processed_instances_count = 0

        while documents_queue:
            api_instances, source_metadata, docs_consumed = self._prepare_batch(documents_queue)
            
            if not api_instances:
                if docs_consumed > 0 and self.verbose and overall_progress_bar: # Consumed docs but no instances (e.g. all empty)
                    overall_progress_bar.update(docs_consumed)
                if not documents_queue: # No more documents and no instances from last try
                    break
                else: # No instances but still docs in queue, means current doc was empty/skipped.
                    continue # Next iteration will try to prepare batch from remaining docs.

            if self.verbose:
                logger.debug(f"Prepared batch with {len(api_instances)} instances from {docs_consumed} document(s). {len(documents_queue)} docs remaining in queue.")

            try:
                batch_embeddings = await self._retry_with_backoff(self._process_batch, api_instances, source_metadata)
                if batch_embeddings:
                    all_embeddings.extend(batch_embeddings)
                    processed_instances_count += len(batch_embeddings)
            except Exception as e: # Catch any exception after retries are exhausted
                logger.error(f"Failed to process a batch after multiple retries: {e}. Associated doc IDs: {[sm['original_doc_id'] for sm in source_metadata]}")
                # Continue processing other documents/batches if any, but this batch is lost.
            
            if self.verbose and overall_progress_bar:
                overall_progress_bar.update(docs_consumed)
        
        if self.verbose and overall_progress_bar:
            overall_progress_bar.close()
            logger.info(f"Finished processing. Generated {len(all_embeddings)} embeddings from {processed_instances_count} instances across {len(documents)} documents.")
            
        return all_embeddings 