from typing import List, Optional, Dict, Any, Tuple
from google.cloud import aiplatform_v1
from google.api_core import exceptions
from ..models.base import Document, Embedding
import asyncio
from tqdm import tqdm # Consider making this optional or removing if not always used
import time
import logging
import random
import tiktoken

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# For text-embedding-004 (and similar gecko models), the token limit per request is 8192.
# However, for RETRIEVAL_DOCUMENT task type, the recommended input token length is often lower
# to optimize for retrieval quality vs. just fitting tokens.
# Let's aim for a practical chunk size that works well for retrieval.
# Gemma tokenizer used by text-embedding-004 might differ slightly from cl100k_base.
# For now, using cl100k_base for estimation and truncation is a reasonable start.
RECOMMENDED_MAX_TOKENS_FOR_RETRIEVAL = 2048 # A common guideline, can be adjusted
TEXT_EMBEDDING_MODEL_MAX_TOKENS = 8191 # Model's hard limit (use 8191 to be safe)


class TextEmbeddingProcessor:
    def __init__(self, project_id: str, 
                 location: str = "us-central1", 
                 model_id: str = "text-embedding-004", # Default to text-embedding-004
                 batch_size: int = 250, # Text embedding models often support larger batches
                 max_retries: int = 3, 
                 verbose: bool = False):
        self.project_id = project_id
        self.location = location
        self.model_id = model_id 
        # API batch size for text embedding models can be much larger (e.g., 250 for text-embedding-004)
        self.api_batch_size = min(batch_size, 250) 
        self.max_retries = max_retries
        self.client = None
        self.endpoint = None
        self.verbose = verbose
        # Tokenizer: cl100k_base is a common one, but text-embedding-004 uses Gemma's.
        # For simplicity, we'll use cl100k_base for truncation logic, acknowledging it's an approximation.
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Could not load cl100k_base tokenizer, falling back to a simple character count for truncation. Error: {e}")
            self.tokenizer = None

        self.initialize_vertex_ai()

    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text using the chosen tokenizer."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text) // 4 # Rough approximation if tokenizer fails

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to a specific maximum token limit."""
        if not self.tokenizer: # Fallback to character-based truncation if tokenizer is not available
            char_limit = max_tokens * 4 # Approximate
            if len(text) > char_limit:
                return text[:char_limit]
            return text

        tokens = self.tokenizer.encode(text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            # Attempt to decode, handling potential errors if truncation breaks multi-byte chars
            try:
                return self.tokenizer.decode(tokens)
            except:
                # Fallback: try decoding a slightly smaller subset of tokens
                try:
                    return self.tokenizer.decode(tokens[:-1])
                except:
                    # Last resort: use simple string slicing based on token proportion
                    # This is very approximate.
                    proportion = max_tokens / len(tokens) if len(tokens) > 0 else 1
                    return text[:int(len(text) * proportion)]
        return text

    def initialize_vertex_ai(self) -> bool:
        """Initialize Vertex AI PredictionServiceClient for Text embeddings."""
        try:
            client_options = {"api_endpoint": f"{self.location}-aiplatform.googleapis.com"}
            self.client = aiplatform_v1.PredictionServiceClient(client_options=client_options)
            # Construct endpoint using the provided model_id
            self.endpoint = f"projects/{self.project_id}/locations/{self.location}/publishers/google/models/{self.model_id}"
            
            # Test the model with a simple text prediction
            # For text-embedding-004, instance is {"content": "...", "task_type": "..."}
            test_instance = [{"content": "test document", "task_type": "RETRIEVAL_DOCUMENT"}]
            
            response = self.client.predict(endpoint=self.endpoint, instances=test_instance)
            
            # Updated check for successful response structure
            valid_response_received = False
            if response and response.predictions and len(response.predictions) > 0:
                first_prediction = response.predictions[0]
                # logger.info(f"[Debug Init Check] Type of first_prediction: {type(first_prediction)}")
                # logger.info(f"[Debug Init Check] Content of first_prediction: {first_prediction}")

                try:
                    # first_prediction is a MapComposite. We expect it to have an 'embeddings' key.
                    if 'embeddings' in first_prediction:
                        embedding_data = first_prediction['embeddings'] # Direct access
                        # logger.info(f"[Debug Init Check] Type of embedding_data (from first_prediction['embeddings']): {type(embedding_data)}")
                        # logger.info(f"[Debug Init Check] Content of embedding_data: {embedding_data}")

                        # embedding_data should also be MapComposite-like or dict-like, containing 'values'
                        if embedding_data and hasattr(embedding_data, 'get'): # Check if it supports .get or is dict-like
                            values_list_raw = embedding_data.get('values')
                            # logger.info(f"[Debug Init Check] Type of values_list_raw (from embedding_data.get('values')): {type(values_list_raw)}")
                            # logger.info(f"[Debug Init Check] Content of values_list_raw: {values_list_raw}")
                            
                            # Explicitly convert to list if it's a RepeatedComposite or similar list-like but not strictly list type
                            if hasattr(values_list_raw, '__iter__') and not isinstance(values_list_raw, list):
                                values_list = list(values_list_raw)
                                # logger.info(f"[Debug Init Check] Converted values_list_raw to list. New type: {type(values_list)}")
                            else:
                                values_list = values_list_raw

                            if values_list is not None and isinstance(values_list, list):
                                # logger.info(f"[Debug Init Check] values_list is a list. SUCCESS.")
                                valid_response_received = True
                            else:
                                logger.warning(f"Initialization check: values_list is None or not a list. Type: {type(values_list)}") # Kept warning, removed debug prefix
                        elif embedding_data and 'values' in embedding_data: # Fallback for direct key access if not .get()
                            values_list_raw = embedding_data['values']
                            # logger.info(f"[Debug Init Check] Type of values_list_raw (from embedding_data['values']): {type(values_list_raw)}")
                            # logger.info(f"[Debug Init Check] Content of values_list_raw: {values_list_raw}")

                            # Explicitly convert to list if it's a RepeatedComposite or similar list-like but not strictly list type
                            if hasattr(values_list_raw, '__iter__') and not isinstance(values_list_raw, list):
                                values_list = list(values_list_raw)
                                # logger.info(f"[Debug Init Check] Converted values_list_raw to list (direct access path). New type: {type(values_list)}")
                            else:
                                values_list = values_list_raw

                            if values_list is not None and isinstance(values_list, list):
                                # logger.info(f"[Debug Init Check] values_list is a list via direct key access. SUCCESS.")
                                valid_response_received = True
                            else:
                                logger.warning(f"Initialization check: values_list (via direct key) is None or not a list. Type: {type(values_list)}") # Kept warning, removed debug prefix
                        else:
                            logger.warning("Initialization check: embedding_data does not have 'values' key or is not suitable for .get().") # Kept warning, removed debug prefix
                    else:
                        logger.warning("Initialization check: 'embeddings' key not found in first_prediction.") # Kept warning, removed debug prefix
                        
                except AttributeError as ae:
                    logger.warning(f"AttributeError during initial response check for {self.model_id}: {ae}. This often happens with protobuf direct field access. Prediction: {first_prediction}")
                    valid_response_received = False
                except TypeError as te:
                    logger.warning(f"TypeError during initial response check for {self.model_id}: {te}. Prediction: {first_prediction}")
                    valid_response_received = False
                except Exception as e_detail: # Catch any other error during detailed check
                    logger.error(f"Unexpected error during detailed response check for {self.model_id}: {e_detail}")
                    valid_response_received = False
            
            if not valid_response_received:
                logger.error(f"Failed to get a valid structured embedding from Vertex AI text model {self.model_id} during initialization.")
                # The detailed logging of the 'response' object (the long JSON) will occur 
                # in the generic 'except Exception' block when this ValueError is caught.
                raise ValueError(f"Initialization check: Unexpected response structure from {self.model_id}")

            logger.info(f"Successfully initialized Vertex AI text embedding model: {self.model_id}")
            return True
        except exceptions.GoogleAPIError as e:
            logger.error(f"Google API Error initializing Vertex AI (text model: {self.model_id}): {e}")
            # Log the full error details if possible
            logger.error(f"Google API Error details: {e.errors}")
            logger.error(f"Underlying exception: {e.args}")
            raise ValueError(f"Failed to initialize Vertex AI text model {self.model_id} due to API error: {e}")
        except Exception as e: # Catch any other exception during the dummy call
            logger.error(f"Generic error during Vertex AI text model ({self.model_id}) initialization check.")
            # Log the type of exception and the exception itself
            logger.error(f"Exception type: {type(e)}")
            logger.error(f"Exception details: {e}")
            # If the error has a response attribute (like from httpx or requests), log it
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status code: {e.response.status_code if hasattr(e.response, 'status_code') else 'N/A'}")
                logger.error(f"Response content: {e.response.text if hasattr(e.response, 'text') else 'N/A'}")
            
            # Also, attempt to log the 'response' variable from the try block IF it exists,
            # in case the error happened after client.predict but before raising ValueError.
            # This 'response' variable is from the try block.
            try:
                if 'response' in locals() and response is not None:
                    logger.error(f"Content of 'response' variable from try block: {response}")
            except Exception as log_e:
                logger.error(f"Could not log 'response' variable: {log_e}")

            # Re-raise as ValueError to be caught by __init__
            raise ValueError(f"Failed to get embeddings from Vertex AI text model {self.model_id}")

    async def _retry_with_backoff(self, func, *args, **kwargs) -> Optional[List]:
        """Execute a function with exponential backoff retry logic."""
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except exceptions.ResourceExhausted as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Max retries reached for ResourceExhausted error: {e}")
                    raise
                wait_time = (2 ** attempt) + (random.random() * 0.1)
                logger.warning(f"Rate limit exceeded, retrying in {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)
            except Exception as e: # Catch other retryable errors if necessary
                if attempt == self.max_retries - 1:
                    logger.error(f"Max retries reached for error: {e}")
                    raise
                wait_time = (2 ** attempt) + (random.random() * 0.1) # Add some jitter
                logger.warning(f"An error occurred, retrying in {wait_time:.2f} seconds... Error: {e}")
                await asyncio.sleep(wait_time)
        return None # Should not be reached if max_retries leads to an exception

    async def _process_batch(self, prepared_instances: List[Dict[str, Any]], source_metadata_for_batch: List[Dict[str, Any]]) -> List[Embedding]:
        """Process a single batch of prepared instances using the text embedding model (async)."""
        if not prepared_instances:
            return []

        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None, 
                lambda: self.client.predict(endpoint=self.endpoint, instances=prepared_instances)
            )
        except Exception as e:
            logger.error(f"API call to Vertex AI text predict failed: {e}")
            logger.error(f"Endpoint: {self.endpoint}")
            logger.error(f"Instances sent: {prepared_instances[:2]}... (first 2 if many)") # Log a sample of instances
            raise 

        raw_embeddings = response.predictions
        
        final_embeddings = []
        if len(raw_embeddings) != len(prepared_instances):
            logger.error(f"Mismatch between number of instances sent ({len(prepared_instances)}) and embeddings received ({len(raw_embeddings)}). Skipping batch.")
            return []

        for i, raw_embedding_result_container in enumerate(raw_embeddings):
            # For text-embedding-004, the structure is typically:
            # raw_embedding_result_container (a MapComposite) = {'embeddings': embeddings_map}
            # embeddings_map (also a MapComposite) = {'values': [0.1, 0.2, ...], 'statistics': ...}
            vector = None
            try:
                # Check if raw_embedding_result_container supports .get (like MapComposite or dict)
                # and if it has an 'embeddings' key.
                embeddings_field = None
                if hasattr(raw_embedding_result_container, 'get'):
                    embeddings_field = raw_embedding_result_container.get('embeddings')
                elif isinstance(raw_embedding_result_container, dict): # Fallback for pure dict
                    embeddings_field = raw_embedding_result_container.get('embeddings')

                if embeddings_field is not None:
                    # Now check if embeddings_field (which should be the embeddings_map)
                    # supports .get and has a 'values' key.
                    values_list_raw = None
                    if hasattr(embeddings_field, 'get'):
                        values_list_raw = embeddings_field.get('values')
                    elif isinstance(embeddings_field, dict): # Fallback for pure dict
                        values_list_raw = embeddings_field.get('values')
                    
                    if values_list_raw is not None:
                        # Ensure it's a Python list and contains numbers
                        if hasattr(values_list_raw, '__iter__') and not isinstance(values_list_raw, list):
                            potential_vector = list(values_list_raw)
                        elif isinstance(values_list_raw, list):
                            potential_vector = values_list_raw
                        else:
                            potential_vector = None # Not iterable or list

                        if potential_vector and all(isinstance(v, (float, int)) for v in potential_vector):
                            vector = potential_vector
                        else:
                            logger.warning(f"Instance {i}: 'values' field did not contain a valid list of numbers. Values: {values_list_raw}")
                    else:
                        logger.warning(f"Instance {i}: 'values' key not found in embeddings field or is None. Embeddings field: {embeddings_field}")
                else:
                    logger.warning(f"Instance {i}: 'embeddings' key not found in prediction result or is None. Result: {raw_embedding_result_container}")

            except Exception as e_parse:
                logger.error(f"Error parsing embedding structure for instance {i}: {e_parse}. Container: {raw_embedding_result_container}")
                vector = None # Ensure vector is None if parsing fails
            
            if not vector: # Simplified check, as previous logic now ensures it's a list of numbers or None
                logger.error(f"Skipping instance {i}: No valid embedding vector extracted. Original container: {raw_embedding_result_container}")
                continue
            
            source_info = source_metadata_for_batch[i]
            doc_embedding = Embedding(
                document_id=source_info['unique_instance_id'], 
                embedding=vector,
                model_name=self.model_id, # Store which model was used
                category=source_info.get('category'),
                metadata={
                    'original_doc_id': source_info['original_doc_id'], 
                    'source_type': source_info['source_type'],
                    'source_url': source_info['source_url'],
                    'original_title': source_info['original_doc_title'],
                    'category': source_info.get('category'),
                    'instance_type': source_info['instance_type'], # Should be 'text_only'
                    'embedding_model_used': self.model_id # Explicitly add model used
                }
            )
            final_embeddings.append(doc_embedding)
            
        if self.verbose and final_embeddings:
            logger.debug(f"Text Embedding Processed batch. Generated {len(final_embeddings)} embeddings. Sample vector start: {final_embeddings[0].embedding[:5]}")
        elif not final_embeddings and prepared_instances:
            logger.warning(f"Text Embedding Processed batch but generated 0 embeddings from {len(prepared_instances)} instances.")
            
        return final_embeddings

    def _prepare_batch(self, documents_queue: List[Document]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
        """
        Prepare a batch of API instances from a queue of documents for text embedding.
        Returns a tuple: (api_instances_batch, source_metadata_batch, documents_consumed_count)
        Each document yields one API instance from its text content.
        Stops when api_batch_size is reached or documents_queue is empty.
        """
        api_instances_batch: List[Dict[str, Any]] = []
        source_metadata_batch: List[Dict[str, Any]] = []
        docs_consumed_from_queue = 0
        
        processed_docs_this_batch_ids = set()

        while documents_queue and len(api_instances_batch) < self.api_batch_size:
            doc = documents_queue[0] # Peek at the first document

            if not doc.content.strip() or doc.id in processed_docs_this_batch_ids:
                logger.debug(f"Document {doc.id} has no text content or already processed in this batch, skipping. This doc will be removed from queue.")
                documents_queue.pop(0) 
                docs_consumed_from_queue += 1
                continue

            # Text models like text-embedding-004 expect input like:
            # {"content": "text to embed", "task_type": "RETRIEVAL_DOCUMENT"}
            # We should truncate based on RECOMMENDED_MAX_TOKENS_FOR_RETRIEVAL for quality,
            # but ensure it does not exceed TEXT_EMBEDDING_MODEL_MAX_TOKENS.
            
            # Prioritize the recommended retrieval token limit for quality.
            # The model itself might handle longer sequences up to its hard limit, but often quality degrades.
            truncated_text = self._truncate_text(doc.content, RECOMMENDED_MAX_TOKENS_FOR_RETRIEVAL)
            
            # As a safeguard, ensure it also respects the model's absolute max token limit
            # This second truncation should rarely trigger if RECOMMENDED_MAX_TOKENS_FOR_RETRIEVAL is set reasonably.
            truncated_text = self._truncate_text(truncated_text, TEXT_EMBEDDING_MODEL_MAX_TOKENS)


            if not truncated_text.strip(): # If truncation results in empty text
                logger.warning(f"Document {doc.id} resulted in empty content after truncation, skipping.")
                documents_queue.pop(0)
                docs_consumed_from_queue +=1
                continue
                
            api_instance = {
                "content": truncated_text,
                "task_type": "RETRIEVAL_DOCUMENT" # Common for document embedding
            }
            
            # Check if adding this instance would overflow the batch
            if len(api_instances_batch) < self.api_batch_size:
                api_instances_batch.append(api_instance)
                source_metadata_batch.append({
                    'unique_instance_id': f"{doc.id}::text_content", # Make it unique for text
                    'original_doc_id': doc.id,
                    'original_doc_title': doc.title,
                    'source_type': doc.source_type,
                    'source_url': doc.source_url,
                    'category': doc.category,
                    'instance_type': 'text_only', # Explicitly 'text_only'
                    'embedding_model_used': self.model_id
                })
                processed_docs_this_batch_ids.add(doc.id)
                documents_queue.pop(0) # Consume the document from the main queue
                docs_consumed_from_queue += 1
            else:
                # Batch is full, stop preparing for this round
                break 
        
        if self.verbose:
            logger.debug(f"Prepared text batch of {len(api_instances_batch)} instances, consuming {docs_consumed_from_queue} documents.")
        return api_instances_batch, source_metadata_batch, docs_consumed_from_queue

    async def generate_embeddings(self, documents: List[Document]) -> List[Embedding]:
        """
        Generate text embeddings for a list of documents.
        Documents are processed in batches.
        """
        all_generated_embeddings: List[Embedding] = []
        
        # Create a mutable copy of the documents list to use as a queue
        documents_queue = list(documents)
        
        # Use tqdm for progress bar if documents_queue is not empty
        # Total iterations will be roughly len(documents) / (docs_consumed_per_batch), 
        # but simpler to just use total number of documents for tqdm's total.
        progress_bar_description = f"Generating Text Embeddings ({self.model_id})"
        with tqdm(total=len(documents_queue), desc=progress_bar_description, disable=not self.verbose) as pbar:
            while documents_queue:
                # Prepare a batch of API instances and their corresponding original document info
                api_instances, source_metadata, docs_consumed = self._prepare_batch(documents_queue) # documents_queue is modified in place
                
                if not api_instances:
                    if docs_consumed == 0 and documents_queue: 
                        # This case implies _prepare_batch couldn't process any of the remaining docs
                        # (e.g., all empty after truncation or some other issue).
                        # Log and break to prevent infinite loop.
                        logger.warning(f"No API instances could be prepared from remaining {len(documents_queue)} documents. Stopping generation.")
                        # Consume remaining queue to update progress bar correctly and exit.
                        pbar.update(len(documents_queue))
                        documents_queue.clear()
                        break 
                    elif not documents_queue: # All docs processed or queue became empty
                        break
                    # If docs_consumed > 0 but no api_instances, it means some docs were skipped (e.g. empty content)
                    # The loop will continue, and pbar will be updated below.

                if api_instances:
                    # Process the batch with retries
                    batch_embeddings = await self._retry_with_backoff(self._process_batch, api_instances, source_metadata)
                    
                    if batch_embeddings:
                        all_generated_embeddings.extend(batch_embeddings)
                        if self.verbose:
                            logger.debug(f"Successfully generated {len(batch_embeddings)} text embeddings in this batch.")
                    else:
                        logger.warning(f"Text embedding batch for {len(api_instances)} instances returned no embeddings after retries.")
                
                pbar.update(docs_consumed if docs_consumed > 0 else (len(api_instances) if not docs_consumed else 0) )

        if self.verbose:
            logger.info(f"Total text embeddings generated ({self.model_id}): {len(all_generated_embeddings)} from {len(documents)} documents.")
        return all_generated_embeddings 