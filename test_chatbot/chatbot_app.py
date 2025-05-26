import sys
import os
import traceback # Added for detailed error reporting
import json # Added for loading user_credentials.json

# Add the project root to sys.path to allow importing the 'agents' module
# This assumes 'test_chatbot_firebase' is one level below the project root where 'agents' directory resides.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv() # Call this early to load .env before other modules might need the variables

import gradio as gr
import asyncio # Added for async operations
from typing import Optional, Dict, List

from google.oauth2.credentials import Credentials # Added for type hinting

# --- Vertex AI / Generative Model Imports ---
import vertexai
from vertexai.generative_models import GenerativeModel, Part, HarmCategory, HarmBlockThreshold

# --- Early Logger Setup for this file ---
import logging
logger = logging.getLogger(__name__) # Define logger for chatbot_app.py

# Basic config if no other module has set it up globally
# This ensures logger is available for functions defined below like load_user_drive_credentials
if not logging.getLogger().hasHandlers(): # Check root logger
    # Or check specific logger: if not logger.hasHandlers() and not logger.propagate:
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Reduce verbosity of HTTP client libraries ---
logging.getLogger("httpcore").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING) # httpx can also be verbose with INFO

# --- Configuration ---
# TODO: Replace with your actual GCP project details and desired names
GCP_PROJECT_ID = "roadmapper-460703" # Corrected to match the actual project ID of the index/endpoint
GCP_LOCATION = "us-west1"
GCS_BUCKET_NAME = "roadmapper-embeddings" # Create this bucket in your GCP project
VECTOR_SEARCH_INDEX_NAME = "document-search-index" # This is the display name (can be the same for the new index type)
VECTOR_SEARCH_INDEX_ID = "5413634615355113472" # <<< NEW STREAMING INDEX ID
VECTOR_SEARCH_ENDPOINT_NAME = "doc-loader-test-endpoint" # We can reuse this or make a new one

# --- Document Loader Agent Imports ---
from agents.document_loader.src.data_source_loader import DataSourceLoader
from agents.document_loader.src.loaders.web_loader import WebLoader
from agents.document_loader.src.loaders.github_loader import GitHubLoader
from agents.document_loader.src.loaders.drive_loader import DriveLoader # Added
# TODO: Add other loaders like DriveLoader if needed
from agents.document_loader.src.processors.embedding_processor import EmbeddingProcessor as MultimodalEmbeddingProcessor # Renamed for clarity
from agents.document_loader.src.processors.text_embedding_processor import TextEmbeddingProcessor # Added
from agents.document_loader.src.models.base import VectorSearchConfig, Document, Embedding
from agents.document_loader.src.vector_search.index_manager import VectorSearchIndexManager

# IMPORTANT: GOOGLE_APPLICATION_CREDENTIALS should be set in the .env file
# and point to the correct service account key JSON file.
# The logic below relies on this environment variable.

# Ensure GOOGLE_APPLICATION_CREDENTIALS is set correctly via .env
cred_path_from_env = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# --- Initialize Vertex AI and Generative Model ---
llm_model: Optional[GenerativeModel] = None
try:
    if GCP_PROJECT_ID and GCP_PROJECT_ID != "your-gcp-project-id" and GCP_LOCATION:
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
        # TODO: Make model name configurable, e.g., "gemini-1.5-flash-001" or "gemini-1.0-pro-001"
        llm_model = GenerativeModel("gemini-2.0-flash") 
        logger.info(f"Vertex AI initialized and GenerativeModel '{llm_model._model_name}' loaded successfully.")
    else:
        logger.error("GCP_PROJECT_ID or GCP_LOCATION is not set or is still a placeholder. Vertex AI LLM cannot be initialized.")
except Exception as e:
    logger.error(f"Error initializing Vertex AI or loading GenerativeModel: {e}", exc_info=True)
    llm_model = None # Ensure it's None if initialization fails

# --- Global Variables for Document Loading Components ---
data_source_loader: Optional[DataSourceLoader] = None
# embedding_processor: Optional[EmbeddingProcessor] = None # Old single processor
multimodal_embedding_processor: Optional[MultimodalEmbeddingProcessor] = None # New
text_embedding_processor: Optional[TextEmbeddingProcessor] = None # New
vector_search_manager: Optional[VectorSearchIndexManager] = None
document_store: Dict[str, Document] = {} # Simple in-memory store for loaded documents by ID

# Define model IDs, could also come from config
TEXT_EMBEDDING_MODEL_ID = "text-embedding-004" # Or your preferred text model
MULTIMODAL_EMBEDDING_MODEL_ID = "multimodalembedding@001"

# --- User Credentials Helper ---
USER_CREDENTIALS_PATH = os.path.join(os.path.dirname(__file__), '..', 'private', 'user_credentials.json')

def load_user_drive_credentials() -> Optional[Credentials]:
    """Loads user's Google Drive OAuth2 credentials from the JSON file."""
    if os.path.exists(USER_CREDENTIALS_PATH):
        try:
            with open(USER_CREDENTIALS_PATH, 'r') as f:
                creds_data = json.load(f)
            # Reconstruct the Credentials object
            # Note: Scopes in creds_data are a list, credentials.from_authorized_user_info expects tuple or None
            scopes_list = creds_data.get('scopes')
            credentials = Credentials(
                token=creds_data.get('token'),
                refresh_token=creds_data.get('refresh_token'),
                token_uri=creds_data.get('token_uri'),
                client_id=creds_data.get('client_id'),
                client_secret=creds_data.get('client_secret'),
                scopes=scopes_list # Pass as list, constructor handles it
            )
            logger.info("Successfully loaded user credentials for Drive access.")
            # Check if token needs refresh (optional here, Drive client library usually handles it)
            # if credentials.expired and credentials.refresh_token:
            #     credentials.refresh(google.auth.transport.requests.Request())
            return credentials
        except Exception as e:
            logger.error(f"Error loading user credentials from {USER_CREDENTIALS_PATH}: {e}")
            return None
    else:
        logger.warning(f"User credentials file not found at {USER_CREDENTIALS_PATH}.")
        return None

async def initialize_document_pipeline():
    """Initializes all components required for document loading, embedding, and search."""
    global data_source_loader, multimodal_embedding_processor, text_embedding_processor, vector_search_manager
    
    logger.info("Attempting to initialize ALL pipeline components (including VectorSearchIndexManager)...")
    data_source_loader = None 
    multimodal_embedding_processor = None 
    text_embedding_processor = None
    vector_search_manager = None
    
    try:
        # 1. Initialize DataSourceLoader 
        data_source_loader = DataSourceLoader(sources_dir="data/sources") 
        logger.info("DataSourceLoader initialized successfully.")

        # 2. Initialize EmbeddingProcessors
        if not GCP_PROJECT_ID or GCP_PROJECT_ID == "your-gcp-project-id":
            logger.error("GCP_PROJECT_ID is not set. Critical for embedding processors.")
            return False
        
        logger.info("Initializing MultimodalEmbeddingProcessor...")
        multimodal_embedding_processor = MultimodalEmbeddingProcessor(
            project_id=GCP_PROJECT_ID, location=GCP_LOCATION, model_id=MULTIMODAL_EMBEDDING_MODEL_ID, verbose=True)
        logger.info("MultimodalEmbeddingProcessor initialized successfully.")

        logger.info("Initializing TextEmbeddingProcessor...")
        text_embedding_processor = TextEmbeddingProcessor(
            project_id=GCP_PROJECT_ID, location=GCP_LOCATION, model_id=TEXT_EMBEDDING_MODEL_ID, verbose=True)
        logger.info("TextEmbeddingProcessor initialized successfully.")
        
        # 3. Initialize VectorSearchIndexManager & Get Index
        if not GCS_BUCKET_NAME or GCS_BUCKET_NAME == "your-gcs-bucket-name-for-embeddings": 
            logger.error("GCS_BUCKET_NAME is not set. Critical for VectorSearchIndexManager.")
            return False
        if not VECTOR_SEARCH_INDEX_ID or VECTOR_SEARCH_INDEX_ID == "YOUR_ACTUAL_INDEX_ID_FROM_GCP": # Placeholder check
            logger.error("VECTOR_SEARCH_INDEX_ID is not set correctly.")
            return False

        logger.info("Initializing VectorSearchIndexManager...")
        vs_config = VectorSearchConfig(
            project_id=GCP_PROJECT_ID, location=GCP_LOCATION, index_name=VECTOR_SEARCH_INDEX_NAME)
        vector_search_manager = VectorSearchIndexManager(config=vs_config)
        logger.info("VectorSearchIndexManager initialized.")

        index_resource_name = f"projects/{GCP_PROJECT_ID}/locations/{GCP_LOCATION}/indexes/{VECTOR_SEARCH_INDEX_ID}"
        logger.info(f"Attempting to retrieve existing Vertex AI Index: {index_resource_name}")
        retrieved_index = vector_search_manager.get_index(index_resource_name=index_resource_name)
        if retrieved_index:
            logger.info(f"Successfully retrieved existing index: {retrieved_index.name}")
            # Now ensure it's deployed to our target endpoint
            logger.info(f"Ensuring index {retrieved_index.name} is deployed to endpoint '{VECTOR_SEARCH_ENDPOINT_NAME}'...")
            deployed_successfully = await asyncio.to_thread(
                vector_search_manager.ensure_index_deployed, 
                VECTOR_SEARCH_ENDPOINT_NAME
            )
            if deployed_successfully:
                logger.info(f"Index {retrieved_index.name} is successfully deployed to endpoint '{VECTOR_SEARCH_ENDPOINT_NAME}'.")
                # vector_search_manager.deployed_index_id should now be set by ensure_index_deployed
            else:
                logger.error(f"Failed to ensure index {retrieved_index.name} is deployed to endpoint '{VECTOR_SEARCH_ENDPOINT_NAME}'. Upserts and searches might fail.")
                # Optionally, return False here if deployment is critical for startup
                # return False 
        else:
            logger.warning(f"Failed to retrieve index using ID '{VECTOR_SEARCH_INDEX_ID}'. It might not exist or there was an error.")
            # Not returning False here, as the manager is init, but index isn't populated in self.index of manager
            # However, if get_index fails, ensure_index_deployed won't be called.

        logger.info("FULL document pipeline initialization attempt complete.")
        return True 
        
    except Exception as e:
        logger.error(f"Error during FULL document pipeline initialization: {e}", exc_info=True) # Log with traceback
        traceback.print_exc()
        return False

async def trigger_document_loading_and_indexing():
    global document_store, data_source_loader, text_embedding_processor, multimodal_embedding_processor, vector_search_manager
    
    status_message = "Starting document loading and indexing..."
    logger.info(status_message)

    # Initial check for critical components
    if not (data_source_loader and text_embedding_processor and vector_search_manager):
        error_msg = "ERROR: Critical components (DataSourceLoader, TextEmbeddingProcessor, or VectorSearchManager) not initialized."
        logger.error(error_msg)
        # Further check for vector_search_manager.index if manager itself is initialized
        if vector_search_manager and not vector_search_manager.index:
            index_error_msg = "ERROR: VectorSearchIndexManager is initialized, but the index itself was not retrieved/set."
            logger.error(index_error_msg)
            return f"{error_msg} {index_error_msg}"
        return error_msg

    all_loaded_documents: List[Document] = []

    try:
        # 1. Load user credentials for Drive (if available)
        user_drive_creds = load_user_drive_credentials()
        if user_drive_creds:
            logger.info("User Drive credentials loaded successfully.")
        else:
            logger.info("No user Drive credentials found or loaded. Drive processing will use service account if configured, or fail.")

        # 2. Load documents from different sources
        status_message = "Loading documents from configured sources..."
        logger.info(status_message)

        # A. Process Google Drive sources
        drive_folders_config = data_source_loader.load_drive_folders() # [(item_id, category, model_type), ...]
        if drive_folders_config:
            logger.info(f"Found {len(drive_folders_config)} Drive folder(s)/file(s) configurations to process.")
            for item_id, category, model_type in drive_folders_config:
                logger.info(f"  Processing Drive item: {item_id} (Category: {category}, Model Type: {model_type})")
                try:
                    # Instantiate DriveLoader with user_credentials if available
                    drive_loader = DriveLoader(
                        item_id=item_id,
                        category=category,
                        embedding_model_type=model_type,
                        user_credentials=user_drive_creds # Pass creds here
                    )
                    # Validate connection (optional, but good for early feedback)
                    if not await drive_loader.validate_connection():
                        logger.warning(f"  Failed to validate connection for Drive item {item_id}. Skipping.")
                        continue
                    
                    async for doc in drive_loader.load(): # Iterate async generator
                        if doc:
                            all_loaded_documents.append(doc)
                    logger.info(f"  Finished loading from Drive item {item_id}.")
                except Exception as e:
                    logger.error(f"Error processing Drive item {item_id}: {e}", exc_info=True)
        else:
            logger.info("No Google Drive sources configured in drive_folders.txt.")

        # B. Process GitHub repositories (Example, implement similarly if needed)
        github_repos_config = data_source_loader.load_github_repos()
        if github_repos_config:
            logger.info(f"Found {len(github_repos_config)} GitHub repo configurations to process.")
            # Get GITHUB_TOKEN from environment once, if available
            github_token = os.getenv('GITHUB_TOKEN')
            if not github_token:
                logger.warning("GITHUB_TOKEN environment variable not set. Private repos or high-rate API access might fail. GitHubLoader might have limited access.")

            for repo_config_name, category, model_type in github_repos_config: # Changed repo_path to repo_config_name
                logger.info(f"  Processing GitHub repo: {repo_config_name} (Category: {category}, Model Type: {model_type})")
                try:
                    github_loader = GitHubLoader(
                        repo_name=repo_config_name, # This is the 'owner/repo' string
                        token=github_token, # Pass the token
                        category=category,
                        embedding_model_type=model_type
                    )
                    
                    if not await github_loader.validate_connection():
                        logger.warning(f"Failed to validate connection for GitHub repo {repo_config_name}. Skipping.")
                        continue
                    
                    async for doc in github_loader.load(): # Iterate async generator
                        if doc:
                            all_loaded_documents.append(doc)
                    logger.info(f"  Finished loading from GitHub repo {repo_config_name}.")
                except Exception as e:
                    logger.error(f"Error processing GitHub repo {repo_config_name}: {e}", exc_info=True)
        else:
            logger.info("No GitHub sources configured in github_repos.txt.")

        # C. Process Web URLs (Example, implement similarly if needed)
        web_urls_config = data_source_loader.load_web_urls()
        if web_urls_config:
            logger.info(f"Found {len(web_urls_config)} Web URL configurations to process.")
            for single_url, category, model_type in web_urls_config: # Renamed url to single_url for clarity
                logger.info(f"  Processing Web URL: {single_url} (Category: {category}, Model Type: {model_type})")
                try:
                    web_loader = WebLoader(
                        urls=[single_url], # WebLoader expects a list of URLs
                        category=category,
                        embedding_model_type=model_type
                    )
                    # Validate connection (optional, but good for early feedback)
                    if not await web_loader.validate_connection():
                        logger.warning(f"Failed to validate connection for WebLoader with URL {single_url}. Skipping.")
                        continue
                    
                    async for doc in web_loader.load(): # Iterate async generator
                        if doc:
                            all_loaded_documents.append(doc)
                    logger.info(f"  Finished loading from Web URL {single_url}.")
                except Exception as e:
                    logger.error(f"Error processing Web URL {single_url}: {e}", exc_info=True)
        else:
            logger.info("No Web URL sources configured in web_urls.txt.")


        if not all_loaded_documents:
            message = "No documents were loaded from any source. Please check configurations and permissions."
            logger.warning(message)
            return message
        
        logger.info(f"Successfully loaded a total of {len(all_loaded_documents)} document(s)/item(s) from all sources.")
        
        document_store.clear()
        for doc in all_loaded_documents:
            if doc and doc.id: # Ensure doc and doc.id are not None
                 document_store[doc.id] = doc
            else:
                logger.warning(f"Encountered a document with no ID during storing: {doc.metadata if doc else 'None'}")

        status_message = f"Loaded {len(all_loaded_documents)} documents. Generating embeddings..."
        logger.info(status_message)

        # 3. Process and Embed Documents
        embeddings_for_indexing: List[Embedding] = []

        if all_loaded_documents:
            logger.info(f"Generating embeddings for {len(all_loaded_documents)} documents using {text_embedding_processor.__class__.__name__}...")
            try:
                # TextEmbeddingProcessor expects a List[Document] and handles batching internally.
                # It also handles selection of .content from each document.
                # TODO: Implement logic to choose between text_embedding_processor and multimodal_embedding_processor
                # based on document types or a general strategy. For now, defaulting to text_embedding_processor for all.
                
                # Filter documents for the appropriate processor or handle mixed types
                # For now, we assume all documents will be processed by text_embedding_processor
                text_documents_to_embed = []
                multimodal_documents_to_embed = [] # Placeholder for future

                for doc in all_loaded_documents:
                    if doc.embedding_model_type == "multimodal" and multimodal_embedding_processor:
                        # If a document is explicitly multimodal and we have the processor, route it there.
                        # This example assumes multimodal_embedding_processor also has a generate_embeddings method.
                        # For now, let's add to a separate list. The actual call to multimodal is TBD.
                        # multimodal_documents_to_embed.append(doc)
                        # logger.info(f"  Document {doc.id} queued for MultimodalEmbeddingProcessor (actual embedding TBD).")
                        # For the current flow, let's still attempt to get text embeddings for it too if content exists
                        # This behavior might need refinement based on desired multimodal strategy.
                        if doc.content: # If multimodal doc also has text content, let text processor handle it for now.
                            text_documents_to_embed.append(doc)
                            logger.info(f"  Document {doc.id} (multimodal) will also be processed for text embeddings.")
                        else:
                            logger.info(f"  Document {doc.id} is multimodal and has no direct text content for TextEmbeddingProcessor. Skipping for text.")
                    elif doc.content: # If it has text content, it can be processed by TextEmbeddingProcessor
                        text_documents_to_embed.append(doc)
                    else:
                        logger.info(f"  Skipping document {doc.id} for text embedding as it has no .content.")
                
                if text_documents_to_embed:
                    logger.info(f"Processing {len(text_documents_to_embed)} documents with TextEmbeddingProcessor...")
                    # The generate_embeddings method is async
                    text_embeddings = await text_embedding_processor.generate_embeddings(documents=text_documents_to_embed)
                    embeddings_for_indexing.extend(text_embeddings)
                    logger.info(f"Successfully generated {len(text_embeddings)} text embeddings.")
                else:
                    logger.info("No documents suitable for text embedding were found.")
                
                # Placeholder: if multimodal_documents_to_embed is populated, process them here
                # if multimodal_documents_to_embed and multimodal_embedding_processor:
                #     logger.info(f"Processing {len(multimodal_documents_to_embed)} documents with MultimodalEmbeddingProcessor...")
                #     multimodal_embeddings = await multimodal_embedding_processor.generate_embeddings(documents=multimodal_documents_to_embed)
                #     embeddings_for_indexing.extend(multimodal_embeddings)
                #     logger.info(f"Successfully generated {len(multimodal_embeddings)} multimodal embeddings.")

            except Exception as e:
                logger.error(f"Error during embedding generation: {e}", exc_info=True)
                # Decide if we should return or try to proceed if some embeddings were generated before error
        else:
            logger.info("No documents were loaded, skipping embedding generation.")

        if not embeddings_for_indexing:
            message = "No embeddings were generated from the loaded documents. Cannot proceed with indexing."
            logger.warning(message)
            return message

        status_message = f"Generated {len(embeddings_for_indexing)} embeddings. Uploading to GCS and indexing..."
        logger.info(status_message)

        # 4. Upload to GCS and Index (Old Batch Method) / Upsert to Index (New Streaming Method)
        # if not GCS_BUCKET_NAME or GCS_BUCKET_NAME == "roadmapper-document-loader": 
        #     pass # GCS_BUCKET_NAME is defined from config, this check is okay.

        if not vector_search_manager.index:
            error_msg = "ERROR: VectorSearchIndexManager does not have a valid index retrieved. Cannot add documents."
            logger.error(error_msg)
            # Attempt to re-retrieve and deploy (this logic might be redundant if init fails robustly)
            index_resource_name = f"projects/{GCP_PROJECT_ID}/locations/{GCP_LOCATION}/indexes/{VECTOR_SEARCH_INDEX_ID}"
            logger.info(f"Attempting to re-retrieve Vertex AI Index: {index_resource_name}")
            retrieved_index = vector_search_manager.get_index(index_resource_name=index_resource_name)
            if not retrieved_index:
                fatal_error_msg = "Fatal: Failed to re-retrieve index. Indexing cannot proceed."
                logger.critical(fatal_error_msg)
                return fatal_error_msg
            logger.info(f"Successfully re-retrieved index: {retrieved_index.name}. Ensuring deployment...")
            deployed_successfully = await asyncio.to_thread(vector_search_manager.ensure_index_deployed, VECTOR_SEARCH_ENDPOINT_NAME)
            if not deployed_successfully:
                fatal_error_msg = f"Fatal: Failed to ensure re-retrieved index {retrieved_index.name} is deployed. Indexing cannot proceed."
                logger.critical(fatal_error_msg)
                return fatal_error_msg
        
        # Check if index is deployed (via ensure_index_deployed in init or re-check here)
        # The new upsert_datapoints method in VectorSearchIndexManager doesn't strictly need self.deployed_index_id,
        # as it calls self.index.upsert_datapoints(). However, a deployed index is needed for searching.
        # The ensure_index_deployed method should set self.deployed_index_id if successful.
        if not vector_search_manager.deployed_index_id:
             logger.warning("Index does not seem to be deployed to an endpoint according to VectorSearchIndexManager state. Searching will fail. Upserting might proceed directly to index if configured for streaming.")
             # For streaming, self.index.upsert_datapoints() is the key.

        logger.info(f"Attempting to upsert {len(embeddings_for_indexing)} embeddings to index via streaming using VectorSearchIndexManager...")
        
        await asyncio.to_thread(
            vector_search_manager.upsert_datapoints,
            embeddings=embeddings_for_indexing
        )

        # Streaming upsert is typically fire-and-forget from client's perspective for LROs
        # The method itself logs success/failure of the call.
        final_message = f"Successfully initiated streaming upsert of {len(embeddings_for_indexing)} embeddings to index {vector_search_manager.index.name if vector_search_manager.index else 'N/A'}."
        logger.info(final_message)
        return final_message

    except Exception as e:
        error_msg = f"Error during document loading and indexing: {e}"
        logger.error(error_msg, exc_info=True)
        traceback.print_exc()
        return f"Error: {e}"

async def perform_test_search():
    """Performs a test search against the configured index."""
    global text_embedding_processor, vector_search_manager, document_store
    status_message = "Performing test search..."
    logger.info(status_message)

    if not text_embedding_processor or not vector_search_manager:
        error_msg = "ERROR: TextEmbeddingProcessor or VectorSearchManager not initialized."
        logger.error(error_msg)
        return error_msg
    
    if not vector_search_manager.index or not vector_search_manager.deployed_index_id:
        error_msg = "ERROR: VectorSearchManager does not have a deployed index. Cannot search."
        logger.error(error_msg)
        return error_msg

    test_query = "What is the Privacy Sandbox?"
    logger.info(f"Test query: '{test_query}'")

    try:
        logger.info("Generating embedding for the test query...")
        # Create a dummy Document object for the query to pass to the embedding processor
        query_doc = Document(
            id="test_query_doc", 
            content=test_query, 
            category="query", 
            embedding_model_type="text",
            title="Test Query Document",  # Added
            source_type="internal_query", # Added
            source_url="N/A"             # Added
        )
        query_embeddings = await text_embedding_processor.generate_embeddings(documents=[query_doc])
        
        if not query_embeddings or not query_embeddings[0].embedding:
            error_msg = "Failed to generate embedding for the test query."
            logger.error(error_msg)
            return error_msg
        
        query_vector = query_embeddings[0].embedding
        logger.info(f"Query embedding generated successfully (vector dim: {len(query_vector) if query_vector else 0}).")

        logger.info(f"Searching index {vector_search_manager.index.name} on endpoint {vector_search_manager.endpoint.name} (deployed_id: {vector_search_manager.deployed_index_id})...")
        
        # Perform search using asyncio.to_thread as search might be blocking
        search_results = await asyncio.to_thread(
            vector_search_manager.search,
            query_embedding=query_vector,
            num_neighbors=3 # Request top 3 neighbors
        )

        logger.info("Test search completed.")
        if search_results:
            logger.info(f"Found {len(search_results)} neighbors:")
            for i, result in enumerate(search_results):
                id_val = result.get('id')
                score_val = result.get('score')
                dist_val = result.get('distance')
                
                # Attempt to retrieve document details from document_store
                doc_info = " (Document details not found in store)"
                if id_val in document_store:
                    doc = document_store[id_val]
                    doc_info = f" (Title: {doc.title}, Source: {doc.source_type})"
                
                score_str = f"{score_val:.4f}" if score_val is not None else "N/A"
                dist_str = f"{dist_val:.4f}" if dist_val is not None else "N/A"
                
                logger.info(f"  {i+1}. ID: {id_val}, Score: {score_str} (Distance: {dist_str}){doc_info}")
            return f"Test search successful. Found {len(search_results)} results (see console for details)."
        else:
            logger.info("No results found for the test query.")
            return "Test search completed. No results found."

    except Exception as e:
        error_msg = f"Error during test search: {e}"
        logger.error(error_msg, exc_info=True)
        traceback.print_exc()
        return f"Error during test search: {e}"

# --- Chatbot Logic (to be expanded) ---
async def get_chatbot_response(user_input, history):
    global text_embedding_processor, vector_search_manager, document_store, llm_model
    
    history = history or [] # Ensure history is a list

    if not llm_model:
        bot_response = "ERROR: LLM model is not initialized. Please check the application logs."
        history.append((user_input, bot_response))
        return "", history

    if not text_embedding_processor or not vector_search_manager or not vector_search_manager.index or not vector_search_manager.deployed_index_id:
        # Fallback to LLM without context if vector search components are not ready
        warning_message = "Warning: Vector search components not fully initialized. Querying LLM without document context."
        logger.warning(warning_message)
        
        # Direct LLM call without context
        try:
            prompt = f"Question: {user_input}"
            # Configure safety settings to be less restrictive if needed, or handle blocked responses
            response = await llm_model.generate_content_async(
                prompt,
                generation_config={"max_output_tokens": 1024}, # Example config
                safety_settings={ # Adjust safety settings as needed
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            bot_response = response.text
        except Exception as e:
            error_msg = f"Error during direct LLM call: {e}"
            logger.error(error_msg, exc_info=True)
            bot_response = "Sorry, I encountered an error trying to process your request without document context."
        
        history.append((user_input, bot_response))
        return "", history

    # Proceed with context retrieval if components are ready
    context_string = "No relevant context found."
    try:
        logger.info(f"User input: '{user_input}'. Generating embedding for context retrieval...")
        query_doc = Document(
            id="chat_query", 
            content=user_input, 
            category="query", 
            embedding_model_type="text",
            title="Chat Query",
            source_type="chat_input",
            source_url="N/A"
        )
        query_embeddings = await text_embedding_processor.generate_embeddings(documents=[query_doc])
        
        if not query_embeddings or not query_embeddings[0].embedding:
            raise ValueError("Failed to generate embedding for the user query.")
        
        query_vector = query_embeddings[0].embedding
        logger.info(f"Query embedding generated (vector dim: {len(query_vector)}). Searching index...")

        search_results = await asyncio.to_thread(
            vector_search_manager.search,
            query_embedding=query_vector,
            num_neighbors=3 
        )
        logger.info(f"Search completed. Found {len(search_results) if search_results else 0} neighbors.")

        if search_results:
            context_docs = []
            for result in search_results:
                doc_id = result.get('id')
                if doc_id and doc_id in document_store:
                    # Ensure content is not None before appending
                    doc_content = document_store[doc_id].content
                    if doc_content:
                         context_docs.append(f"Document ID: {doc_id}\nContent:\n{doc_content}\n---")
                    else:
                        logger.info(f"Document {doc_id} found in store but has no content.")
                else:
                    logger.info(f"Document ID {doc_id} from search results not found in document_store or ID is None.")
            
            if context_docs:
                context_string = "\n\n".join(context_docs)
                logger.info(f"Context constructed from {len(context_docs)} documents.")
            else:
                logger.info("No valid content found in document_store for search results.")
        else:
            logger.info("No search results found for the query.")

    except Exception as e:
        error_msg = f"Error during context retrieval: {e}"
        logger.error(error_msg, exc_info=True)
        # Don't stop here, proceed to LLM with "No relevant context found."

    # Construct prompt for LLM
    prompt_template = f"""You are a helpful AI assistant for the Roadmapper application.
Your goal is to answer the user's question based on the provided context.
If the context is empty or not relevant to the question, try to answer the question based on your general knowledge,
but clearly state that the information is not from the provided documents.
If you use information from the context, please cite the Document ID(s) that provided the information.

Context:
{context_string}

User Question: {user_input}

Answer:
"""
    logger.info(f"Prompting LLM. Context length: {len(context_string)} chars. Question: {user_input}")

    try:
        # Configure safety settings to be less restrictive if needed
        response = await llm_model.generate_content_async(
            prompt_template,
            generation_config={"max_output_tokens": 1500, "temperature": 0.7}, # Example: increased tokens
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        # Check for blocked response
        if not response.candidates or not response.candidates[0].content.parts:
            bot_response = "I'm sorry, I couldn't generate a response for that query due to content restrictions or an unexpected issue."
            if response.prompt_feedback.block_reason:
                 bot_response += f" (Reason: {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason.name})"
            logger.warning(f"LLM response was blocked or empty. Prompt feedback: {response.prompt_feedback}")
        else:
            bot_response = response.text
        
    except Exception as e:
        error_msg = f"Error during LLM call with context: {e}"
        logger.error(error_msg, exc_info=True)
        bot_response = "Sorry, I encountered an error trying to process your request with document context."

    history.append((user_input, bot_response))
    return "", history

# --- Gradio UI ---
# Removed redundant chat_interface_with_startup() function.
# The UI is now solely defined and launched within run_app().

# --- Simple Test Function for Button Click Debugging (Can be commented out or removed now) ---
# async def simple_test_button_click(): 
#     print("DEBUG: $$$ ASYNC GLOBAL simple_test_button_click FUNCTION CALLED $$$ DEBUG")
#     return "ASYNC GLOBAL simple_test_button_click successful (check console)!"

async def run_app():
    success = await initialize_document_pipeline() # Calls STUBBED version
    if not success:
        logger.warning("Document pipeline failed to initialize.") # Simplified message

    with gr.Blocks(title="Roadmapper Test Chatbot") as demo:
        gr.Markdown("## Roadmapper Test Chatbot")
        gr.Markdown("Ask questions before and after embedding documents to test the document loader agent. Ensure GCP settings are correct.")
        
        with gr.Row():
            load_docs_button = gr.Button("Load and Index Documents")
            test_search_button = gr.Button("Perform Test Search") # Add new button
            status_display = gr.Textbox(label="Status", interactive=False, lines=2, max_lines=5)
        
        chatbot_component = gr.Chatbot(height=500, label="Chat")
        msg_input = gr.Textbox(label="Your message:", show_label=False, placeholder="Type your message here...")
        clear_button = gr.Button("Clear Chat")

        msg_input.submit(get_chatbot_response, [msg_input, chatbot_component], [msg_input, chatbot_component])
        clear_button.click(lambda: (None, None), None, [msg_input, chatbot_component], queue=False)

        # Wire up the button to the STUBBED ASYNC trigger_document_loading_and_indexing function
        load_docs_button.click(trigger_document_loading_and_indexing, [], [status_display])
        test_search_button.click(perform_test_search, [], [status_display]) # Wire the new button

    demo.launch()

if __name__ == "__main__":
    asyncio.run(run_app()) 