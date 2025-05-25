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

import firebase_admin
from firebase_admin import credentials as firebase_creds # Aliased to avoid confusion
from firebase_admin import firestore
import gradio as gr
import asyncio # Added for async operations
from typing import Optional, Dict, List

from google.oauth2.credentials import Credentials # Added for type hinting

# --- Early Logger Setup for this file ---
import logging
logger = logging.getLogger(__name__) # Define logger for chatbot_app.py
# Basic config if no other module has set it up globally
# This ensures logger is available for functions defined below like load_user_drive_credentials
if not logging.getLogger().hasHandlers(): # Check root logger
    # Or check specific logger: if not logger.hasHandlers() and not logger.propagate:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Configuration ---
# TODO: Replace with your actual GCP project details and desired names
GCP_PROJECT_ID = "roadmapper-460703" 
GCP_LOCATION = "us-west1"
GCS_BUCKET_NAME = "roadmapper-document-loader" # Create this bucket in your GCP project
VECTOR_SEARCH_INDEX_NAME = "document-search-index" # This is the display name
VECTOR_SEARCH_INDEX_ID = "5530728205666746368" # <<< IMPORTANT: SET THIS VALUE
VECTOR_SEARCH_ENDPOINT_NAME = "doc-loader-test-endpoint"

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

# --- Firebase Initialization ---
# IMPORTANT: Replace 'path/to/your/serviceAccountKey.json' with the actual path to your key file.
# Ensure this key file is in your .gitignore to prevent committing it.
SERVICE_ACCOUNT_KEY_PATH = "../private/roadmapper-e7f85-firebase-adminsdk-fbsvc-fca400b730.json"

# Ensure GOOGLE_APPLICATION_CREDENTIALS is set if not using the same key as Firebase
# or if the Firebase key doesn't have Vertex AI/Storage permissions.
# It's often simpler to set the environment variable.
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_KEY_PATH 

try:
    cred = firebase_creds.Certificate(SERVICE_ACCOUNT_KEY_PATH)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase initialized successfully.")
except Exception as e:
    print(f"Error initializing Firebase: {e}")
    db = None

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
    
    print("Attempting to initialize ALL pipeline components (including VectorSearchIndexManager)...")
    data_source_loader = None 
    multimodal_embedding_processor = None 
    text_embedding_processor = None
    vector_search_manager = None
    
    try:
        # 1. Initialize DataSourceLoader 
        data_source_loader = DataSourceLoader(sources_dir="data/sources") 
        print("DataSourceLoader initialized successfully.")

        # 2. Initialize EmbeddingProcessors
        if not GCP_PROJECT_ID or GCP_PROJECT_ID == "your-gcp-project-id":
            logger.error("GCP_PROJECT_ID is not set. Critical for embedding processors.")
            return False
        
        print("Initializing MultimodalEmbeddingProcessor...")
        multimodal_embedding_processor = MultimodalEmbeddingProcessor(
            project_id=GCP_PROJECT_ID, location=GCP_LOCATION, model_id=MULTIMODAL_EMBEDDING_MODEL_ID, verbose=True)
        print("MultimodalEmbeddingProcessor initialized successfully.")

        print("Initializing TextEmbeddingProcessor...")
        text_embedding_processor = TextEmbeddingProcessor(
            project_id=GCP_PROJECT_ID, location=GCP_LOCATION, model_id=TEXT_EMBEDDING_MODEL_ID, verbose=True)
        print("TextEmbeddingProcessor initialized successfully.")
        
        # 3. Initialize VectorSearchIndexManager & Get Index
        if not GCS_BUCKET_NAME or GCS_BUCKET_NAME == "your-gcs-bucket-name-for-embeddings": 
            logger.error("GCS_BUCKET_NAME is not set. Critical for VectorSearchIndexManager.")
            return False
        if not VECTOR_SEARCH_INDEX_ID or VECTOR_SEARCH_INDEX_ID == "YOUR_ACTUAL_INDEX_ID_FROM_GCP": # Placeholder check
            logger.error("VECTOR_SEARCH_INDEX_ID is not set correctly.")
            return False

        print("Initializing VectorSearchIndexManager...")
        vs_config = VectorSearchConfig(
            project_id=GCP_PROJECT_ID, location=GCP_LOCATION, index_name=VECTOR_SEARCH_INDEX_NAME)
        vector_search_manager = VectorSearchIndexManager(config=vs_config)
        print("VectorSearchIndexManager initialized.")

        index_resource_name = f"projects/{GCP_PROJECT_ID}/locations/{GCP_LOCATION}/indexes/{VECTOR_SEARCH_INDEX_ID}"
        print(f"Attempting to retrieve existing Vertex AI Index: {index_resource_name}")
        retrieved_index = vector_search_manager.get_index(index_resource_name=index_resource_name)
        if retrieved_index:
            print(f"Successfully retrieved existing index: {retrieved_index.name}")
        else:
            logger.warning(f"Failed to retrieve index using ID '{VECTOR_SEARCH_INDEX_ID}'. It might not exist or there was an error.")
            # Not returning False here, as the manager is init, but index isn't populated in self.index of manager

        print("FULL document pipeline initialization attempt complete.")
        return True 
        
    except Exception as e:
        print(f"Error during FULL document pipeline initialization: {e}")
        logger.error(f"Error during FULL document pipeline initialization: {e}", exc_info=True) # Log with traceback
        traceback.print_exc()
        return False

async def trigger_document_loading_and_indexing():
    global document_store, data_source_loader, text_embedding_processor, multimodal_embedding_processor, vector_search_manager
    
    status_message = "Starting document loading and indexing..."
    print(status_message)

    # Initial check for critical components
    if not (data_source_loader and text_embedding_processor and vector_search_manager):
        error_msg = "ERROR: Critical components (DataSourceLoader, TextEmbeddingProcessor, or VectorSearchManager) not initialized."
        print(error_msg)
        # Further check for vector_search_manager.index if manager itself is initialized
        if vector_search_manager and not vector_search_manager.index:
            index_error_msg = "ERROR: VectorSearchIndexManager is initialized, but the index itself was not retrieved/set."
            print(index_error_msg)
            return f"{error_msg} {index_error_msg}"
        return error_msg

    all_loaded_documents: List[Document] = []

    try:
        # 1. Load user credentials for Drive (if available)
        user_drive_creds = load_user_drive_credentials()
        if user_drive_creds:
            print("User Drive credentials loaded successfully.")
        else:
            print("No user Drive credentials found or loaded. Drive processing will use service account if configured, or fail.")

        # 2. Load documents from different sources
        status_message = "Loading documents from configured sources..."
        print(status_message)

        # A. Process Google Drive sources
        drive_folders_config = data_source_loader.load_drive_folders() # [(item_id, category, model_type), ...]
        if drive_folders_config:
            print(f"Found {len(drive_folders_config)} Drive folder(s)/file(s) configurations to process.")
            for item_id, category, model_type in drive_folders_config:
                print(f"  Processing Drive item: {item_id} (Category: {category}, Model Type: {model_type})")
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
                        print(f"  Failed to validate connection for Drive item {item_id}. Skipping.")
                        continue
                    
                    async for doc in drive_loader.load(): # Iterate async generator
                        if doc:
                            all_loaded_documents.append(doc)
                    print(f"  Finished loading from Drive item {item_id}.")
                except Exception as e:
                    print(f"  Error processing Drive item {item_id}: {e}")
                    logger.error(f"Error processing Drive item {item_id}: {e}", exc_info=True)
        else:
            print("No Google Drive sources configured in drive_folders.txt.")

        # B. Process GitHub repositories (Example, implement similarly if needed)
        github_repos_config = data_source_loader.load_github_repos()
        if github_repos_config:
            print(f"Found {len(github_repos_config)} GitHub repo configurations to process.")
            for repo_config_path, category, model_type in github_repos_config: # Renamed repo_path to repo_config_path for clarity
                print(f"  Processing GitHub repo: {repo_config_path} (Category: {category}, Model Type: {model_type})")
                try:
                    github_loader = GitHubLoader(
                        repo_name=repo_config_path, # Corrected: repo_path to repo_name
                        category=category,
                        embedding_model_type=model_type
                        # Add credentials if GitHubLoader supports them, e.g., token
                    )
                    # Add connection validation if GitHubLoader has it
                    # async for doc in github_loader.load():
                    # all_loaded_documents.append(doc)
                    # print(f"  Finished loading from GitHub repo {repo_config_path}.")
                    print(f"  GitHub loader for {repo_config_path} not fully implemented in this flow yet. Skipping actual loading.")
                except Exception as e:
                    print(f"  Error processing GitHub repo {repo_config_path}: {e}")
                    logger.error(f"Error processing GitHub repo {repo_config_path}: {e}", exc_info=True)
        else:
            print("No GitHub sources configured in github_repos.txt.")

        # C. Process Web URLs (Example, implement similarly if needed)
        web_urls_config = data_source_loader.load_web_urls()
        if web_urls_config:
            print(f"Found {len(web_urls_config)} Web URL configurations to process.")
            for single_url, category, model_type in web_urls_config: # Renamed url to single_url for clarity
                print(f"  Processing Web URL: {single_url} (Category: {category}, Model Type: {model_type})")
                try:
                    web_loader = WebLoader(
                        urls=[single_url], # Corrected: url to urls=[single_url]
                        category=category,
                        embedding_model_type=model_type
                    )
                    # Add connection validation if WebLoader has it
                    # async for doc in web_loader.load():
                    # all_loaded_documents.append(doc)
                    # print(f"  Finished loading from Web URL {single_url}.")
                    print(f"  Web loader for {single_url} not fully implemented in this flow yet. Skipping actual loading.")
                except Exception as e:
                    print(f"  Error processing Web URL {single_url}: {e}")
                    logger.error(f"Error processing Web URL {single_url}: {e}", exc_info=True)
        else:
            print("No Web URL sources configured in web_urls.txt.")


        if not all_loaded_documents:
            message = "No documents were loaded from any source. Please check configurations and permissions."
            print(message)
            return message
        
        print(f"Successfully loaded a total of {len(all_loaded_documents)} document(s)/item(s) from all sources.")
        
        document_store.clear()
        for doc in all_loaded_documents:
            if doc and doc.id: # Ensure doc and doc.id are not None
                 document_store[doc.id] = doc
            else:
                logger.warning(f"Encountered a document with no ID during storing: {doc.metadata if doc else 'None'}")

        status_message = f"Loaded {len(all_loaded_documents)} documents. Generating embeddings..."
        print(status_message)

        # 3. Process and Embed Documents
        embeddings_for_indexing: List[Embedding] = []

        if all_loaded_documents:
            print(f"Generating embeddings for {len(all_loaded_documents)} documents using {text_embedding_processor.__class__.__name__}...")
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
                        # print(f"  Document {doc.id} queued for MultimodalEmbeddingProcessor (actual embedding TBD).")
                        # For the current flow, let's still attempt to get text embeddings for it too if content exists
                        # This behavior might need refinement based on desired multimodal strategy.
                        if doc.content: # If multimodal doc also has text content, let text processor handle it for now.
                            text_documents_to_embed.append(doc)
                            print(f"  Document {doc.id} (multimodal) will also be processed for text embeddings.")
                        else:
                            print(f"  Document {doc.id} is multimodal and has no direct text content for TextEmbeddingProcessor. Skipping for text.")
                    elif doc.content: # If it has text content, it can be processed by TextEmbeddingProcessor
                        text_documents_to_embed.append(doc)
                    else:
                        print(f"  Skipping document {doc.id} for text embedding as it has no .content.")
                
                if text_documents_to_embed:
                    print(f"Processing {len(text_documents_to_embed)} documents with TextEmbeddingProcessor...")
                    # The generate_embeddings method is async
                    text_embeddings = await text_embedding_processor.generate_embeddings(documents=text_documents_to_embed)
                    embeddings_for_indexing.extend(text_embeddings)
                    print(f"Successfully generated {len(text_embeddings)} text embeddings.")
                else:
                    print("No documents suitable for text embedding were found.")
                
                # Placeholder: if multimodal_documents_to_embed is populated, process them here
                # if multimodal_documents_to_embed and multimodal_embedding_processor:
                #     print(f"Processing {len(multimodal_documents_to_embed)} documents with MultimodalEmbeddingProcessor...")
                #     multimodal_embeddings = await multimodal_embedding_processor.generate_embeddings(documents=multimodal_documents_to_embed)
                #     embeddings_for_indexing.extend(multimodal_embeddings)
                #     print(f"Successfully generated {len(multimodal_embeddings)} multimodal embeddings.")

            except Exception as e:
                print(f"Error during embedding generation: {e}")
                logger.error(f"Error during embedding generation: {e}", exc_info=True)
                # Decide if we should return or try to proceed if some embeddings were generated before error
        else:
            print("No documents were loaded, skipping embedding generation.")

        if not embeddings_for_indexing:
            message = "No embeddings were generated from the loaded documents. Cannot proceed with indexing."
            print(message)
            return message

        status_message = f"Generated {len(embeddings_for_indexing)} embeddings. Uploading to GCS and indexing..."
        print(status_message)

        # 4. Upload to GCS and Index
        if not GCS_BUCKET_NAME or GCS_BUCKET_NAME == "roadmapper-document-loader": 
            pass # GCS_BUCKET_NAME is defined from config, this check is okay.

        # This is the line you'll manually fix the indentation for:
        if not vector_search_manager.index:
            error_msg = "ERROR: VectorSearchIndexManager does not have a valid index retrieved. Cannot add documents."
            print(error_msg)
            logger.error(error_msg)
            index_resource_name = f"projects/{GCP_PROJECT_ID}/locations/{GCP_LOCATION}/indexes/{VECTOR_SEARCH_INDEX_ID}"
            print(f"Attempting to re-retrieve Vertex AI Index: {index_resource_name}")
            retrieved_index = vector_search_manager.get_index(index_resource_name=index_resource_name)
            if not retrieved_index:
                fatal_error_msg = "Fatal: Failed to re-retrieve index. Indexing cannot proceed."
                print(fatal_error_msg)
                return fatal_error_msg
            print(f"Successfully re-retrieved index: {retrieved_index.name}")
        
        # The upload_embeddings method in VectorSearchIndexManager will create its own GCS file structure.
        # We just need to pass the embeddings and the bucket name.
        print(f"Attempting to upload {len(embeddings_for_indexing)} embeddings to GCS and update index via VectorSearchIndexManager...")
        
        # Call the synchronous method upload_embeddings
        # It handles creating the JSONL file in GCS and updating the index.
        gcs_file_uri_or_response = vector_search_manager.upload_embeddings(
            embeddings=embeddings_for_indexing,
            bucket_name=GCS_BUCKET_NAME
            # The file_prefix is optional in upload_embeddings and not strictly needed here yet.
        )

        # upload_embeddings returns the GCS URI of the uploaded file.
        if gcs_file_uri_or_response: 
            final_message = f"Successfully loaded {len(all_loaded_documents)} docs, generated {len(embeddings_for_indexing)} embeddings. Index update initiated with GCS data. GCS file URI: {gcs_file_uri_or_response}"
            print(final_message)
            logger.info(final_message)
            return final_message
        else:
            # This case might not be hit if upload_embeddings raises an exception on failure.
            # Depending on its error handling, it might return None or an empty string on some failures.
            error_message = f"Failed to initiate index update. {len(embeddings_for_indexing)} embeddings were generated. Check logs for details from VectorSearchIndexManager."
            print(error_message)
            logger.error(error_message)
            return error_message

    except Exception as e:
        error_msg = f"Error during document loading and indexing: {e}"
        print(error_msg)
        logger.error(error_msg, exc_info=True)
        traceback.print_exc()
        return f"Error: {e}"

# --- Chatbot Logic (to be expanded) ---
def get_chatbot_response(user_input, history):
    # Placeholder for chatbot response logic
    # In the future, this will:
    # 1. Query the vector DB for context (if documents are embedded).
    # 2. Call a language model with the user_input and context.
    # 3. Optionally, save chat history to Firebase.
    
    bot_response = f"Echo: {user_input}" # Simple echo for now
    
    # Example: Save to Firestore (optional)
    if db:
        try:
            chat_ref = db.collection('chats').document()
            chat_ref.set({
                'user_input': user_input,
                'bot_response': bot_response,
                'timestamp': firestore.SERVER_TIMESTAMP
            })
        except Exception as e:
            print(f"Error saving chat to Firebase: {e}")
            
    return bot_response

# --- Gradio UI ---
async def chat_interface_with_startup():
    # Initialize the document pipeline on startup
    # In a real app, you might do this in a separate startup script or event
    # For Gradio, we can try to run it before launching the interface
    success = await initialize_document_pipeline()
    if not success:
        print("WARNING: Document pipeline failed to initialize. Document loading features will not work.")
        # Optionally, disable parts of the UI or show a persistent error message

    iface = gr.ChatInterface(
        fn=get_chatbot_response,
        title="Roadmapped Test Chatbot",
        description="Ask questions before and after embedding documents to test the document loader agent. Ensure GCP settings are correct.",
        chatbot=gr.Chatbot(height=600),
        # TODO: Add UI elements for triggering document loading
    )
    
    # Add a button to trigger document loading
    with iface.blocks as demo:
        # Re-create the chat interface parts if you need to mix with other gr.Blocks
        # For just adding a button, we might need to structure it differently or use gr.Blocks directly
        # For simplicity with gr.ChatInterface, let's add a button separately if possible, 
        # or integrate it if we switch to full gr.Blocks.
        # A simple way with current structure is to have a separate button that calls the function.
        # This might not integrate perfectly into the ChatInterface layout itself without more advanced Gradio usage.
        # For now, let's assume we can add a button that appears alongside or above/below.
        pass # Placeholder for now, will adjust UI structure in a moment

    # A better way to add custom components with gr.ChatInterface:
    with gr.Blocks() as demo:
        gr.Markdown("Roadmapper Test Chatbot")
        gr.Markdown("Ask questions before and after embedding documents to test the document loader agent. Ensure GCP settings are correct.")
        
        with gr.Row():
            load_docs_button = gr.Button("Load and Index Documents")
            # TODO: Add status indicators here, e.g., gr.Textbox(label="Status", interactive=False)
        
        chatbot_component = gr.Chatbot(height=500, label="Chat")
        msg_input = gr.Textbox(label="Your message:", show_label=False, placeholder="Type your message here...")
        clear_button = gr.Button("Clear Chat")

        # Wire up the chat logic
        msg_input.submit(get_chatbot_response, [msg_input, chatbot_component], [msg_input, chatbot_component])
        clear_button.click(lambda: (None, None), None, [msg_input, chatbot_component], queue=False)

        # Wire up the button to the STUBBED ASYNC trigger_document_loading_and_indexing function
        load_docs_button.click(trigger_document_loading_and_indexing, [], [])

    demo.launch()

# --- Simple Test Function for Button Click Debugging (Can be commented out or removed now) ---
# async def simple_test_button_click(): 
#     print("DEBUG: $$$ ASYNC GLOBAL simple_test_button_click FUNCTION CALLED $$$ DEBUG")
#     return "ASYNC GLOBAL simple_test_button_click successful (check console)!"

async def run_app():
    success = await initialize_document_pipeline() # Calls STUBBED version
    if not success:
        print("WARNING: Document pipeline failed to initialize (stubbed version should not fail).")

    with gr.Blocks(title="Document Loader Test Chatbot") as demo:
        gr.Markdown("## Roadmapper Test Chatbot")
        gr.Markdown("Ask questions before and after embedding documents to test the document loader agent. Ensure GCP settings are correct.")
        
        with gr.Row():
            load_docs_button = gr.Button("Load and Index Documents")
            status_display = gr.Textbox(label="Status", interactive=False, lines=2, max_lines=5)
        
        chatbot_component = gr.Chatbot(height=500, label="Chat")
        msg_input = gr.Textbox(label="Your message:", show_label=False, placeholder="Type your message here...")
        clear_button = gr.Button("Clear Chat")

        msg_input.submit(get_chatbot_response, [msg_input, chatbot_component], [msg_input, chatbot_component])
        clear_button.click(lambda: (None, None), None, [msg_input, chatbot_component], queue=False)

        # Wire up the button to the STUBBED ASYNC trigger_document_loading_and_indexing function
        load_docs_button.click(trigger_document_loading_and_indexing, [], [status_display])

    print("DEBUG: $$$ ABOUT TO CALL demo.launch() $$$ DEBUG")
    demo.launch()

if __name__ == "__main__":
    asyncio.run(run_app()) 