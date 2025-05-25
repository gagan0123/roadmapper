import os
from typing import List, Optional, Tuple, Set, AsyncGenerator
from googleapiclient.discovery import build, Resource
from googleapiclient.http import MediaIoBaseDownload
import io
import mimetypes
import PyPDF2
import pdfplumber
from ..models.base import Document
from .base import BaseLoader
import logging
import requests
from googleapiclient.errors import HttpError
from ..utils.gcp_utils import get_google_credentials
from google.oauth2.credentials import Credentials
import fitz # PyMuPDF

# Get a logger for this module
logger = logging.getLogger(__name__)

# Added Slides API scope for thumbnails
SCOPES = ['https://www.googleapis.com/auth/drive.readonly', 'https://www.googleapis.com/auth/presentations.readonly']
DEFAULT_MAX_IMAGES_PER_DOC = 5 # Added default
DEFAULT_EMBEDDING_MODEL_TYPE = "multimodal" # Added default for loaders

class DriveLoader(BaseLoader):
    SUPPORTED_MIME_TYPES = {
        "application/vnd.google-apps.document": "text/plain",
        "application/vnd.google-apps.spreadsheet": "text/csv",
        "application/vnd.google-apps.presentation": "text/plain", # Simplified, might need specific handling
        "application/pdf": "application/pdf",
        "text/plain": "text/plain",
        "text/csv": "text/csv",
        "image/jpeg": "image/jpeg",
        "image/png": "image/png",
    }
    # Add other supported image MIME types as needed

    # For Google Workspace types, we often export them to a common format (e.g., PDF or text)
    # and then process that common format.
    EXPORT_MIMES = {
        "application/vnd.google-apps.document": "application/pdf", # Export to PDF to get images and text
        "application/vnd.google-apps.spreadsheet": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", # Export to XLSX
        "application/vnd.google-apps.presentation": "text/plain", # Export to TEXT for content, images via Slides API
    }

    def __init__(self, item_id: str, category: Optional[str] = None, embedding_model_type: str = DEFAULT_EMBEDDING_MODEL_TYPE, user_credentials: Optional[Credentials] = None, drive_service: Optional[Resource] = None, max_images_per_doc: int = DEFAULT_MAX_IMAGES_PER_DOC):
        """
        Initializes the DriveLoader.

        Args:
            item_id: The ID of the Google Drive folder or file.
            category: The optional category to assign to the documents.
                      If loading a folder, this category can be overridden by subfolder names.
            embedding_model_type: The preferred embedding model type ("text" or "multimodal").
            user_credentials: Optional. User's OAuth2 credentials. If provided, these are used.
                              Otherwise, attempts to use service account credentials via gcp_utils.
            drive_service: An optional pre-initialized Google Drive API service (overrides credential logic).
            max_images_per_doc: Maximum number of images to extract from a document.
        """
        super().__init__()
        self.item_id = item_id
        self.initial_category = category # Store the category passed during initialization
        self.embedding_model_type = embedding_model_type # Store it
        self.max_images_per_doc = max_images_per_doc
        self.processed_item_ids: Set[str] = set() # To prevent infinite loops with shortcuts or complex structures
        self.drive_service = None
        self.slides_service = None

        try:
            if drive_service:
                logger.info("Using pre-initialized Drive service.")
                self.drive_service = drive_service
                # If drive_service is provided, assume slides_service should also be from a similar source or re-created.
                # For simplicity, if drive_service is passed, we might need a way to pass slides_service too, or build it from same creds if possible.
                # This part might need refinement if pre-initialized services are a common use case.
                if hasattr(drive_service, '_http') and hasattr(drive_service._http, 'credentials'):
                     self.slides_service = build('slides', 'v1', credentials=drive_service._http.credentials)
                else: # Fallback or raise error if slides can't be initialized
                    logger.warning("Pre-initialized drive_service provided without clear credentials for slides_service. Slides API features might fail.")
                    # Attempt to build slides_service with user_credentials if available, else service account
                    current_creds_for_slides = user_credentials if user_credentials else get_google_credentials(SCOPES)
                    if current_creds_for_slides:
                        self.slides_service = build('slides', 'v1', credentials=current_creds_for_slides)
                    else:
                        logger.error("Could not initialize slides_service with pre-built drive_service and no other credentials.")

            elif user_credentials:
                logger.info("Using provided user OAuth2 credentials for Drive and Slides services.")
                self.drive_service = build('drive', 'v3', credentials=user_credentials)
                self.slides_service = build('slides', 'v1', credentials=user_credentials)
            else:
                logger.info("Using service account credentials (via gcp_utils) for Drive and Slides services.")
                # This is the fallback to Application Default Credentials or service account key from env
                sa_creds = get_google_credentials(SCOPES) # SCOPES might be used by some ADC flows
                if not sa_creds:
                    raise ConnectionError("Failed to obtain any Google credentials (user or service account).")
                self.drive_service = build('drive', 'v3', credentials=sa_creds)
                self.slides_service = build('slides', 'v1', credentials=sa_creds)
            
            if not self.drive_service or not self.slides_service:
                 raise ConnectionError("Failed to initialize Google Drive and/or Slides API service clients.")

        except Exception as e:
            logger.error(f"Failed to initialize Google services: {e}")
            raise
        
        logging.info(f"DriveLoader initialized for item ID: {self.item_id} with category: {self.initial_category}. Model type: {self.embedding_model_type}")

    async def validate_connection(self) -> bool:
        """Validates the connection to Google Drive by trying to fetch the initial item's metadata."""
        if not self.drive_service:
            logger.error("Drive service not initialized. Validation failed.")
            return False
        try:
            # Try to get metadata for the initial item_id. A successful call indicates a valid connection.
            self.drive_service.files().get(fileId=self.item_id, fields="id").execute()
            logger.info(f"Successfully validated connection to Drive for item ID: {self.item_id}")
            return True
        except HttpError as e:
            logger.error(f"Failed to validate connection to Drive for item ID {self.item_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during Drive connection validation for item ID {self.item_id}: {e}")
            return False

    async def load(self) -> AsyncGenerator[Document, None]:
        """
        Loads documents from the specified Google Drive folder or file.

        Yields:
            Document objects.
        """
        if not self.drive_service:
            logger.error("Drive service not initialized. Cannot load documents.")
            return # Use return for empty async generator

        self.processed_item_ids.clear()
        # documents: List[Document] = [] # Removed list
        logging.info(f"Starting to load from Drive item ID: {self.item_id}")

        try:
            item_metadata = self.drive_service.files().get(
                fileId=self.item_id,
                fields="id, name, mimeType, webViewLink, parents" # Added webViewLink
            ).execute()
            logging.debug(f"Fetched metadata for item {self.item_id}: {item_metadata}")

            mime_type = item_metadata.get("mimeType")

            if mime_type == "application/vnd.google-apps.folder":
                logging.info(f"Item {self.item_id} ('{item_metadata.get('name')}') is a folder. Processing folder contents.")
                # Pass the initial category which might be None. _process_folder will handle category logic.
                async for doc in self._process_folder(self.item_id, item_metadata.get('name'), self.initial_category, self.embedding_model_type): # Changed to async for and yield from
                    yield doc
            else:
                logging.info(f"Item {self.item_id} ('{item_metadata.get('name')}') is a file. Processing as a single file.")
                # For a single file, its category is the initial_category passed to the loader.
                # Assuming _create_document_from_drive_item can be called directly. If it becomes async, this needs await.
                doc = self._create_document_from_drive_item(item_metadata, self.initial_category, self.embedding_model_type)
                if doc:
                    yield doc # Changed to yield
        except HttpError as e:
            logging.error(f"Error accessing Google Drive item {self.item_id}: {e}")
            # Potentially raise a custom exception or return
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading from item {self.item_id}: {e}")

        # logging.info(f"Finished loading from Drive item ID: {self.item_id}. Found {len(documents)} documents.") # Commented out/remove as we stream
        logging.info(f"Finished streaming documents from Drive item ID: {self.item_id}.")

    def _get_folder_name_and_confirm_folder(self, item_id: str) -> Tuple[Optional[str], bool]:
        """Gets the name of an item and confirms if it's a folder."""
        try:
            item_metadata = self.drive_service.files().get(fileId=item_id, fields="name, mimeType").execute()
            is_folder = item_metadata.get("mimeType") == "application/vnd.google-apps.folder"
            return item_metadata.get("name"), is_folder
        except HttpError as e:
            logger.error(f"Error fetching metadata for item {item_id}: {e}")
            return None, False

    async def _process_folder(self, folder_id: str, folder_name: Optional[str], current_category: Optional[str], embedding_model_type: str) -> AsyncGenerator[Document, None]:
        """
        Recursively processes a folder's contents.
        The category is determined by:
        1. The current_category passed down (which could be from a parent or the initial category).
        2. If the folder_name itself is a valid category (e.g., "Strategy", "Product Design"), it overrides.
           This needs a mechanism to check if a folder name *is* a category,
           perhaps by checking against a predefined list or a naming convention.
           For now, let's assume folder names can act as categories.
        """
        if folder_id in self.processed_item_ids:
            logger.warning(f"Skipping already processed folder ID: {folder_id}")
            return # Use return for empty async generator
        self.processed_item_ids.add(folder_id)

        # Determine effective category for items in this folder
        # For simplicity, if folder_name is not None/empty, it's used as category.
        # Otherwise, the current_category (from parent or initial) is used.
        effective_category = folder_name if folder_name else current_category
        # The embedding_model_type is passed down from the parent or initial call
        # It does not change per sub-folder in this logic.
        logger.info(f"Processing folder '{folder_name}' (ID: {folder_id}) with category: '{effective_category}', model_type: '{embedding_model_type}'")

        # documents: List[Document] = [] # Removed list
        try:
            query = f"'{folder_id}' in parents and trashed = false"
            page_token = None
            while True:
                results = self.drive_service.files().list(
                    q=query,
                    pageSize=100, # Adjust as needed
                    fields="nextPageToken, files(id, name, mimeType, webViewLink)", # Added webViewLink
                    pageToken=page_token
                ).execute()
                items = results.get('files', [])

                for item in items:
                    item_id_loop = item['id'] # Renamed to avoid conflict with outer scope item_id if any confusion
                    item_name = item.get('name')
                    item_mime_type = item.get('mimeType')
                    logger.debug(f"Found item: '{item_name}' (ID: {item_id_loop}, Type: {item_mime_type}) in folder '{folder_name}'")

                    if item_id_loop in self.processed_item_ids: # Avoid processing items multiple times if linked in multiple places
                        logger.warning(f"Skipping already processed item ID: {item_id_loop} ('{item_name}')")
                        continue
                    
                    # Check for shortcuts and resolve them, but be careful about recursion
                    if item_mime_type == "application/vnd.google-apps.shortcut":
                        target_id = item.get('shortcutDetails', {}).get('targetId')
                        if target_id:
                            logger.info(f"Found shortcut '{item_name}' pointing to {target_id}. Resolving.")
                            # Get target item's metadata to decide how to process
                            try:
                                target_metadata = self.drive_service.files().get(
                                    fileId=target_id,
                                    fields="id, name, mimeType, webViewLink"
                                ).execute()
                                target_mime_type = target_metadata.get("mimeType")
                                if target_mime_type == "application/vnd.google-apps.folder":
                                    # If shortcut target is a folder, process it.
                                    target_folder_name = target_metadata.get('name')
                                    # Pass model type for shortcut to folder
                                    async for doc_from_shortcut_folder in self._process_folder(target_id, target_folder_name, effective_category, embedding_model_type):
                                        yield doc_from_shortcut_folder
                                else:
                                    # If shortcut target is a file, create a document.
                                    doc = self._create_document_from_drive_item(target_metadata, effective_category, embedding_model_type)
                                    if doc:
                                        yield doc # Changed to yield
                                        self.processed_item_ids.add(target_id) # Mark target as processed
                            except HttpError as e:
                                logger.error(f"Error resolving shortcut target {target_id} for '{item_name}': {e}")
                        else:
                            logger.warning(f"Shortcut '{item_name}' has no targetId.")
                        self.processed_item_ids.add(item_id_loop) # Mark shortcut itself as processed
                        continue # Move to next item after handling shortcut


                    if item_mime_type == "application/vnd.google-apps.folder":
                        # Pass model type to sub-folder processing
                        async for doc_from_folder in self._process_folder(item_id_loop, item_name, effective_category, embedding_model_type): # Changed to async for and yield from
                            yield doc_from_folder
                    elif item_mime_type in self.SUPPORTED_MIME_TYPES or item_mime_type in self.EXPORT_MIMES:
                        # Pass model type to file processing
                        doc = self._create_document_from_drive_item(item, effective_category, embedding_model_type)
                        if doc:
                            yield doc # Changed to yield
                            self.processed_item_ids.add(item_id_loop) # Mark file as processed
                    else:
                        logger.warning(f"Unsupported MIME type '{item_mime_type}' for file '{item_name}' (ID: {item_id_loop}). Skipping.")
                        self.processed_item_ids.add(item_id_loop) # Mark as processed to avoid re-evaluating

                page_token = results.get('nextPageToken')
                if not page_token:
                    break
        except HttpError as e:
            logger.error(f"Error listing files in Google Drive folder {folder_id} ('{folder_name}'): {e}")
        # Removed return documents, as this is a generator now

    def _create_document_from_drive_item(self, item_metadata: dict, category: Optional[str], embedding_model_type: str) -> Optional[Document]:
        """
        Creates a Document object from a Google Drive file item's metadata.
        This involves downloading, extracting text, and potentially images.
        """
        item_id = item_metadata['id']
        item_name = item_metadata.get('name', 'Untitled Drive Item')
        mime_type = item_metadata.get('mimeType')
        source_url = item_metadata.get('webViewLink', f"https://docs.google.com/document/d/{item_id}") # Fallback for safety

        logger.info(f"Processing file: '{item_name}' (ID: {item_id}, Type: {mime_type}), Category: {category}, ModelType: {embedding_model_type}")

        content = ""
        images: List[bytes] = []

        try:
            # Determine the effective MIME type for processing (export if necessary)
            process_mime_type = self.EXPORT_MIMES.get(mime_type, mime_type)
            logger.debug(f"DRIVE_LOADER_DEBUG: Item: '{item_name}' (ID: {item_id}), Original MimeType: '{mime_type}', Process MimeType: '{process_mime_type}'") # DEBUG LOG

            if process_mime_type == "application/pdf" or mime_type == "application/pdf": # Original is PDF or exported to PDF
                logger.debug(f"DRIVE_LOADER_DEBUG: Item: '{item_name}' entering PDF processing path.") # DEBUG LOG
                file_bytes = self._download_file_bytes(item_id, export_mime_type=(None if mime_type == "application/pdf" else "application/pdf") )
                if file_bytes:
                    content, images = self._extract_text_and_images_from_pdf_bytes(file_bytes)
            elif process_mime_type == "text/plain" or process_mime_type == "text/csv":
                # Determine if an export is needed for text content
                text_export_mime = None
                if mime_type in self.EXPORT_MIMES and self.EXPORT_MIMES[mime_type] == "text/plain": # e.g. GSlides to text
                    text_export_mime = "text/plain"
                # if mime_type is already text/plain or text/csv, no export is needed (text_export_mime remains None)
                
                file_bytes = self._download_file_bytes(item_id, export_mime_type=text_export_mime)
                if file_bytes:
                    content = file_bytes.decode('utf-8', errors='replace')
                
                # If the original type was a presentation, also fetch its images
                if mime_type == "application/vnd.google-apps.presentation":
                    logger.debug(f"DRIVE_LOADER_DEBUG: Item: '{item_name}' is GSlides, attempting to fetch images.") # DEBUG LOG
                    try:
                        logger.debug(f"Fetching images for presentation {item_name} (ID: {item_id})")
                        slide_image_urls = self._get_slide_ids_and_image_urls(item_id) # This method uses slides_service
                        for url in slide_image_urls[:self.max_images_per_doc]: # Respect limit
                            img_bytes = self._download_image_from_url(url) # This method uses requests.get
                            if img_bytes:
                                images.append(img_bytes)
                            if len(images) >= self.max_images_per_doc:
                                logger.info(f"Reached max images ({self.max_images_per_doc}) for presentation {item_name}, stopping further slide image extraction.")
                                break
                    except Exception as e:
                        logger.error(f"Error extracting images from presentation {item_name} (ID: {item_id}): {e}")

            elif mime_type.startswith("image/"): # Native image types
                file_bytes = self._download_file_bytes(item_id)
                if file_bytes:
                    images.append(file_bytes)
                    # Potentially add filename as content or leave content empty for pure images
                    content = f"Image: {item_name}" if not content else content
            elif process_mime_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": # Spreadsheets (e.g. Google Sheets exported to XLSX)
                # For spreadsheets, we'll attempt to extract text content.
                # This is a placeholder; robust spreadsheet parsing would require libraries like openpyxl or pandas.
                file_bytes = self._download_file_bytes(item_id, export_mime_type=self.EXPORT_MIMES.get(mime_type))
                if file_bytes:
                    # Basic text extraction: Could be improved with actual XLSX parsing
                    # For now, we'll just indicate it's a spreadsheet.
                    content = f"Spreadsheet content from: {item_name}. Further parsing needed for full data."
                    logger.warning(f"Spreadsheet '{item_name}' content extraction is basic. Consider using a dedicated parser for XLSX.")

            else:
                logger.warning(f"Cannot extract content for MIME type '{mime_type}' (processed as '{process_mime_type}') for file '{item_name}'.")
                return None


            if not content and not images:
                logger.warning(f"No content or images extracted from file '{item_name}' (ID: {item_id}). Skipping document creation.")
                return None

            return Document(
                id=f"drive_{item_id}",
                title=item_name,
                content=content.strip() if content else "",
                source_type="google_drive",
                source_url=source_url,
                category=category,
                images=images[:self.max_images_per_doc], # Respect max_images_per_doc
                embedding_model_type=embedding_model_type, # Pass it here
                metadata={
                    "drive_file_id": item_id,
                    "original_mime_type": mime_type,
                    "processed_mime_type": process_mime_type
                }
            )

        except HttpError as e:
            logger.error(f"HTTP error processing file '{item_name}' (ID: {item_id}): {e}")
        except Exception as e:
            logger.error(f"Unexpected error creating document from Drive item '{item_name}' (ID: {item_id}): {e}")
        
        return None

    def _download_file_bytes(self, file_id: str, export_mime_type: Optional[str] = None) -> Optional[bytes]:
        """Downloads a file or exports it and returns its content as bytes."""
        try:
            if export_mime_type:
                request = self.drive_service.files().export_media(fileId=file_id, mimeType=export_mime_type)
                logger.debug(f"Exporting file {file_id} as {export_mime_type}")
            else:
                request = self.drive_service.files().get_media(fileId=file_id)
                logger.debug(f"Downloading file {file_id}")
            
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                logger.debug(f"Download/Export for {file_id}: {int(status.progress() * 100)}% complete.")
            
            logger.info(f"Successfully downloaded/exported file {file_id}.")
            return fh.getvalue()
        except HttpError as e:
            logger.error(f"Error downloading/exporting file {file_id} (export type {export_mime_type}): {e}")
            # If a specific export fails (e.g., empty GSlides presentation), it might throw 403
            if e.resp.status == 403 and export_mime_type:
                 logger.warning(f"Could not export {file_id} as {export_mime_type}. It might be an empty document or have unsupported features for export.")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during file bytes download for {file_id}: {e}")
            return None

    def _extract_text_and_images_from_pdf_bytes(self, pdf_bytes: bytes) -> Tuple[str, List[bytes]]:
        """Extracts text and images from PDF bytes using PyMuPDF."""
        text_content = ""
        images: List[bytes] = []
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text_content += page.get_text() + "\\n" # Add newline between pages
                
                # Limit images per page or per document as needed
                if len(images) < self.max_images_per_doc:
                    image_list = page.get_images(full=True)
                    for img_index, img in enumerate(image_list):
                        if len(images) >= self.max_images_per_doc:
                            logger.info(f"Reached max images ({self.max_images_per_doc}), skipping further image extraction from PDF.")
                            break
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        images.append(image_bytes)
            logger.info(f"Extracted {len(images)} images and {len(text_content)} chars of text from PDF.")
        except Exception as e:
            logger.error(f"Error extracting text/images from PDF: {e}")
        return text_content.strip(), images

    def _get_slide_ids_and_image_urls(self, presentation_id: str) -> List[str]:
        """
        Fetches slide IDs and their thumbnail content URLs for a presentation.
        Returns a list of thumbnail URLs.
        """
        image_urls = []
        try:
            if not self.slides_service: # Should have been initialized in __init__
                logger.error("Slides service not initialized. Cannot fetch slide images.")
                return []

            presentation = self.slides_service.presentations().get(presentationId=presentation_id).execute()
            slides = presentation.get('slides', [])
            
            count = 0
            for slide in slides:
                if count >= self.max_images_per_doc:
                    logger.info(f"Reached max_images_per_doc ({self.max_images_per_doc}) while fetching slide image URLs for {presentation_id}.")
                    break
                
                page_object_id = slide.get('objectId')
                if page_object_id:
                    thumbnail_request = self.slides_service.presentations().pages().getThumbnail(
                        presentationId=presentation_id,
                        pageObjectId=page_object_id,
                        thumbnailProperties_mimeType='PNG', 
                        thumbnailProperties_thumbnailSize='LARGE' 
                    )
                    thumbnail_response = thumbnail_request.execute()
                    content_url = thumbnail_response.get('contentUrl')
                    if content_url:
                        image_urls.append(content_url)
                        count += 1
                    else:
                        logger.warning(f"No contentUrl found for slide {page_object_id} in presentation {presentation_id}")
            logger.info(f"Found {len(image_urls)} image URLs for presentation {presentation_id}")
        except HttpError as e:
            logger.error(f"Google API HTTP error getting slide images for {presentation_id}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error getting slide images for {presentation_id}: {e}")
        return image_urls

    def _download_image_from_url(self, url: str) -> Optional[bytes]:
        """Downloads image bytes from a URL."""
        try:
            response = requests.get(url, timeout=10) 
            response.raise_for_status() 
            return response.content
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading image from URL {url}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error downloading image {url}: {e}")
        return None

    def _is_valid_category(self, category_name: Optional[str]) -> bool:
        """
        Checks if a given string is a valid category.
        Placeholder: Implement actual validation (e.g., against a predefined list, regex).
        """
        if category_name and category_name.strip(): # Basic check: not None and not empty/whitespace
            # Example: return category_name in ["Strategy", "Product Design", "Technical Spec"]
            return True # For now, any non-empty string is "valid"
        return False

    # Helper to get direct parent's name - might be useful for more complex category logic later
    def _get_parent_folder_info(self, item_metadata: dict) -> Optional[Tuple[str, str]]:
        """Gets ID and name of the first parent folder if available."""
        parents = item_metadata.get("parents")
        if parents:
            parent_id = parents[0] # Assuming direct parent is most relevant
            try:
                parent_meta = self.drive_service.files().get(fileId=parent_id, fields="id, name").execute()
                return parent_meta.get("id"), parent_meta.get("name")
            except HttpError as e:
                logger.error(f"Could not fetch parent folder info for {parent_id}: {e}")
        return None 