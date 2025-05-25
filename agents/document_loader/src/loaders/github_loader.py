import os
from typing import AsyncGenerator, Optional
from github import Github, GithubException
from ..models.base import Document
from .base import BaseLoader
import base64
import mimetypes
import PyPDF2
import io
import requests
import asyncio
import pdfplumber
import re
from collections import deque
import logging

# Get a logger for this module
logger = logging.getLogger(__name__)

DEFAULT_MAX_IMAGES_PER_DOC = 10 # Added default
DEFAULT_EMBEDDING_MODEL_TYPE = "multimodal" # Added

class GitHubLoader(BaseLoader):
    def __init__(self, repo_name: str, token: str = None, max_retries: int = 3, max_images_per_doc: int = DEFAULT_MAX_IMAGES_PER_DOC, category: Optional[str] = None, embedding_model_type: str = DEFAULT_EMBEDDING_MODEL_TYPE):
        self.repo_name = repo_name
        self.token = token or os.getenv('GITHUB_TOKEN')
        self.github = None
        self.repo = None
        self.max_retries = max_retries
        self.max_images_per_doc = max_images_per_doc # Initialize max_images_per_doc
        self.category = category # Store category
        self.embedding_model_type = embedding_model_type # Store it

    async def validate_connection(self) -> bool:
        try:
            logger.debug(f"Attempting to connect to GitHub repo: {self.repo_name}")
            self.github = Github(self.token)
            self.repo = self.github.get_repo(self.repo_name)
            logger.info(f"Successfully connected to: https://github.com/{self.repo_name}")
            if self.token:
                logger.debug(f"Using GitHub token: {self.token[:6]}...{self.token[-4:]}")
            else:
                logger.debug("No GitHub token provided; using anonymous access if configured.")
            return True
        except Exception as e:
            logger.error(f"Error validating GitHub connection to {self.repo_name}: {e}", exc_info=True)
            return False

    async def _retry_with_backoff(self, func, *args, **kwargs) -> AsyncGenerator[Document, None]:
        for attempt in range(self.max_retries):
            try:
                async for item in func(*args, **kwargs):
                    yield item
                return # Successful, exit retry loop
            except (GithubException, requests.exceptions.RequestException) as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Max retries reached for GitHub operation. Error: {e}", exc_info=True)
                    raise
                wait = 2 ** attempt
                logger.warning(f"GitHub operation failed (attempt {attempt+1}/{self.max_retries}): {e}. Retrying in {wait} seconds...")
                await asyncio.sleep(wait)
            # Restore generic exception handler
            except Exception as e: # Catch other unexpected errors
                if attempt == self.max_retries - 1:
                    logger.error(f"Max retries reached. Unexpected error in GitHub operation: {e}", exc_info=True)
                    raise
                wait = 2 ** attempt
                logger.warning(f"Unexpected error in GitHub operation (attempt {attempt+1}/{self.max_retries}): {e}. Retrying in {wait} seconds...")
                await asyncio.sleep(wait)

    async def load(self) -> AsyncGenerator[Document, None]:
        if not self.repo:
            if not await self.validate_connection():
                raise Exception("Failed to establish GitHub connection")

        async def process_contents() -> AsyncGenerator[Document, None]:
            # This is now an async generator
            try:
                logger.info(f"Loading contents from repository: https://github.com/{self.repo_name}")
                loop = asyncio.get_running_loop()
                # Initial contents retrieval
                contents_list = await loop.run_in_executor(None, lambda: list(self.repo.get_contents("")))
                
                # Use a deque for efficient pop from left
                contents_deque = deque(contents_list)

                while contents_deque:
                    file_content = contents_deque.popleft()
                    if file_content.type == "dir":
                        # Get directory contents non-blockingly
                        logger.debug(f"Processing directory: {file_content.path}")
                        try:
                            more_contents = await loop.run_in_executor(None, lambda path=file_content.path: list(self.repo.get_contents(path)))
                            contents_deque.extend(more_contents)
                        except Exception as dir_e:
                            logger.error(f"Error getting contents for directory {file_content.path}: {dir_e}", exc_info=True)
                            continue # Skip this directory if there's an error
                    else:
                        filename = file_content.name
                        images = []
                        # Only process .md, .txt, and .pdf files
                        if filename.lower().endswith('.md'):
                            try:
                                content = base64.b64decode(file_content.content).decode('utf-8')
                                # Extract image links from markdown
                                image_links = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', content)
                                for url in image_links:
                                    if len(images) >= self.max_images_per_doc:
                                        logger.info(f"Reached max images ({self.max_images_per_doc}) for Markdown {filename}, stopping image extraction.")
                                        break # Stop if max images reached
                                    
                                    # Check for base64 data URL
                                    if url.startswith('data:image/'):
                                        try:
                                            # data:image/png;base64,iVBORw0KGgo...
                                            header, encoded_data = url.split(',', 1)
                                            # Further check mime type if necessary, e.g. header.split(';')[0].split('/')[1] for 'png'
                                            # For now, we assume it's a valid image type if it matches data:image/
                                            if not header.startswith('data:image/') or ';base64' not in header:
                                                logger.warning(f"Skipping malformed data URL: {url[:100]}... in Markdown file {filename}")
                                                continue
                                            
                                            img_bytes = base64.b64decode(encoded_data)
                                            images.append(img_bytes)
                                            logger.debug(f"Successfully decoded base64 image from Markdown {filename}.")
                                        except Exception as e:
                                            logger.warning(f"Error decoding base64 image from data URL in {filename}: {e}. URL: {url[:100]}...")
                                        continue # Move to the next URL

                                    absolute_url = ""
                                    try:
                                        if url.startswith(('http://', 'https://')):
                                            absolute_url = url
                                        else:
                                            # Construct base URL for raw content
                                            # Default to main/master, or fetch default_branch if needed
                                            # For simplicity, let's try to get default_branch, 
                                            # but this might require an extra API call if not already fetched.
                                            # A common pattern is that self.repo.default_branch is available.
                                            default_branch = self.repo.default_branch
                                            
                                            # Get the directory of the current markdown file
                                            markdown_file_dir = os.path.dirname(file_content.path)
                                            if markdown_file_dir == ".": # Handles files in root
                                                markdown_file_dir = ""
                                                
                                            # Join the markdown file's directory with the relative image URL
                                            # Normalize the path to handle '..' etc.
                                            normalized_image_path = os.path.normpath(os.path.join(markdown_file_dir, url.lstrip('/')))
                                            
                                            # Prevent escaping the repository root if path is ../../../image.png
                                            if normalized_image_path.startswith(".."):
                                                logger.warning(f"Skipping image with relative path potentially traversing out of repo: '{url}' in {filename}")
                                                continue

                                            absolute_url = f"https://raw.githubusercontent.com/{self.repo.full_name}/{default_branch}/{normalized_image_path}"
                                        
                                        if not absolute_url:
                                            logger.warning(f"Could not resolve URL for image '{url}' in Markdown file {filename}")
                                            continue

                                        # Use asyncio.to_thread (via run_in_executor) for the blocking requests.get() call
                                        # This whole block handles the download and error reporting for a single image URL
                                        try:
                                            loop = asyncio.get_running_loop()
                                            # For Python 3.9+ you could use: response = await asyncio.to_thread(requests.get, absolute_url)
                                            response = await loop.run_in_executor(None, lambda: requests.get(absolute_url, timeout=10)) # Added timeout
                                            
                                            if response.status_code == 200:
                                                images.append(response.content)
                                            else:
                                                logger.warning(f"Failed to download image {absolute_url} (status: {response.status_code}) for Markdown file {filename}")
                                        except requests.exceptions.MissingSchema:
                                            logger.warning(f"Invalid URL (missing schema) for image '{url}' (resolved to {absolute_url}) in Markdown file {filename}")
                                        except requests.exceptions.Timeout:
                                            logger.warning(f"Timeout downloading image {absolute_url} for Markdown file {filename}")
                                        except Exception as e:
                                            logger.warning(f"Error downloading image '{url}' (resolved to {absolute_url}) in Markdown file {filename}: {e}")
                                    except requests.exceptions.MissingSchema:
                                        logger.warning(f"Invalid URL (missing schema) for image '{url}' in Markdown file {filename}")
                                    except Exception as e:
                                        logger.warning(f"Generic error processing image link '{url}' in Markdown file {filename}: {e}")
                            except UnicodeDecodeError as e:
                                logger.error(f"Error decoding Markdown file {filename} as UTF-8: {e}")
                                continue
                            except Exception as e:
                                logger.error(f"Error processing Markdown file {filename}: {e}", exc_info=True)
                                continue
                        elif filename.lower().endswith('.txt'):
                            try:
                                content = base64.b64decode(file_content.content).decode('utf-8')
                            except UnicodeDecodeError as e:
                                logger.error(f"Error decoding text file {filename} as UTF-8: {e}")
                                continue
                            except Exception as e:
                                logger.error(f"Error processing text file {filename}: {e}", exc_info=True)
                                continue
                        elif filename.lower().endswith('.pdf'):
                            try:
                                file_bytes = base64.b64decode(file_content.content)
                                # Extract text
                                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
                                content = ""
                                for page in pdf_reader.pages:
                                    content += page.extract_text() or ""
                                # Extract images using pdfplumber
                                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                                    full_text_parts = []
                                    for i, page in enumerate(pdf.pages):
                                        page_text = page.extract_text()
                                        if page_text:
                                            full_text_parts.append(page_text)
                                        
                                        # Image extraction from PDF
                                        if len(images) < self.max_images_per_doc:
                                            for img_idx, img_obj in enumerate(page.images):
                                                if len(images) >= self.max_images_per_doc:
                                                    logger.info(f"Reached max images ({self.max_images_per_doc}) for PDF {filename}, stopping image extraction.")
                                                    break
                                                try:
                                                    # Corrected coordinate fetching based on common pdfplumber structure
                                                    x0, top, x1, bottom = img_obj['x0'], img_obj['top'], img_obj['x1'], img_obj['bottom']
                                                    
                                                    # Basic validation of coordinates
                                                    if not (isinstance(x0, (int, float)) and isinstance(top, (int, float)) and 
                                                            isinstance(x1, (int, float)) and isinstance(bottom, (int, float)) and 
                                                            x1 > x0 and bottom > top and x0 >= 0 and top >= 0 and 
                                                            x1 <= page.width and bottom <= page.height):
                                                        logger.warning(f"Invalid image coordinates in PDF {filename}, page {i+1}, img {img_idx}. Skipping.")
                                                        continue
                                                        
                                                    cropped = page.crop((x0, top, x1, bottom)).to_image(resolution=150)
                                                    img_data = io.BytesIO()
                                                    cropped.save(img_data, format="PNG")
                                                    images.append(img_data.getvalue())
                                                except Exception as img_e:
                                                    logger.warning(f"Error extracting image {img_idx} from page {i+1} of PDF {filename}: {img_e}")
                                            if len(images) >= self.max_images_per_doc:
                                                break # Break from page loop if max images reached
                                    content = "\n".join(full_text_parts)
                            except Exception as e:
                                # This outer except catches errors during PDF text extraction or if pdfplumber.open fails
                                logger.error(f"Error extracting text or initializing image extraction for PDF {filename}: {e}", exc_info=True)
                                if not content: # If content is still empty, means text extraction likely failed
                                    content = "" # Ensure content is a string
                                # We will yield the doc with whatever content/images were processed before this error
                                continue # Skip to next file in repo if this PDF has a major processing error
                        else:
                            logger.debug(f"Skipping unsupported file in GitHub: {filename} (type: {file_content.type}, path: {file_content.path})")
                            continue
                        doc = Document(
                            id=file_content.sha, # Use SHA as a unique ID for the file version
                            title=file_content.name,
                            content=content,
                            source_type='github',
                            source_url=file_content.html_url,
                            images=images,
                            category=self.category, # Assign category
                            embedding_model_type=self.embedding_model_type, # Pass it here
                            metadata={
                                'path': file_content.path,
                                'sha': file_content.sha,
                                'size': file_content.size
                            }
                        )
                        yield doc
            except (GithubException, requests.exceptions.RequestException) as e:
                logger.error(f"GitHub API or Network error during content processing: {e}", exc_info=True)
                # No longer need to specifically print traceback for NameError here
                raise 
            except Exception as e:
                logger.error(f"Unexpected error during content processing: {e}", exc_info=True)
                import traceback # Keep this for general unexpected errors during development
                traceback.print_exc()
                raise # Re-raise

        async for doc in self._retry_with_backoff(process_contents):
            yield doc 