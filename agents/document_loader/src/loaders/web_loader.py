import aiohttp
from typing import AsyncGenerator, List, Optional
from bs4 import BeautifulSoup
from ..models.base import Document
from .base import BaseLoader
import asyncio
from urllib.parse import urlparse, urljoin
import re
import base64
import logging

# Get a logger for this module
logger = logging.getLogger(__name__)

# Supported image MIME types and their common extensions
# This can be expanded. Used for basic filtering.
SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")
MAX_IMAGES_PER_DOC = 10 # Arbitrary limit to prevent too many images
DEFAULT_EMBEDDING_MODEL_TYPE = "multimodal" # Added

class WebLoader(BaseLoader):
    def __init__(self, urls: List[str], 
                 category: Optional[str] = None, 
                 embedding_model_type: str = DEFAULT_EMBEDDING_MODEL_TYPE): # Added
        if not urls:
            raise ValueError("URLs list cannot be empty for WebLoader.")
        self.urls = urls
        self.category = category
        self.embedding_model_type = embedding_model_type # Store it
        self.session = None

    async def validate_connection(self) -> bool:
        try:
            self.session = aiohttp.ClientSession(headers={"User-Agent": "Mozilla/5.0"})
            # Try to fetch a test URL
            async with self.session.get('https://example.com') as response:
                return response.status == 200
        except Exception as e:
            # print(f"Error validating web connection: {e}")
            logger.error(f"Error validating web connection: {e}", exc_info=True)
            return False

    async def load(self) -> AsyncGenerator[Document, None]:
        if not self.session:
            if not await self.validate_connection():
                raise Exception("Failed to establish web connection")

        async def fetch_url(url: str, category_for_url: Optional[str], model_type_for_url: str) -> Document:
            try:
                async with self.session.get(url) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to fetch {url}: {response.status}")
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    # Get text content from body if possible, otherwise from the whole soup
                    text_source = soup.body if soup.body else soup
                    text = text_source.get_text()
                    
                    # Clean up text
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = ' '.join(chunk for chunk in chunks if chunk)
                    
                    # Get title
                    title = soup.title.string if soup.title else urlparse(url).netloc

                    # --- Extract Images --- 
                    image_bytes_list: List[bytes] = []
                    if self.session: # Ensure session is available for image downloads
                        img_tags = soup.find_all('img', limit=MAX_IMAGES_PER_DOC * 2) # Find more initially to filter later
                        image_download_tasks = []

                        for img_tag in img_tags:
                            if len(image_bytes_list) >= MAX_IMAGES_PER_DOC:
                                break
                            
                            img_src = img_tag.get('src')
                            if not img_src:
                                continue

                            # Handle base64 encoded data URLs
                            if img_src.startswith('data:image'):
                                try:
                                    header, encoded = img_src.split(',', 1)
                                    # Check if it's a supported image type from the data URL header
                                    if any(ext in header for ext in ['png', 'jpeg', 'gif', 'webp']):
                                        img_data = base64.b64decode(encoded)
                                        image_bytes_list.append(img_data)
                                    else:
                                        # print(f"Skipping data URL with unsupported image type: {header[:30]}")
                                        logger.debug(f"Skipping data URL with unsupported image type: {header[:30]}")
                                except Exception as e:
                                    # print(f"Error processing data URL {img_src[:30]}...: {e}")
                                    logger.warning(f"Error processing data URL {img_src[:30]}...: {e}")
                                continue # Move to next image tag

                            # Resolve relative URLs to absolute
                            absolute_img_url = urljoin(url, img_src)
                            parsed_img_url = urlparse(absolute_img_url)

                            # Basic filter for image extensions
                            if not parsed_img_url.path.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
                                # print(f"Skipping image with unsupported extension: {absolute_img_url}")
                                logger.debug(f"Skipping image with unsupported extension: {absolute_img_url}")
                                continue
                            
                            # Add task to download the image
                            # Use a separate function to capture variables correctly in async tasks
                            async def download_image(img_url_to_download: str):
                                try:
                                    async with self.session.get(img_url_to_download) as img_response:
                                        if img_response.status == 200:
                                            return await img_response.read()
                                        else:
                                            # print(f"Failed to download image {img_url_to_download}: {img_response.status}")
                                            logger.warning(f"Failed to download image {img_url_to_download}: {img_response.status}")
                                except Exception as e:
                                    # print(f"Error downloading image {img_url_to_download}: {e}")
                                    logger.warning(f"Error downloading image {img_url_to_download}: {e}")
                                return None
                            
                            image_download_tasks.append(download_image(absolute_img_url))

                        if image_download_tasks:
                            downloaded_images_data = await asyncio.gather(*image_download_tasks)
                            for img_data in downloaded_images_data:
                                if img_data and len(image_bytes_list) < MAX_IMAGES_PER_DOC:
                                    image_bytes_list.append(img_data)
                    # --- End Extract Images ---

                    return Document(
                        title=title,
                        content=text,
                        source_type='web',
                        source_url=url,
                        images=image_bytes_list, # Add downloaded image bytes
                        category=category_for_url,
                        embedding_model_type=model_type_for_url, # Pass it here
                        metadata={
                            'domain': urlparse(url).netloc,
                            'content_type': response.headers.get('content-type', '')
                        }
                    )
            except Exception as e:
                # print(f"Error processing URL {url}: {e}")
                logger.error(f"Error processing URL {url}: {e}")
                return None

        # Process URLs concurrently
        tasks = [fetch_url(url, self.category, self.embedding_model_type) for url in self.urls]
        documents = await asyncio.gather(*tasks)
        
        for doc in documents:
            if doc:
                yield doc

        await self.session.close() 