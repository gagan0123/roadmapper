import unittest
from unittest.mock import MagicMock, AsyncMock, patch, call
import base64
import asyncio

from agents.document_loader.src.loaders.github_loader import GitHubLoader
from agents.document_loader.src.models.base import Document

# A sample base64 encoded PNG (1x1 transparent pixel)
SAMPLE_B64_IMAGE_DATA = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
SAMPLE_B64_IMAGE_BYTES = base64.b64decode(SAMPLE_B64_IMAGE_DATA)

class TestGitHubLoader(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.repo_name = "user/repo"
        self.mock_github_api = MagicMock()
        self.mock_repo = MagicMock()
        self.mock_github_api.get_repo.return_value = self.mock_repo
        
        # Mock default_branch attribute
        self.mock_repo.default_branch = "main"

        # Patch Github class instantiation to return our mock
        self.github_patcher = patch('agents.document_loader.src.loaders.github_loader.Github', return_value=self.mock_github_api)
        self.mock_github_constructor = self.github_patcher.start()

        self.loader = GitHubLoader(repo_name=self.repo_name, token="fake_token")
        # We need to manually set the repo and github object if validate_connection is not called in a test
        # or if we are bypassing it.
        self.loader.github = self.mock_github_api
        self.loader.repo = self.mock_repo
        
    def tearDown(self):
        self.github_patcher.stop()

    async def test_load_markdown_with_base64_and_remote_image(self):
        markdown_content = f"""
# Test Document

This is a test markdown file.

## Base64 Image
![Base64 Image](data:image/png;base64,{SAMPLE_B64_IMAGE_DATA})

## Remote Image
![Remote Image](http{':'}//example.com/remote_image.jpg)

Some more text.
"""
        encoded_markdown_content = base64.b64encode(markdown_content.encode('utf-8')).decode('utf-8')

        mock_file_content = MagicMock()
        mock_file_content.name = "test.md"
        mock_file_content.path = "test.md"
        mock_file_content.type = "file"
        mock_file_content.content = encoded_markdown_content
        mock_file_content.html_url = "https://github.com/user/repo/blob/main/test.md"
        mock_file_content.sha = "12345"
        mock_file_content.size = len(markdown_content)
        
        self.mock_repo.get_contents.return_value = [mock_file_content]

        # Mock requests.get for the remote image
        mock_remote_image_bytes = b"remote_image_bytes"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = mock_remote_image_bytes
        
        # Since requests.get is called within run_in_executor, we need to ensure the mock
        # is picklable or that the lambda resolves to a coroutine if AsyncMock is used directly.
        # Here, the lambda calls a plain MagicMock.
        mock_requests_get = MagicMock(return_value=mock_response)

        documents = []
        # Patch 'requests.get' within the scope of this test method
        with patch('requests.get', mock_requests_get):
            # We need to simulate that validate_connection was successful
            self.loader.repo = self.mock_repo # Ensure repo is set
            async for doc in self.loader.load():
                documents.append(doc)

        self.assertEqual(len(documents), 1)
        doc = documents[0]

        self.assertEqual(doc.title, "test.md")
        self.assertIn("Test Document", doc.content)
        self.assertIn("Base64 Image", doc.content)
        self.assertIn("Remote Image", doc.content)
        self.assertEqual(len(doc.images), 2)
        
        # Check base64 image
        self.assertEqual(doc.images[0], SAMPLE_B64_IMAGE_BYTES)
        
        # Check remote image
        self.assertEqual(doc.images[1], mock_remote_image_bytes)
        
        self.mock_repo.get_contents.assert_called_once_with("")
        mock_requests_get.assert_called_once_with("http://example.com/remote_image.jpg", timeout=10)

    async def test_load_markdown_max_images(self):
        max_images = 1 # Test with a limit of 1 image
        # Re-initialize loader with specific max_images_per_doc
        self.loader = GitHubLoader(repo_name=self.repo_name, token="fake_token", max_images_per_doc=max_images)
        # Manually set the repo and github object as validate_connection is not called
        self.loader.github = self.mock_github_api
        self.loader.repo = self.mock_repo


        markdown_content = f"""
# Test Max Images
This markdown has multiple images.

![Base64 Image](data:image/png;base64,{SAMPLE_B64_IMAGE_DATA})
![Remote Image 1](http://example.com/image1.jpg)
![Remote Image 2](http://example.com/image2.png)
"""
        encoded_markdown_content = base64.b64encode(markdown_content.encode('utf-8')).decode('utf-8')

        mock_file_content = MagicMock()
        mock_file_content.name = "max_images_test.md"
        mock_file_content.path = "max_images_test.md"
        mock_file_content.type = "file"
        mock_file_content.content = encoded_markdown_content
        mock_file_content.html_url = "http://example.com/max_images_test.md"
        mock_file_content.sha = "67890"
        mock_file_content.size = len(markdown_content)
        
        self.mock_repo.get_contents.return_value = [mock_file_content]

        # Mock requests.get for remote images
        mock_remote_image1_bytes = b"remote_image1_bytes"
        # mock_remote_image2_bytes = b"remote_image2_bytes" # This one should not be called

        mock_response1 = MagicMock()
        mock_response1.status_code = 200
        mock_response1.content = mock_remote_image1_bytes
        
        # mock_response2 = MagicMock() # Not needed if not called
        # mock_response2.status_code = 200
        # mock_response2.content = mock_remote_image2_bytes
        
        # requests.get will be called for image1.jpg, but not for image2.png if max_images is 1 and base64 is processed first.
        # If base64 is first, only it will be processed. Let's ensure the test reflects that.
        # The base64 image will be processed first.
        
        mock_requests_get = MagicMock() # No remote images should be fetched if max_images=1 and base64 is first

        documents = []
        with patch('requests.get', mock_requests_get), \
             patch('agents.document_loader.src.loaders.github_loader.logger') as mock_logger:
            # We need to simulate that validate_connection was successful
            self.loader.repo = self.mock_repo 
            async for doc in self.loader.load():
                documents.append(doc)

        self.assertEqual(len(documents), 1)
        doc = documents[0]

        self.assertEqual(len(doc.images), max_images)
        if max_images > 0:
            self.assertEqual(doc.images[0], SAMPLE_B64_IMAGE_BYTES) # Base64 image should be first

        # Check that requests.get was NOT called because the limit was met by the b64 image
        mock_requests_get.assert_not_called()
        
        # Check for the log message
        expected_log_message = f"Reached max images ({max_images}) for Markdown {mock_file_content.name}, stopping image extraction."
        
        # Verify logger.info was called with the expected message
        called_logs = False
        for log_call in mock_logger.info.call_args_list:
            if expected_log_message in log_call[0][0]:
                called_logs = True
                break
        self.assertTrue(called_logs, f"Expected log message not found: {expected_log_message}")


    # TODO: Add tests for PDF extraction, text file extraction, MAX_IMAGES_PER_DOC, error handling etc.

if __name__ == '__main__':
    unittest.main() 