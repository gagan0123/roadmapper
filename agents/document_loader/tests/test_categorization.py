#!/usr/bin/env python3
"""
Test suite for the document categorization feature.
This test validates that categories are properly assigned to documents from all source types.
"""

import unittest
from unittest.mock import patch, MagicMock
import asyncio
from agents.document_loader.src.loaders.github_loader import GitHubLoader
from agents.document_loader.src.loaders.web_loader import WebLoader
from agents.document_loader.src.data_source_loader import DataSourceLoader


class TestDocumentCategorization(unittest.IsolatedAsyncioTestCase):
    """Test cases for document categorization functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.data_source_loader = DataSourceLoader()

    @patch('agents.document_loader.src.loaders.web_loader.aiohttp')
    async def test_web_loader_category_assignment(self, mock_aiohttp):
        """Test that WebLoader correctly assigns categories to documents."""
        # Mock the HTTP response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.text = MagicMock()
        mock_response.text.return_value = asyncio.Future()
        mock_response.text.return_value.set_result("""
        <!DOCTYPE html>
        <html>
        <head><title>Test Page</title></head>
        <body><h1>Test Content</h1><p>This is test content.</p></body>
        </html>
        """)
        mock_response.headers = {'content-type': 'text/html'}
        
        # Mock the session
        mock_session = MagicMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        mock_session.close = MagicMock()
        mock_session.close.return_value = asyncio.Future()
        mock_session.close.return_value.set_result(None)
        
        # Mock ClientSession
        mock_aiohttp.ClientSession.return_value = mock_session

        # Test with category
        test_category = "Test Category"
        web_loader = WebLoader(urls=["https://example.com"], category=test_category)
        
        # Set the session directly
        web_loader.session = mock_session

        documents = []
        async for doc in web_loader.load():
            documents.append(doc)

        # Verify category assignment
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].category, test_category)
        self.assertEqual(documents[0].source_type, 'web')

    @patch('agents.document_loader.src.loaders.github_loader.Github')
    async def test_github_loader_category_assignment(self, mock_github_class):
        """Test that GitHubLoader correctly assigns categories to documents."""
        # Mock GitHub API responses
        mock_github = MagicMock()
        mock_repo = MagicMock()
        mock_github.get_repo.return_value = mock_repo
        mock_repo.default_branch = "main"
        mock_repo.full_name = "test/repo"
        
        # Mock file content with proper string values
        mock_file = MagicMock()
        mock_file.type = "file"
        mock_file.name = "README.md"
        mock_file.path = "README.md"
        mock_file.content = "VGVzdCBjb250ZW50"  # base64 encoded "Test content"
        mock_file.sha = "abc123def456"  # Mock SHA string
        mock_file.html_url = "https://github.com/test/repo/blob/main/README.md"  # Mock HTML URL
        mock_file.size = 100  # Mock size
        
        mock_repo.get_contents.return_value = [mock_file]
        mock_github_class.return_value = mock_github

        # Test with category
        test_category = "Documentation"
        github_loader = GitHubLoader(repo_name="test/repo", category=test_category, token="fake_token")
        
        # Set the github and repo objects directly to bypass validation
        github_loader.github = mock_github
        github_loader.repo = mock_repo

        documents = []
        async for doc in github_loader.load():
            documents.append(doc)

        # Verify category assignment
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].category, test_category)
        self.assertEqual(documents[0].source_type, 'github')

    def test_data_source_loader_parsing(self):
        """Test that DataSourceLoader correctly parses categories from data source files."""
        # Test parsing with categories and model types
        test_cases = [
            ("item1 Category1 text", ("item1", "Category1", "text")),
            ('item2 "Category with spaces" multimodal', ("item2", "Category with spaces", "multimodal")),
            ("item3 CategoryOnly", ("item3", "CategoryOnly", "multimodal")), # Default model type
            ("item4", ("item4", None, "multimodal")), # Default model type
            ("", (None, None, None)), # Corrected: Empty line yields no data
            ("# comment", (None, None, None)), # Corrected: Comment line yields no data
            ("item5 Category2 unknown_model_type", ("item5", "Category2", "multimodal")), # Invalid model falls back to default
        ]

        for line, expected in test_cases:
            result = self.data_source_loader._parse_line(line)
            self.assertEqual(result, expected, f"Failed parsing line: {line}")

    def test_integration_data_source_loading(self):
        """Test loading actual data sources with categories."""
        # Load actual data sources
        drive_folders = self.data_source_loader.load_drive_folders()
        github_repos = self.data_source_loader.load_github_repos()
        web_urls = self.data_source_loader.load_web_urls()

        # Verify structure: each should be a list of tuples (item, category, embedding_model_type)
        for source_list in [drive_folders, github_repos, web_urls]:
            self.assertIsInstance(source_list, list)
            for item_tuple in source_list: # Renamed 'item' to 'item_tuple' for clarity
                self.assertIsInstance(item_tuple, tuple)
                self.assertEqual(len(item_tuple), 3)
                # First element should be string (the item)
                self.assertIsInstance(item_tuple[0], str)
                # Second should be string or None (category)
                self.assertTrue(isinstance(item_tuple[1], (str, type(None))))
                # Third should be string (embedding_model_type)
                self.assertIsInstance(item_tuple[2], str)
                self.assertIn(item_tuple[2], ["text", "multimodal"])


        print(f"✓ Loaded {len(drive_folders)} Drive folders with categories and model types")
        print(f"✓ Loaded {len(github_repos)} GitHub repos with categories and model types")
        print(f"✓ Loaded {len(web_urls)} Web URLs with categories and model types")

    async def test_end_to_end_categorization_demo(self):
        """
        Demonstrate end-to-end categorization functionality.
        This test serves as both validation and documentation.
        """
        print("\n=== Document Categorization Feature Demonstration ===")
        
        # Load data sources
        github_repos = self.data_source_loader.load_github_repos()
        web_urls = self.data_source_loader.load_web_urls()
        
        print(f"\nData sources loaded:")
        print(f"GitHub repos: {github_repos}")
        print(f"Web URLs: {web_urls}")
        
        all_documents = []
        
        # Process a subset to avoid long test times
        if github_repos:
            # Unpack all three values now
            repo_name, category, model_type = github_repos[0]  # Test first repo only
            print(f"\nTesting GitHub repository: {repo_name} (Category: {category or 'N/A'}, Model: {model_type})")
            try:
                # Pass model_type to the loader
                github_loader = GitHubLoader(repo_name=repo_name, category=category, embedding_model_type=model_type)
                if await github_loader.validate_connection():
                    doc_count = 0
                    async for doc in github_loader.load():
                        all_documents.append(doc)
                        doc_count += 1
                        print(f"  ✓ Loaded: '{doc.title}' → Category: '{doc.category}'")
                        if doc_count >= 2:  # Limit for test performance
                            break
                else:
                    print(f"  ⚠ Could not connect to {repo_name}")
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        if web_urls:
            # Unpack all three values now
            url, category, model_type = web_urls[0]  # Test first URL only
            print(f"\nTesting Web URL: {url} (Category: {category or 'N/A'}, Model: {model_type})")
            try:
                # Pass model_type to the loader
                web_loader = WebLoader(urls=[url], category=category, embedding_model_type=model_type)
                if await web_loader.validate_connection():
                    async for doc in web_loader.load():
                        all_documents.append(doc)
                        print(f"  ✓ Loaded: '{doc.title}' → Category: '{doc.category}'")
                        break  # Just test one for performance
                else:
                    print(f"  ⚠ Could not connect to {url}")
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        # Verify categories are properly assigned
        print(f"\n=== Results ===")
        print(f"Total documents loaded: {len(all_documents)}")
        
        # Group by category
        categories = {}
        for doc in all_documents:
            cat = doc.category or "No Category"
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(doc.title)
        
        print("\nDocuments by category:")
        for category, titles in categories.items():
            print(f"  📁 {category}: {len(titles)} document(s)")
            for title in titles[:3]:  # Show max 3 for readability
                print(f"    • {title}")
        
        # Assert that if we loaded documents, they have the expected categories
        for doc in all_documents:
            self.assertIsNotNone(doc.category, f"Document '{doc.title}' should have a category")
            self.assertIsInstance(doc.category, str, f"Category should be a string")
            self.assertNotEqual(doc.category.strip(), "", f"Category should not be empty")


if __name__ == "__main__":
    unittest.main() 