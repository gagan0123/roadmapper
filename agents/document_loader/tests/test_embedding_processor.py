import asyncio
import sys
import os
import base64
import unittest
from unittest.mock import patch, MagicMock

# Adjust path to import from src
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agents.document_loader.src.models.base import Document, Embedding
from agents.document_loader.src.processors.embedding_processor import EmbeddingProcessor, TEXT_CHAR_LIMIT_FOR_MULTIMODAL

# Dummy image data
DUMMY_IMAGE_BYTES = base64.b64decode(
    b'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII='
)
OVERSIZED_DUMMY_IMAGE_BYTES = b'A' * (20 * 1024 * 1024 + 1) # 20MB + 1 byte

# Define a consistent embedding dimension for mock responses
MOCK_EMBEDDING_DIMENSION = 768 # Updated to match multimodal model

def create_mock_prediction(num_vectors=1, dimension=MOCK_EMBEDDING_DIMENSION):
    mock_predictions = []
    for _ in range(num_vectors):
        mock_predictions.append({'values': [0.1] * dimension})
    return mock_predictions

class TestEmbeddingProcessor(unittest.IsolatedAsyncioTestCase):

    @patch('agents.document_loader.src.processors.embedding_processor.aiplatform_v1.PredictionServiceClient')
    async def asyncSetUp(self, MockPredictionServiceClient):
        self.mock_vertex_client = MockPredictionServiceClient.return_value
        
        # Standard mock predict function for successful initialization and most tests
        def default_mock_predict(endpoint, instances):
            response_obj = MagicMock()
            # Ensure predictions is a list of dicts, even if it's just one prediction
            # It should be [{ 'embedding': [...] }] or [{ 'values': [...] }]
            # Based on multimodalembedding@001, it expects a list of dicts with an "embedding" key.
            # Let's align create_mock_prediction with this.
            raw_preds = []
            for _ in range(len(instances)):
                raw_preds.append({'embedding': [0.1] * MOCK_EMBEDDING_DIMENSION })
            response_obj.predictions = raw_preds

            TestEmbeddingProcessor.last_instances_called_for_predict = instances
            return response_obj
        
        # Set this default behavior for the client's predict method initially
        self.mock_vertex_client.predict.side_effect = default_mock_predict

        self.processor = EmbeddingProcessor(
            project_id="test-project",
            location="test-location",
            verbose=False,
            batch_size=3 # Use a smaller batch_size for some tests like batch_limit
        )
        # initialize_vertex_ai is called during EmbeddingProcessor init
        # It makes one call to predict. We assert it here and then reset.
        self.mock_vertex_client.predict.assert_called_once()
        self.mock_vertex_client.predict.reset_mock() # Reset for subsequent test-specific calls
        # Crucially, reset the side_effect too if it's not meant to be global for all tests
        # Or, ensure each test sets its required side_effect.
        # For now, the default_mock_predict is fine as a general side_effect unless a test overrides it.
        TestEmbeddingProcessor.last_instances_called_for_predict = None

    async def test_text_only_document(self):
        doc_id = "text_doc_01"
        test_category = "TestCategory"
        long_text = "This is a long piece of text. " * 100 # Create a long text
        # Text-only instances from Multimodal processor are truncated by char limit for multimodal
        # after initial token truncation.
        # First, truncate to max_tokens_per_text_instance (though this long_text might not hit it)
        token_truncated_text = self.processor._truncate_text(long_text, self.processor.max_tokens_per_text_instance)
        # Then, apply the character limit specific to multimodal text instances
        expected_final_text = token_truncated_text[:TEXT_CHAR_LIMIT_FOR_MULTIMODAL]

        doc = Document(id=doc_id, title="Text Only Doc", content=long_text, source_type="test", source_url="test/text_only", category=test_category)
        
        embeddings = await self.processor.generate_embeddings([doc])

        self.assertEqual(len(embeddings), 1)
        self.mock_vertex_client.predict.assert_called_once()
        
        self.assertEqual(len(TestEmbeddingProcessor.last_instances_called_for_predict), 1)
        instance_sent = TestEmbeddingProcessor.last_instances_called_for_predict[0]
        # The instance sent should be the character-limited text
        self.assertEqual(instance_sent.get("text"), expected_final_text)
        self.assertIsNone(instance_sent.get("image"))

        embedding = embeddings[0]
        self.assertEqual(embedding.document_id, f"{doc_id}::text")
        self.assertEqual(embedding.category, test_category)
        self.assertEqual(embedding.metadata['original_doc_id'], doc_id)
        self.assertEqual(embedding.metadata['instance_type'], 'text_only')
        self.assertEqual(embedding.metadata['category'], test_category)
        self.assertIsNone(embedding.metadata['image_index'])
        self.assertEqual(len(embedding.embedding), MOCK_EMBEDDING_DIMENSION)

    async def test_image_only_document(self):
        doc_id = "img_doc_01"
        test_category = "ImageDocs"
        # Document with no text content, only an image
        doc = Document(id=doc_id, title="Image Only Doc", content="", images=[DUMMY_IMAGE_BYTES], source_type="test", source_url="test/image_only", category=test_category)

        embeddings = await self.processor.generate_embeddings([doc])
        
        # Expected: 1 embedding for the image
        self.assertEqual(len(embeddings), 1)
        self.mock_vertex_client.predict.assert_called_once()

        self.assertEqual(len(TestEmbeddingProcessor.last_instances_called_for_predict), 1)
        instance_sent = TestEmbeddingProcessor.last_instances_called_for_predict[0]
        self.assertIsNone(instance_sent.get("text")) # No text for image_only if original content is empty
        self.assertIsNotNone(instance_sent.get("image"))
        self.assertEqual(instance_sent["image"].get("bytesBase64Encoded"), base64.b64encode(DUMMY_IMAGE_BYTES).decode('utf-8'))

        embedding = embeddings[0]
        self.assertEqual(embedding.document_id, f"{doc_id}::image_0")
        self.assertEqual(embedding.category, test_category)
        self.assertEqual(embedding.metadata['original_doc_id'], doc_id)
        self.assertEqual(embedding.metadata['instance_type'], 'image_only_0')
        self.assertEqual(embedding.metadata['category'], test_category)
        self.assertEqual(embedding.metadata['image_index'], 0)

    async def test_text_and_one_image_document(self):
        doc_id = "text_img_doc_01"
        test_category = "MultimodalDoc"
        text_content = "Associated text for an image. " * 5
        expected_truncated_full_text = self.processor._truncate_text(text_content, self.processor.max_tokens_per_text_instance)
        expected_truncated_multimodal_text = self.processor._truncate_text(text_content, self.processor.max_tokens_for_multimodal_text)
        
        doc = Document(id=doc_id, title="Text & Image Doc", content=text_content, images=[DUMMY_IMAGE_BYTES], source_type="test", source_url="test/text_image", category=test_category)

        embeddings = await self.processor.generate_embeddings([doc])

        # Expected: 1 for full text, 1 for image = 2 embeddings
        self.assertEqual(len(embeddings), 2)
        self.mock_vertex_client.predict.assert_called_once()
        
        self.assertEqual(len(TestEmbeddingProcessor.last_instances_called_for_predict), 2)
        
        # Instance 1: Text Only
        text_instance = next(inst for inst in TestEmbeddingProcessor.last_instances_called_for_predict if "text" in inst and "image" not in inst)
        self.assertEqual(text_instance.get("text"), expected_truncated_full_text)

        # Instance 2: Text + Image
        image_instance = next(inst for inst in TestEmbeddingProcessor.last_instances_called_for_predict if "image" in inst)
        self.assertEqual(image_instance.get("text"), expected_truncated_multimodal_text)
        self.assertEqual(image_instance["image"].get("bytesBase64Encoded"), base64.b64encode(DUMMY_IMAGE_BYTES).decode('utf-8'))

        # Embedding 1: Text
        text_embedding = next(e for e in embeddings if e.document_id == f"{doc_id}::text")
        self.assertEqual(text_embedding.category, test_category)
        self.assertEqual(text_embedding.metadata['original_doc_id'], doc_id)
        self.assertEqual(text_embedding.metadata['instance_type'], 'text_only')

        # Embedding 2: Image
        image_embedding = next(e for e in embeddings if e.document_id == f"{doc_id}::image_0")
        self.assertEqual(image_embedding.category, test_category)
        self.assertEqual(image_embedding.metadata['original_doc_id'], doc_id)
        self.assertEqual(image_embedding.metadata['instance_type'], 'text_image_0')
        self.assertEqual(image_embedding.metadata['image_index'], 0)

    async def test_text_and_multiple_images(self):
        doc_id = "text_multi_img_doc_01"
        test_category = "ManyImages"
        text_content = "Text with two images. " * 3
        dummy_image_2_bytes = base64.b64decode(b'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==')
        images = [DUMMY_IMAGE_BYTES, dummy_image_2_bytes]

        doc = Document(id=doc_id, title="Text & Multi-Images", content=text_content, images=images, source_type="test", source_url="test/text_multi_image", category=test_category)
        embeddings = await self.processor.generate_embeddings([doc])

        # Expected: 1 for text + 2 for images = 3 embeddings
        self.assertEqual(len(embeddings), 1 + len(images))
        self.mock_vertex_client.predict.assert_called_once()
        self.assertEqual(len(TestEmbeddingProcessor.last_instances_called_for_predict), 1 + len(images))

        # Check text embedding
        text_embedding = next(e for e in embeddings if e.metadata['instance_type'] == 'text_only')
        self.assertEqual(text_embedding.document_id, f"{doc_id}::text")
        self.assertEqual(text_embedding.category, test_category)

        # Check image embeddings
        for i in range(len(images)):
            img_embedding = next(e for e in embeddings if e.metadata.get('image_index') == i)
            self.assertEqual(img_embedding.document_id, f"{doc_id}::image_{i}")
            self.assertEqual(img_embedding.category, test_category)
            self.assertTrue(img_embedding.metadata['instance_type'].startswith(('text_image_', 'image_only_')))

    async def test_batch_processing_mixed_documents(self):
        # Processor batch_size is 3 (from asyncSetUp)
        docs = []
        default_category = "BatchTest"

        # Doc1: Text-only (1 instance)
        doc1_id = "batch_text_01"
        docs.append(Document(id=doc1_id, title="Batch Doc 1", content="Text for batch 1", source_type="batch", source_url="b/1", category=default_category))

        # Doc2: Image-only (1 instance)
        doc2_id = "batch_img_01"
        docs.append(Document(id=doc2_id, title="Batch Doc 2", content="", images=[DUMMY_IMAGE_BYTES], source_type="batch", source_url="b/2", category=default_category))

        # Doc3: Text and 2 images (1 text instance + 2 image instances = 3 instances)
        doc3_id = "batch_text_multi_img_01"
        dummy_image_2 = base64.b64decode(b'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==')
        doc3_images = [DUMMY_IMAGE_BYTES, dummy_image_2]
        docs.append(Document(id=doc3_id, title="Batch Doc 3", content="Text for doc 3 with images", images=doc3_images, source_type="batch", source_url="b/3", category=default_category))
        
        # Doc4: Text and 1 OVERSIZED image (1 text instance, image is skipped by _prepare_batch)
        doc4_id = "batch_text_oversized_img_01"
        doc4_content = "Text for doc 4 with oversized image"
        # OVERSIZED_DUMMY_IMAGE_BYTES is defined globally
        docs.append(Document(id=doc4_id, title="Batch Doc 4", content=doc4_content, images=[OVERSIZED_DUMMY_IMAGE_BYTES], source_type="batch", source_url="b/4", category=default_category))

        # Expected instances: Doc1 (1), Doc2 (1), Doc3 (3), Doc4 (1 for text) = 6 total instances
        # Expected embeddings = 6
        # Expected API calls with batch_size=3: ceil(6/3) = 2 if perfectly packed, but due to doc boundaries, it's 3.
        # Call 1: Doc1_text, Doc2_img, Doc3_text (3 instances)
        # Call 2: Doc3_img0, Doc3_img1, Doc4_text (3 instances)
        # Oh, wait, if Doc3_text fills the first batch, Doc3's images will be in the next batch with Doc4_text.
        # Let's trace _prepare_batch carefully for consumption from documents_queue:
        # Batch 1: 
        #   - Doc1 (1 inst): added, Doc1 popped. queue=[D2,D3,D4]. consumed_batch=1
        #   - Doc2 (1 inst): added, Doc2 popped. queue=[D3,D4]. consumed_batch=2
        #   - Doc3 (text part, 1 inst): added. batch full. Doc3 NOT popped. queue=[D3,D4]. consumed_batch=2
        #   Returns 3 instances. API call 1.
        # Batch 2: 
        #   - Doc3 (remaining image parts, 2 inst): added. Doc3 popped. queue=[D4]. consumed_batch=1 (for this _prepare_batch call)
        #   - Doc4 (text part, 1 inst): added. batch full. Doc4 popped. queue=[]. consumed_batch=2
        #   Returns 3 instances. API call 2.
        # This makes 2 API calls. Total instances = 3+3=6.

        with self.assertLogs(level='WARNING') as log:
            embeddings = await self.processor.generate_embeddings(docs)
            # Check if the oversized image warning for Doc4 was logged
            self.assertTrue(any(f"Image 0 for document {doc4_id} exceeds size limit" in msg for msg in log.output),
                            f"Oversized image warning for {doc4_id} not found. Logs: {log.output}")

        self.assertEqual(len(embeddings), 6)
        # With the current _prepare_batch logic:
        # Call 1: D1_text, D2_img (2 instances, batch not full yet, D3 is next)
        #   _prepare_batch for D3 (3 instances) finds 2 + 3 > 3, so D3 is not added. Batch ends.
        # Call 2: D3_text, D3_img0, D3_img1 (3 instances from D3). Batch full.
        # Call 3: D4_text (1 instance from D4).
        # This results in 3 API calls.
        self.assertEqual(self.mock_vertex_client.predict.call_count, 3)
        
        total_instances_sent_to_api = 0
        for call_args in self.mock_vertex_client.predict.call_args_list:
            # instances is always passed as a keyword argument in the processor
            total_instances_sent_to_api += len(call_args.kwargs['instances'])
        self.assertEqual(total_instances_sent_to_api, 6)

        # Verify specific embeddings
        self.assertIsNotNone(next((e for e in embeddings if e.document_id == f"{doc1_id}::text"), None))
        self.assertIsNotNone(next((e for e in embeddings if e.document_id == f"{doc2_id}::image_0"), None))
        self.assertIsNotNone(next((e for e in embeddings if e.document_id == f"{doc3_id}::text"), None))
        self.assertIsNotNone(next((e for e in embeddings if e.document_id == f"{doc3_id}::image_0"), None))
        self.assertIsNotNone(next((e for e in embeddings if e.document_id == f"{doc3_id}::image_1"), None))
        self.assertIsNotNone(next((e for e in embeddings if e.document_id == f"{doc4_id}::text"), None), "Text from Doc4 should be processed")
        # Ensure embedding for oversized image from Doc4 was NOT created
        self.assertIsNone(next((e for e in embeddings if e.document_id == f"{doc4_id}::image_0"), None), "Embedding for oversized image in Doc4 should not exist")

    async def test_api_error_handling(self):
        # Set the specific side effect for this test AFTER processor initialization
        self.mock_vertex_client.predict.side_effect = Exception("Simulated API Error")
        
        doc = Document(id="error_doc", title="Error Test", content="Some text", source_type="test", source_url="test/error", category="ErrorCategory")
        
        with self.assertLogs(level='ERROR') as log:
            embeddings = await self.processor.generate_embeddings([doc])
            # Check for the specific error message from generate_embeddings after retries
            expected_log_part = f"Failed to process a batch after multiple retries: Simulated API Error. Associated doc IDs: ['{doc.id}']"
            self.assertTrue(any(expected_log_part in message for message in log.output),
                            f"Expected log message not found. Logs: {log.output}")
        
        self.assertEqual(len(embeddings), 0)
        # Retry logic in generate_embeddings calls _retry_with_backoff, which calls _process_batch.
        # _process_batch calls self.client.predict. This will be called self.processor.max_retries times.
        self.assertEqual(self.mock_vertex_client.predict.call_count, self.processor.max_retries)

    async def test_oversized_image_only_document_no_text(self):
        """Test a document that only has an oversized image and no text."""
        doc_id = "oversized_only_img_doc"
        doc = Document(id=doc_id, title="Oversized Only", content="", images=[OVERSIZED_DUMMY_IMAGE_BYTES], source_type="test", source_url="test/oversized_only", category="Oversized")
        
        with self.assertLogs(level='WARNING') as log:
            embeddings = await self.processor.generate_embeddings([doc])
            self.assertTrue(any(f"Image 0 for document {doc_id} exceeds size limit" in msg for msg in log.output))
            # Also check for the "yielded no processable instances" log if it's the only content
            self.assertTrue(any(f"Document {doc_id} yielded no processable instances" in msg for msg in log.output))
            
        self.assertEqual(len(embeddings), 0) # No embeddings should be generated
        self.mock_vertex_client.predict.assert_not_called() # No API call if no instances

    async def test_oversized_image_with_text_document(self):
        """Test a document with text and an oversized image. Text should still be processed."""
        doc_id = "text_oversized_img_doc"
        text_content = "This document has text and an oversized image."
        doc = Document(id=doc_id, title="Text Oversized", content=text_content, images=[OVERSIZED_DUMMY_IMAGE_BYTES], source_type="test", source_url="test/text_oversized", category="TextWithBigImage")

        with self.assertLogs(level='WARNING') as log:
            embeddings = await self.processor.generate_embeddings([doc])
            self.assertTrue(any(f"Image 0 for document {doc_id} exceeds size limit" in msg for msg in log.output))

        self.assertEqual(len(embeddings), 1) # Only one embedding for the text part
        self.mock_vertex_client.predict.assert_called_once()
        self.assertEqual(len(TestEmbeddingProcessor.last_instances_called_for_predict), 1)
        instance_sent = TestEmbeddingProcessor.last_instances_called_for_predict[0]
        self.assertEqual(instance_sent.get("text"), text_content)
        self.assertIsNone(instance_sent.get("image"))

        embedding = embeddings[0]
        self.assertEqual(embedding.document_id, f"{doc_id}::text")
        self.assertEqual(embedding.category, "TextWithBigImage")
        self.assertEqual(embedding.metadata['original_doc_id'], doc_id)
        self.assertEqual(embedding.metadata['instance_type'], 'text_only')

    async def test_empty_document_list(self):
        embeddings = await self.processor.generate_embeddings([])
        self.assertEqual(len(embeddings), 0)
        self.mock_vertex_client.predict.assert_not_called()

    async def test_document_with_no_content_or_images(self):
        doc_id = "empty_doc"
        # Document with no text and no images
        doc = Document(id=doc_id, title="Empty Doc", content="    ", images=[], source_type="test", source_url="test/empty", category="EmptyTest")
        
        with self.assertLogs(level='WARNING') as log:
            embeddings = await self.processor.generate_embeddings([doc])
            self.assertTrue(any(f"Document {doc_id} yielded no processable instances" in msg for msg in log.output) or 
                            any(f"Document {doc_id} has no content/images or already processed" in msg for msg in log.output) )

        self.assertEqual(len(embeddings), 0)
        self.mock_vertex_client.predict.assert_not_called()
        
    async def test_document_with_multiple_oversized_images_and_text(self):
        doc_id = "text_multi_oversized_doc"
        text_content = "Text with multiple oversized images."
        images = [OVERSIZED_DUMMY_IMAGE_BYTES, OVERSIZED_DUMMY_IMAGE_BYTES]
        doc = Document(id=doc_id, title="Text Multi Oversized", content=text_content, images=images, source_type="test", source_url="test/multi_oversized", category="MultiOversized")

        with self.assertLogs(level='WARNING') as log:
            embeddings = await self.processor.generate_embeddings([doc])
            self.assertTrue(any(f"Image 0 for document {doc_id} exceeds size limit" in msg for msg in log.output))
            self.assertTrue(any(f"Image 1 for document {doc_id} exceeds size limit" in msg for msg in log.output))

        self.assertEqual(len(embeddings), 1) # Only text embedding
        self.mock_vertex_client.predict.assert_called_once()
        text_embedding = embeddings[0]
        self.assertEqual(text_embedding.document_id, f"{doc_id}::text")
        self.assertEqual(text_embedding.category, "MultiOversized")

    async def test_document_batch_limit_behavior(self):
        """Test how _prepare_batch handles a single document whose parts exceed api_batch_size."""
        # batch_size is 3 from asyncSetUp
        self.assertEqual(self.processor.api_batch_size, 3)

        doc_id = "batch_limit_doc_01"
        # Create a document that will yield more instances than batch_size (3)
        # e.g., Text (1) + 5 images (5) = 6 instances
        text_content = "Document for batch limit test."
        images_for_doc = [DUMMY_IMAGE_BYTES] * 5 # 5 images

        doc1 = Document(
            id=doc_id, 
            title="Batch Limit Test Doc", 
            content=text_content, 
            images=images_for_doc, 
            source_type="test", 
            source_url="test/batch_limit", 
            category="BatchLimitCat"
        )

        with self.assertLogs(level='WARNING') as log:
            embeddings = await self.processor.generate_embeddings([doc1])
        
        # Check for the specific warning about dropping parts of a large document
        expected_log_msg_part = f"Document {doc_id} with {1 + len(images_for_doc)} parts exceeds api_batch_size of {self.processor.api_batch_size}. Processing only the first {self.processor.api_batch_size} parts."
        self.assertTrue(any(expected_log_msg_part in message for message in log.output),
                        f"Expected log message not found. Logs: {log.output}")

        # Only the first api_batch_size (3) instances should be processed and yield embeddings
        self.assertEqual(len(embeddings), self.processor.api_batch_size)
        self.mock_vertex_client.predict.assert_called_once() # Should only be one API call

        # Verify the instances sent to the API
        self.assertEqual(len(TestEmbeddingProcessor.last_instances_called_for_predict), self.processor.api_batch_size)
        
        # Check that the instances are: text, image0, image1 (if batch_size is 3)
        sent_instances = TestEmbeddingProcessor.last_instances_called_for_predict
        self.assertTrue("text" in sent_instances[0]) # First instance is text
        self.assertTrue("image" in sent_instances[1]) # Second is an image
        self.assertTrue("image" in sent_instances[2]) # Third is an image

        # Verify embedding IDs correspond to the processed parts
        self.assertIsNotNone(next((e for e in embeddings if e.document_id == f"{doc_id}::text"), None))
        self.assertIsNotNone(next((e for e in embeddings if e.document_id == f"{doc_id}::image_0"), None))
        self.assertIsNotNone(next((e for e in embeddings if e.document_id == f"{doc_id}::image_1"), None))
        # Ensure embeddings for dropped images (image_2, image_3, image_4) were NOT created
        self.assertIsNone(next((e for e in embeddings if e.document_id == f"{doc_id}::image_2"), None))
        self.assertIsNone(next((e for e in embeddings if e.document_id == f"{doc_id}::image_3"), None))
        self.assertIsNone(next((e for e in embeddings if e.document_id == f"{doc_id}::image_4"), None))

if __name__ == '__main__':
    unittest.main() 