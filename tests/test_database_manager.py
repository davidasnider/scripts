import importlib
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


class _DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))

    def decode(self, tokens):
        return " ".join(str(token) for token in tokens)


# Use a pytest fixture to patch the tokenizer and import database_manager
import types

@pytest.fixture(autouse=True, scope="module")
def patch_tokenizer_and_import_db_manager():
    with patch(
        "transformers.AutoTokenizer.from_pretrained", return_value=_DummyTokenizer()
    ):
        global database_manager, add_file_to_db, generate_embedding, initialize_db
        database_manager = importlib.import_module("src.database_manager")
        add_file_to_db = database_manager.add_file_to_db
        generate_embedding = database_manager.generate_embedding
        initialize_db = database_manager.initialize_db
        yield
class TestDatabaseManager(unittest.TestCase):
    @patch("src.database_manager.chromadb.PersistentClient")
    def test_initialize_db(self, mock_persistent_client):
        """Test that initialize_db returns a collection."""
        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_persistent_client.return_value = mock_client

        collection = initialize_db("some/path")

        self.assertEqual(collection, mock_collection)
        mock_persistent_client.assert_called_once_with(path="some/path")
        mock_client.get_or_create_collection.assert_called_once_with(
            name="digital_archive"
        )

    @patch("src.database_manager.ollama.embeddings")
    def test_generate_embedding_success(self, mock_ollama_embeddings):
        """Test successful embedding generation."""
        mock_ollama_embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}
        embedding = generate_embedding("some text")
        self.assertEqual(embedding, [0.1, 0.2, 0.3])

    @patch("src.database_manager.ollama.embeddings")
    def test_generate_embedding_failure(self, mock_ollama_embeddings):
        """Test embedding generation failure."""
        mock_ollama_embeddings.side_effect = Exception("Ollama broke")
        embedding = generate_embedding("some text")
        self.assertEqual(embedding, [0.0] * 768)

    @patch("src.database_manager.chunk_text")
    @patch("src.database_manager.ollama.embeddings")
    def test_generate_embedding_chunking(
        self, mock_ollama_embeddings, mock_chunk_text
    ):
        """Test embedding generation with chunking and averaging."""
        # Simulate two chunks
        mock_chunk_text.return_value = ["chunk1", "chunk2"]
        mock_ollama_embeddings.side_effect = [
            {"embedding": [0.1, 0.2, 0.3]},
            {"embedding": [0.3, 0.4, 0.5]},
        ]
        embedding = generate_embedding("some long text")
        expected_embedding = (
            np.mean([[0.1, 0.2, 0.3], [0.3, 0.4, 0.5]], axis=0).tolist()
        )
        self.assertEqual(embedding, expected_embedding)

    @patch("src.database_manager.generate_embedding")
    def test_add_file_to_db_with_text(self, mock_generate_embedding):
        """Test adding a file with text to the database."""
        mock_generate_embedding.return_value = [0.4, 0.5, 0.6]
        mock_collection = MagicMock()
        file_data = {
            "file_path": "a/b/c.txt",
            "file_name": "c.txt",
            "summary": "A summary",
            "description": "A description",
            "extracted_text": "Some text",
            "mentioned_people": ["John Doe"],
            "mime_type": "text/plain",
            "is_nsfw": True,
            "has_financial_red_flags": False,
        }

        add_file_to_db(file_data, mock_collection)

        mock_collection.add.assert_called_once()
        args, kwargs = mock_collection.add.call_args
        self.assertEqual(kwargs["ids"], ["a/b/c.txt"])
        self.assertEqual(kwargs["embeddings"], [[0.4, 0.5, 0.6]])

    @patch("src.database_manager.generate_embedding")
    def test_add_file_to_db_no_text(self, mock_generate_embedding):
        """Test adding a file with no text to the database."""
        mock_collection = MagicMock()
        file_data = {"file_path": "a/b/c.txt"}

        add_file_to_db(file_data, mock_collection)

        mock_generate_embedding.assert_not_called()
        mock_collection.add.assert_called_once()
        args, kwargs = mock_collection.add.call_args
        embedding = kwargs["embeddings"][0]
        self.assertEqual(len(embedding), DEFAULT_EMBEDDING_DIM)
        self.assertTrue(all(x == 0.0 for x in embedding))
        self.assertTrue(len(set(embedding)) <= 1)  # all values are the same
        self.assertGreater(len(embedding), 0)  # embedding is not empty
