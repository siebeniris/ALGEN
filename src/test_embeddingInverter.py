import unittest
import torch
from InversionModel import EmbeddingInverter


class TestEmbeddingInverter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize the model with default parameters
        cls.inverter = EmbeddingInverter(
            model_G_name_or_path="t5-base",
            model_S_name_or_path="t5-small",
            max_length=128,
            align_method="linear"
        )
        cls.test_texts = [
            "This is a test sentence.",
            "Another example for testing."
        ]

    def test_initialization(self):
        """Test if the model initializes correctly with different alignment methods."""
        align_methods = ["linear", "neural", "orthogonal", "ot"]

        for method in align_methods:
            print(f"Testing the align method {method}")
            inverter = EmbeddingInverter(
                model_G_name_or_path="t5-base",
                model_S_name_or_path="t5-small",
                align_method=method
            )
            self.assertIsNotNone(inverter.aligner)
            self.assertEqual(inverter.align_method, method)
            print("*"*50)

    def test_get_embeddings_S(self):
        """Test if black-box model generates embeddings correctly."""
        embeddings, input_ids, attention_mask = self.inverter.get_embeddings_S(self.test_texts)

        # Check shapes
        batch_size = len(self.test_texts)
        self.assertEqual(len(embeddings.shape), 3)  # [batch_size, seq_length, hidden_size]
        self.assertEqual(embeddings.shape[0], batch_size)
        self.assertEqual(embeddings.shape[2], self.inverter.hidden_size_S)

        # Check device
        self.assertEqual(str(embeddings.device), str(self.inverter.device))

        # Check if attention mask matches sequence length
        self.assertEqual(attention_mask.shape[1], embeddings.shape[1])

    def test_get_embeddings_G(self):
        """Test if teacher model generates embeddings correctly."""
        embeddings, input_ids, attention_mask = self.inverter.get_embeddings_G(self.test_texts)

        # Check shapes
        batch_size = len(self.test_texts)
        self.assertEqual(len(embeddings.shape), 3)
        self.assertEqual(embeddings.shape[0], batch_size)
        self.assertEqual(embeddings.shape[2], self.inverter.hidden_size_G)

        # Check device
        self.assertEqual(str(embeddings.device), str(self.inverter.device))

    def test_decode_embeddings(self):
        """Test if model can decode embeddings back to text."""
        # Get embeddings from teacher model
        embeddings, _, attention_mask = self.inverter.get_embeddings_G(self.test_texts)

        # Try decoding
        decoded_texts = self.inverter.decode_embeddings(embeddings, attention_mask)

        # Check if we get the expected number of outputs
        self.assertEqual(len(decoded_texts), len(self.test_texts))

        # Check if outputs are strings
        for text in decoded_texts:
            self.assertIsInstance(text, str)
            self.assertGreater(len(text), 0)

    def test_forward_pass(self):
        """Test the complete forward pass of the model."""
        aligned_embeddings, decoded_texts = self.inverter.forward(self.test_texts)

        # Check aligned embeddings shape
        self.assertEqual(aligned_embeddings.shape[2], self.inverter.hidden_size_G)

        # Check decoded texts
        self.assertEqual(len(decoded_texts), len(self.test_texts))
        for text in decoded_texts:
            self.assertIsInstance(text, str)

    def test_max_length_constraint(self):
        """Test if the model respects max_length constraint."""
        long_text = "This is a very long sentence that should be truncated. " * 10
        embeddings, input_ids, _ = self.inverter.get_embeddings_S([long_text])

        self.assertLessEqual(
            embeddings.shape[1],
            self.inverter.max_length,
            "Embedding sequence length exceeds max_length"
        )

    def test_batch_processing(self):
        """Test if the model can handle batched inputs."""
        batch_texts = ["Text " + str(i) for i in range(5)]
        aligned_embeddings, decoded_texts = self.inverter.forward(batch_texts)

        self.assertEqual(len(decoded_texts), len(batch_texts))
        self.assertEqual(aligned_embeddings.shape[0], len(batch_texts))

    def test_sanity_checks(self):
        """Test the sanity check methods."""
        try:
            self.inverter.sanity_check()
            self.inverter.sanity_check_random_embedding()
        except Exception as e:
            self.fail(f"Sanity checks failed with error: {str(e)}")

    def test_different_decoding_strategies(self):
        """Test different decoding strategies."""
        strategies = ["beam", "neucleus"]
        test_text = self.test_texts[0]

        for strategy in strategies:
            inverter = EmbeddingInverter(
                decoding_strategy=strategy
            )
            aligned_embeddings, decoded_texts = inverter.forward([test_text])
            self.assertIsInstance(decoded_texts[0], str)


if __name__ == '__main__':
    unittest.main()