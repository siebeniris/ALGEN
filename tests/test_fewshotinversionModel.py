import unittest
import torch
from src.models.config import InversionConfig
from src.models.fewShotInversion import FewShotInversionModel


# python -m unittest tests/test_fewshotinversionModel.py


class TestFewShotInversionModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = InversionConfig(
            embedder_model_api=None,
            encoder_decoder_name_or_path="google/mt5-small",
            use_lora=False,
            max_seq_length=128,
            embedder_model_name="me5",
            aligning_strategy="svd",
            embedder_no_grad=True,
            num_repeat_tokens=16
        )

        cls.model = FewShotInversionModel(config=cls.config)
        cls.num_repeat_tokens =16

        cls.batch_size = 2
        cls.seq_length = 10
        cls.input_ids = torch.randint(0, 30522, (cls.batch_size, cls.seq_length))  # Random input IDs
        cls.attention_mask = torch.ones((cls.batch_size, cls.seq_length), dtype=torch.long)
        cls.labels = torch.randint(0, 30522, (cls.batch_size, cls.seq_length))  # Random labels

    def test_initialization(self):
        # Check if the model and components are correctly initialized
        self.assertIsNotNone(self.model.embedder, "Embedder should be initialized")
        self.assertIsNotNone(self.model.embedder_tokenizer, "Embedder tokenizer should be initialized")
        self.assertIsNotNone(self.model.encoder_decoder, "Encoder-decoder should be initialized")
        self.assertIsNotNone(self.model.tokenizer, "Encoder-decoder Tokenizer should be initialized")
        if self.model.embedder_dim != self.model.encoder_hidden_dim:
            self.assertIsInstance(self.model.linear_aligner, torch.nn.Linear, "Linear aligner should be initialized")

    def test_embeddings_dimension(self):
        model_output = self.model.embedder(input_ids=self.input_ids, attention_mask=self.attention_mask)
        embedder_embeddings = self.model._process_embeddings(model_output, self.attention_mask)
        self.assertEqual(embedder_embeddings.shape, (self.batch_size, self.model.embedder_dim))

    def test_embedding_alignment(self):
        # Test the alignment of embeddings
        aligned_embeddings, aligned_attention_mask = self.model.align_embeddings(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask
        )
        self.assertEqual(aligned_embeddings.shape, (self.batch_size, self.num_repeat_tokens, self.model.encoder_hidden_dim))
        self.assertEqual(aligned_attention_mask.shape, (self.batch_size, self.num_repeat_tokens))

    def test_forward(self):
        # Test the forward method with labels
        outputs = self.model.forward(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            labels=self.labels
        )
        self.assertIn("loss", outputs, "Output should contain 'loss' when labels are provided")
        self.assertIn("logits", outputs, "Output should contain 'logits'")

        # Test the forward method without labels
        outputs_no_labels = self.model.forward(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            decoder_input_ids=self.input_ids
        )
        self.assertNotIn("loss", outputs_no_labels, "Output should not contain 'loss' when labels are not provided")
        self.assertIn("logits", outputs_no_labels, "Output should contain 'logits'")

    def test_generate(self):
        # Test the generation method for expected behavior
        generation_kwargs = {
            "max_length": 20,
            "num_return_sequences": 1,
        }
        generated = self.model.generate(
            inputs={"embedder_input_ids": self.input_ids, "embedder_attention_mask": self.attention_mask},
            generation_kwargs=generation_kwargs
        )
        self.assertIsInstance(generated, torch.Tensor, "Generated output should be a tensor")
        self.assertEqual(generated.shape[0], self.batch_size * generation_kwargs["num_return_sequences"])


if __name__ == '__main__':
    unittest.main()
