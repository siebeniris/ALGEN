import unittest
import os
import shutil
import yaml
from trainerConfig import TrainerConfig, get_default_config, get_linear_config, get_neural_config, \
    get_orthogonal_config, get_ot_config
from train import save_config, load_config_from_yaml, train_with_config
from InversionTrainer import EmbeddingInverterTrainer
import torch


class TestTrainerConfig(unittest.TestCase):
    def test_get_default_config(self):
        config = get_default_config()
        self.assertIsInstance(config, TrainerConfig)

    def test_to_dict(self):
        config = TrainerConfig()  # Assuming this initializes a valid config
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)

    def test_get_linear_config(self):
        config = get_linear_config()
        self.assertIsInstance(config, TrainerConfig)

    def test_get_neural_config(self):
        config = get_neural_config()
        self.assertIsInstance(config, TrainerConfig)

    def test_get_ot_config(self):
        config = get_ot_config()
        self.assertIsInstance(config, TrainerConfig)


#
# class TestTrainFunctions(unittest.TestCase):
#     def setUp(self):
#         self.test_dir = "test_config_dir"
#         os.makedirs(self.test_dir, exist_ok=True)
#         self.test_config = {"param1": "value1", "param2": "value2"}
#         self.test_yaml_path = os.path.join(self.test_dir, "test_config.yaml")
#         with open(self.test_yaml_path, "w") as file:
#             yaml.dump(self.test_config, file)
#
#     def tearDown(self):
#         shutil.rmtree(self.test_dir)
#
#     def test_save_config(self):
#         save_config(self.test_config, self.test_dir)
#         config_path = os.path.join(self.test_dir, "config.yaml")
#         self.assertTrue(os.path.exists(config_path))
#
#     def test_load_config_from_yaml(self):
#         loaded_config = load_config_from_yaml(self.test_yaml_path)
#         self.assertEqual(loaded_config, self.test_config)
#
#     def test_train_with_config(self):
#         # Mocking train_with_config for functionality since no actual training is defined
#         try:
#             train_with_config(self.test_config)
#         except Exception as e:
#             self.fail(f"train_with_config raised an exception: {e}")


class TestInversionTrainer(unittest.TestCase):
    def setUp(self):
        self.trainer = EmbeddingInverterTrainer(
            model_G_name="t5-base",
            model_S_name="t5-small",
            save_dir="test_save_dir",
            checkpoint_path=None,
            resume_training=False,
            use_wandb=False,
            align_method="linear",
            learning_rate=0.001,
            batch_size=32,
            num_epochs=1,
            max_length=128,
            adjust_weights_with_magnitude=False,
            ot_reg=0.1,
            ot_reg_m=0.1,
            decoding_strategy="strategy",
            dataset_name="dataset",
            language_script="script",
            train_samples=100,
            eval_samples=10
        )

    # def tearDown(self):
    #     shutil.rmtree(self.trainer.save_dir, ignore_errors=True)

    def test_compute_token_f1(self):
        result = self.trainer.compute_token_f1(["a b c"], ["a b c"])
        self.assertIsInstance(result, float)

    def test_compute_embedding_similarity(self):
        aligned_embeddings = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        target_embeddings = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        attention_mask = torch.tensor([1, 1])
        result = self.trainer.compute_embedding_similarity(
            aligned_embeddings, target_embeddings, attention_mask
        )
        self.assertIsNotNone(result)

    def test_train_step(self):
        # Mock batch data
        batch_size = 32
        seq = 5
        batch = {"emb_s": torch.rand((batch_size, seq, 512)),
                 "emb_g": torch.rand((batch_size, seq, 768)),
                 'attention_mask_g': torch.rand((batch_size, seq)),
                 "attention_mask_s": torch.rand((batch_size, seq))}
        try:
            self.trainer.train_step(batch)
        except Exception as e:
            self.fail(f"train_step raised an exception: {e}")

    def test_save_checkpoint(self):
        self.trainer.save_checkpoint(epoch=1, metrics={}, is_best=True)
        checkpoint_path = os.path.join(self.trainer.save_dir, "best_model_linear.pt")
        self.assertTrue(os.path.exists(checkpoint_path))

    def test_load_checkpoint(self):
        # Save and load a checkpoint
        self.trainer.save_checkpoint(epoch=1, metrics={}, is_best=True)
        checkpoint_path = os.path.join(self.trainer.save_dir, "best_model_linear.pt")
        self.trainer.load_checkpoint(checkpoint_path, resume_training=False)
        self.assertTrue(os.path.exists(checkpoint_path))


if __name__ == "__main__":
    unittest.main()
