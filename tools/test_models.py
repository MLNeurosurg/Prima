import unittest
import torch
import os
import yaml
import sys
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from tools.models import ModelLoader
from Prima_training_and_evaluation import (
    RachelDataset,
    CLIP,
    ViT,
    GPTWrapper,
    HierViT,
    SerieTransformerEncoder
)

class TestModelLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test configuration and paths"""
        # Create a minimal test config
        cls.test_config = {
            'model': {
                'vqvae': {
                    'spatial_dims': 3,
                    'in_channels': 1,
                    'out_channels': 1,
                    'num_res_layers': 2,
                    'downsample_parameters': [(2, 4, 1, 1), (2, 4, 1, 1)],
                    'upsample_parameters': [(1, 4, 1, 0), (1, 4, 1, 0)],
                    'num_channels': [32, 64],
                    'num_res_channels': [32, 64],
                    'num_embeddings': 512,
                    'embedding_dim': 64,
                    'ckpt_path': 'test_models/test_vqvae.pt'
                },
                'clip_ckpt': 'test_models/test_clip.pt',
                'visual': {
                    'type': 'hiervit',
                    'inner': {
                        'dim': 1024,
                        'depth': 6,
                        'heads': 8,
                        'mlp_dim': 2048,
                        'dim_head': 64,
                        'clsnum': 1
                    },
                    'outer': {
                        'dim': 256,
                        'depth': 4,
                        'heads': 4,
                        'mlp_dim': 512,
                        'dim_head': 64,
                        'clsnum': 1
                    }
                },
                'text': {
                    'type': 'gpt2',
                    'feature_dim': 256
                }
            },
            'classification_heads': {
                'test_condition': {
                    'model_path': 'test_models/test_head.pt',
                    'threshold': 0.5
                }
            }
        }
        
        # Create test directory if it doesn't exist
        os.makedirs('test_models', exist_ok=True)
        
        # Create dummy model files for testing
        cls._create_dummy_models()
        
    @classmethod
    def _create_dummy_models(cls):
        """Create dummy model files for testing"""
        # Create dummy VQVAE model
        vqvae = torch.nn.Sequential(
            torch.nn.Conv3d(1, 32, 3),
            torch.nn.ReLU(),
            torch.nn.Conv3d(32, 1, 3)
        )
        torch.save(vqvae.state_dict(), 'test_models/test_vqvae.pt')
        
        # Create dummy CLIP model
        clip = CLIP(cls.test_config)
        torch.save(clip, 'test_models/test_clip.pt')
        
        # Create dummy classification head
        head = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )
        torch.save(head, 'test_models/test_head.pt')

    def test_vqvae_loading(self):
        """Test VQVAE model loading"""
        model_loader = ModelLoader(self.test_config, gpu_num=0)
        vqvae = model_loader.load_vqvae_model()
        
        # Check if model is loaded and on correct device
        self.assertIsInstance(vqvae, torch.nn.Module)
        self.assertEqual(str(vqvae.device), f'cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Test forward pass with dummy input
        dummy_input = torch.randn(1, 1, 32, 32, 32)
        try:
            output = vqvae(dummy_input)
            self.assertIsInstance(output, torch.Tensor)
        except Exception as e:
            self.fail(f"VQVAE forward pass failed: {str(e)}")

    def test_prima_model_loading(self):
        """Test PRIMA model loading"""
        model_loader = ModelLoader(self.test_config, gpu_num=0)
        prima = model_loader.load_prima_model()
        
        # Check if model is loaded and on correct device
        self.assertIsInstance(prima, torch.nn.Module)
        self.assertEqual(str(prima.device), f'cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Test forward pass with dummy input
        dummy_input = {
            'visual': torch.randn(1, 10, 1024),  # batch_size, sequence_length, embedding_dim
            'lens': torch.tensor([10])  # sequence lengths
        }
        try:
            output = prima.visual_model(dummy_input)
            self.assertIsInstance(output, torch.Tensor)
        except Exception as e:
            self.fail(f"PRIMA model forward pass failed: {str(e)}")

    def test_classification_heads_loading(self):
        """Test classification heads loading"""
        model_loader = ModelLoader(self.test_config, gpu_num=0)
        heads = model_loader.load_classification_heads()
        
        # Check if heads are loaded correctly
        self.assertIsInstance(heads, dict)
        self.assertIn('test_condition', heads)
        
        # Check head structure
        head_info = heads['test_condition']
        self.assertIn('model', head_info)
        self.assertIn('threshold', head_info)
        self.assertIsInstance(head_info['model'], torch.nn.Module)
        self.assertIsInstance(head_info['threshold'], float)
        
        # Test forward pass with dummy input
        dummy_input = torch.randn(1, 256)  # batch_size, embedding_dim
        try:
            output = head_info['model'](dummy_input)
            self.assertIsInstance(output, torch.Tensor)
        except Exception as e:
            self.fail(f"Classification head forward pass failed: {str(e)}")

    def test_dataset_loading(self):
        """Test dataset loading"""
        # Create minimal dataset config
        dataset_config = {
            'datajson': 'test_data/data.json',
            'datarootdir': 'test_data',
            'text_max_len': 200,
            'is_train': True,
            'tokenizer': 'gpt2',
            'vqvaename': 'test_vqvae',
            'visualhashonly': True
        }
        
        # Create test data directory and files
        os.makedirs('test_data', exist_ok=True)
        with open('test_data/data.json', 'w') as f:
            f.write('{"test_study": {"series": ["test_series"]}}')
        
        try:
            dataset = RachelDataset(**dataset_config)
            self.assertIsInstance(dataset, RachelDataset)
            
            # Test dataset length
            self.assertGreater(len(dataset), 0)
            
            # Test getting an item
            item = dataset[0]
            self.assertIsInstance(item, dict)
            self.assertIn('visual', item)
            self.assertIn('hash', item)
            
        except Exception as e:
            self.fail(f"Dataset loading failed: {str(e)}")
        finally:
            # Clean up test files
            if os.path.exists('test_data'):
                import shutil
                shutil.rmtree('test_data')

    @classmethod
    def tearDownClass(cls):
        """Clean up test files"""
        if os.path.exists('test_models'):
            import shutil
            shutil.rmtree('test_models')

if __name__ == '__main__':
    unittest.main() 