"""Small CPU forward/backward and pre-norm regression (requires transformers)."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/iteredit'))
from configs.config import LevTModelConfig
from models.levenshtein_transformer import LevenshteinTransformer


class IterEditTests(unittest.TestCase):
    def test_pre_norm_forward_backward_and_reload(self):
        config = LevTModelConfig(vocab_size=32, hidden_size=16,
            num_hidden_layers=2, num_attention_heads=2, intermediate_size=32,
            max_position_embeddings=64, pad_token_id=0, dropout=0.)
        model = LevenshteinTransformer(config).eval()
        ids = torch.tensor([[1, 2, 3, 0]])
        mask = torch.tensor([[1, 1, 1, 0]])
        events = []
        layer = model.bert.encoder.layer[0]
        handles = [module.register_forward_hook(
            lambda _m, _i, _o, name=name: events.append(name)) for name, module in (
                ('norm_attention', layer.attention.output.LayerNorm),
                ('attention', layer.attention.self),
                ('norm_ffn', layer.output.LayerNorm),
                ('ffn', layer.intermediate))]
        result = model(ids, mask)
        for handle in handles:
            handle.remove()
        self.assertEqual(events, ['norm_attention', 'attention', 'norm_ffn', 'ffn'])
        self.assertEqual(result['del_logits'].shape, (1, 4, 2))
        self.assertEqual(result['ins_logits'].shape, (1, 5, 21))
        self.assertEqual(result['tok_logits'].shape, (1, 4, 32))
        sum(x.square().mean() for x in result.values()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters() if p.requires_grad))
        modified = ids.clone()
        modified[0, -1] = 9
        torch.testing.assert_close(model(ids, mask)['tok_logits'][:, :3],
                                   model(modified, mask)['tok_logits'][:, :3])
        clone = LevenshteinTransformer(config).eval()
        clone.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(clone(ids, mask)['tok_logits'], result['tok_logits'])


if __name__ == '__main__':
    unittest.main()
