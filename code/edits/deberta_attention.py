"""Native DeBERTa CLS-query capture for full-length final-checkpoint inputs.

Matches the training run's trajectory method, without importing its training
worker or materializing the entire tokenized dataset. Dense layer weights are
released immediately after their CLS row has been copied.
"""
import numpy as np


def collect(model, batch):
    import torch
    if model.training or model.config.model_type != 'deberta-v2':
        raise ValueError('Expected an eval-mode DeBERTa-v2/v3 classifier')
    if batch['input_ids'].shape[0] != 1:
        raise ValueError('DeBERTa attention requires one document per forward')
    captured = []
    handles = []

    def capture(module, args, output):
        hidden, weights = output
        if weights.ndim != 4 or weights.shape[0] != 1:
            raise ValueError('Unexpected native DeBERTa attention shape')
        values = weights[0, :, 0, :].float().detach().cpu().numpy()
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Invalid DeBERTa attention probabilities')
        if not np.allclose(values.sum(-1), 1., atol=.015, rtol=0):
            raise ValueError('DeBERTa attention does not sum to one')
        masked = batch['attention_mask'][0].detach().cpu().numpy() == 0
        if masked.any() and np.max(np.abs(values[:, masked])) > 1e-6:
            raise ValueError('DeBERTa padding received attention')
        captured.append(values)
        return hidden, None

    try:
        for layer in model.deberta.encoder.layer:
            handles.append(layer.attention.self.register_forward_hook(capture))
        with torch.no_grad():
            logits = model(**batch, output_attentions=True).logits.float()[0].clone()
        if len(captured) != model.config.num_hidden_layers:
            raise ValueError('Incomplete DeBERTa attention capture')
        return logits, np.stack(captured)
    finally:
        for handle in handles:
            handle.remove()
