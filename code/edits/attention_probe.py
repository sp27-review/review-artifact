"""Read-only classification-query attention for the frozen HF 4.45.2 models.

Capture the actual (possibly LoRA-adapted) Q/K projection outputs, then reconstruct
only one query row using the model's RoPE, mask, scaling and softmax. The normal
SDPA/Longformer forward and its returned tensors are never replaced. No T x T
attention tensor is constructed by this probe. Batch size one; no KV cache.
"""
from __future__ import annotations

import math
import numpy as np
import torch


def rotate_half(x):
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


class ClassificationAttentionProbe:
    def __init__(self, model):
        if model.training:
            raise ValueError('Call model.eval() before attaching the attention probe')
        import transformers
        if transformers.__version__ != '4.45.2':
            raise RuntimeError('This collector is validated against transformers 4.45.2')
        core = model.get_base_model() if hasattr(model, 'peft_config') else model
        self.kind = core.config.model_type
        if self.kind in ('llama', 'qwen2'):
            modules = [layer.self_attn for layer in core.model.layers]
        elif self.kind == 'longformer':
            modules = [layer.attention.self for layer in core.longformer.encoder.layer]
        else:
            raise ValueError(f'Unsupported architecture: {self.kind}')
        self.modules = modules
        self.handles = []
        self.active = False
        self.rows = {}
        self.cache = {}
        for index, module in enumerate(modules):
            self.handles.append(module.register_forward_pre_hook(self._before(index), with_kwargs=True))
            query = module.query_global if self.kind == 'longformer' else module.q_proj
            key = module.key_global if self.kind == 'longformer' else module.k_proj
            self.handles.append(query.register_forward_hook(self._projection(index, 'q')))
            self.handles.append(key.register_forward_hook(self._projection(index, 'k')))
            if self.kind != 'longformer':
                self.handles.append(module.rotary_emb.register_forward_hook(self._rotary(index)))
            self.handles.append(module.register_forward_hook(self._after(index)))

    def start(self, token_count, query_position=None):
        if self.active:
            raise RuntimeError('Previous probe forward has not finished')
        self.token_count = int(token_count)
        self.position = (0 if self.kind == 'longformer' else token_count - 1) if query_position is None else int(query_position)
        if not 0 <= self.position < self.token_count:
            raise ValueError('Invalid classification query position')
        self.rows, self.cache = {}, {}
        self.active = True

    def _before(self, index):
        def hook(module, args, kwargs):
            if not self.active:
                return
            if module.training:
                raise RuntimeError('Attention collection requires eval mode')
            hidden = kwargs.get('hidden_states', args[0] if args else None)
            if hidden is None or hidden.shape[0] != 1:
                raise ValueError('Only one document per forward is supported')
            if kwargs.get('past_key_value') is not None or kwargs.get('use_cache', False):
                raise ValueError('KV cache must be disabled')
            if kwargs.get('layer_head_mask') is not None:
                raise ValueError('Custom head masks are not supported')
            self.cache[index] = dict(kwargs=kwargs)
        return hook

    def _projection(self, index, name):
        def hook(module, args, output):
            if self.active:
                # Longformer scales its global query in-place after this hook.
                value = output.detach()
                self.cache[index][name] = value.clone() if self.kind == 'longformer' and name == 'q' else value
        return hook

    def _rotary(self, index):
        def hook(module, args, output):
            if self.active:
                self.cache[index]['rotary'] = output
        return hook

    def _after(self, index):
        def hook(module, args, output):
            if not self.active:
                return
            with torch.no_grad():
                cached = self.cache.pop(index)
                row = self._longformer(module, cached) if self.kind == 'longformer' else self._decoder(module, cached)
                if not torch.isfinite(row).all() or (row < 0).any():
                    raise ValueError(f'Invalid attention probabilities in layer {index}')
                if not torch.allclose(row.float().sum(-1), torch.ones(row.shape[0], device=row.device), atol=.015, rtol=0):
                    raise ValueError(f'Attention row does not sum to one in layer {index}')
                if row.shape[-1] > self.token_count and row[:, self.token_count:].float().abs().max() > 1e-6:
                    raise ValueError('Padded tokens received attention')
                self.rows[index] = row[:, :self.token_count].float().cpu().numpy()
            # Returning None leaves the model output entirely untouched.
        return hook

    def _decoder(self, module, cached):
        q, k, kw = cached['q'], cached['k'], cached['kwargs']
        seq = q.shape[1]
        if seq != self.token_count or k.shape[1] != seq:
            raise ValueError('Unexpected decoder sequence length or KV cache')
        q = q.view(1, seq, module.num_heads, module.head_dim).transpose(1, 2)
        k = k.view(1, seq, module.num_key_value_heads, module.head_dim).transpose(1, 2)
        # Qwen2's 4.45.2 eager fallback recomputes RoPE internally, even when
        # the outer SDPA call received position_embeddings. Prefer what ran.
        rotary = cached.get('rotary')
        if rotary is None:
            rotary = kw.get('position_embeddings')
        if rotary is None:
            raise ValueError('Could not capture actual RoPE embeddings')
        cos, sin = (v.to(q.device).unsqueeze(1) for v in rotary)
        pos = self.position
        q = q[:, :, pos:pos+1, :]
        q = q * cos[:, :, pos:pos+1, :] + rotate_half(q) * sin[:, :, pos:pos+1, :]
        k = k * cos + rotate_half(k) * sin
        k = k.repeat_interleave(module.num_key_value_groups, dim=1)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(module.head_dim)
        mask = kw.get('attention_mask')
        if mask is not None:
            if mask.ndim != 4 or mask.dtype == torch.bool:
                raise ValueError('Expected HF additive 4D attention mask')
            query_index = pos if mask.shape[-2] > 1 else 0
            scores = scores + mask[:, :, query_index:query_index+1, :seq].to(scores.device)
        else:
            scores[..., pos+1:] = torch.finfo(scores.dtype).min
        return torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)[0, :, 0, :]

    def _longformer(self, module, cached):
        q, k, kw = cached['q'], cached['k'], cached['kwargs']
        global_mask = kw.get('is_index_global_attn')
        if self.position != 0 or global_mask is None or int(global_mask.sum()) != 1 or not bool(global_mask[0, 0]):
            raise ValueError('Longformer probe requires CLS as the only global token')
        if q.shape[:2] != (1, 1) or k.shape[1] != 1:
            raise ValueError('Unexpected Longformer global projection shape')
        q = q / math.sqrt(module.head_dim)
        q = q.contiguous().view(1, module.num_heads, module.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, module.num_heads, module.head_dim).transpose(0, 1)
        scores = torch.bmm(q, k.transpose(1, 2))
        masked = kw['is_index_masked'].to(scores.device)
        scores = scores.masked_fill(masked[:, None, :], torch.finfo(scores.dtype).min)
        return torch.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)[:, 0, :]

    def finish(self):
        self.active = False
        if set(self.rows) != set(range(len(self.modules))) or self.cache:
            raise RuntimeError('Incomplete attention capture')
        values = np.stack([self.rows[i] for i in range(len(self.modules))])
        self.rows = {}
        return values

    def close(self):
        self.active = False
        self.rows.clear()
        self.cache.clear()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
