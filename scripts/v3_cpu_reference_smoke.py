import torch
import transformers

from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

import llm_runtime.v3_extension_contracts
import llm_runtime.v3_plugin_registry

torch.set_num_threads(2)
torch.manual_seed(42)

print("PyTorch:", torch.__version__, flush=True)
print("Transformers:", transformers.__version__, flush=True)

config = GPT2Config(
    vocab_size=128,
    n_positions=64,
    n_ctx=64,
    n_embd=64,
    n_layer=4,
    n_head=4,
    resid_pdrop=0.0,
    embd_pdrop=0.0,
    attn_pdrop=0.0,
    use_cache=True,
)

model = GPT2LMHeadModel(config).eval()
tokens = torch.tensor([[4, 12, 15, 31, 8, 26]])

with torch.no_grad():
    reference = model(
        input_ids=tokens,
        use_cache=True,
    )

print("PASS: GPT-2 reference forward", flush=True)

# Check incremental decoding against full sequence.
with torch.no_grad():
    prefix = model(
        input_ids=tokens[:, :-1],
        use_cache=True,
    )

    incremental = model(
        input_ids=tokens[:, -1:],
        past_key_values=prefix.past_key_values,
        use_cache=True,
    )

torch.testing.assert_close(
    reference.logits[:, -1, :],
    incremental.logits[:, -1, :],
    rtol=1e-4,
    atol=1e-4,
)

print("PASS: Incremental KV-cache decoding", flush=True)

# Install a neutral wrapper on one attention layer.
class IdentityAttention(nn.Module):
    def __init__(self, native):
        super().__init__()
        self.native = native
        self.calls = 0

    def forward(self, *args, **kwargs):
        self.calls += 1
        return self.native(*args, **kwargs)


native_layer_0 = model.transformer.h[0].attn
native_layer_2 = model.transformer.h[2].attn

wrapper = IdentityAttention(
    model.transformer.h[1].attn
)
model.transformer.h[1].attn = wrapper

with torch.no_grad():
    identity = model(input_ids=tokens)

assert wrapper.calls > 0

torch.testing.assert_close(
    reference.logits,
    identity.logits,
    rtol=0,
    atol=0,
)

assert model.transformer.h[0].attn is native_layer_0
assert model.transformer.h[2].attn is native_layer_2

print("PASS: Selected-layer identity attention", flush=True)
print("PASS: Nonselected layers unchanged", flush=True)

# Apply a controlled attention-output modification.
class ModifiedAttention(nn.Module):
    def __init__(self, native):
        super().__init__()
        self.native = native
        self.calls = 0

    def forward(self, *args, **kwargs):
        self.calls += 1
        result = self.native(*args, **kwargs)
        altered = result[0].clone()
        altered[..., 0] += 0.5
        return (altered, *result[1:])


modified = ModifiedAttention(native_layer_2)
model.transformer.h[2].attn = modified

with torch.no_grad():
    changed = model(input_ids=tokens)

assert modified.calls > 0

difference = (
    changed.logits - identity.logits
).abs().max().item()

assert difference > 1e-6, difference

print("PASS: Controlled attention modification", flush=True)
print("Maximum logit difference:", difference, flush=True)
print("PASS: CPU reference execution smoke test", flush=True)
