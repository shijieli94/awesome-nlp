import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TRANSFORMERS_DIR = Path(Path.home(), "Data", ".cache", "transformers").as_posix()

assert torch.cuda.is_available()

device = "cuda"  # the device to load the model onto
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, cache_dir=TRANSFORMERS_DIR)
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=TRANSFORMERS_DIR)

messages = [
    {
        "role": "user",
        "content": "What is your favourite condiment?",
    },
    {
        "role": "assistant",
        "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!",
    },
    {
        "role": "user",
        "content": "Do you have mayonnaise recipes?",
    },
]

input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

encoded = tokenizer(input_text, return_tensors="pt", add_special_tokens=False).to(device)

print("Input:\n" + input_text + "\n\n")


@torch.no_grad()
def sample_autoregressive(token_ids: torch.Tensor, num_tokens=128):
    len_prefix = token_ids.shape[1]
    t = torch.cat([token_ids, token_ids.new_full((1, num_tokens), fill_value=0)], dim=-1)
    for i in range(num_tokens):
        new_token_id = model(t[:, : len_prefix + i], use_cache=False).logits[0, -1, :].argmax()
        t[0, len_prefix + i] = new_token_id.item()

    return t[0, len_prefix:].reshape((1, -1))


torch.cuda.synchronize()
t0 = time.time()

generated_ids = sample_autoregressive(encoded["input_ids"], num_tokens=128)

torch.cuda.synchronize()
t1 = time.time()

decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=False)

print(f"sample_autoregressive ({t1 - t0:.2f}s):\n" + 100 * "-" + "\n" + decoded + "\n\n")


@torch.no_grad()
def sample_jacobi_decode(token_ids: torch.Tensor, num_tokens=128, num_extra=3):
    assert token_ids.shape[0] == 1

    len_prefix = token_ids.shape[1]
    t = torch.cat([token_ids, token_ids.new_full((1, num_tokens), 0)], dim=-1)

    i = len_prefix  # write index
    while i < len_prefix + num_tokens:
        n = min(num_extra, len_prefix + num_tokens - i - 1)

        # forward pass
        indices = model(t[:, : i + n], use_cache=False).logits[0, -(n + 1) :, :].argmax(dim=-1)

        # comparison
        nhits = 0
        for j in range(n):
            if indices[j].item() != t[0, i + j].item():
                break
            nhits += 1

        # update guesses for next round 👍
        t[0, i : i + n + 1] = indices

        i += nhits + 1

    return t[0, len_prefix:].reshape((1, -1))


torch.cuda.synchronize()
t0 = time.time()

generated_ids = sample_jacobi_decode(encoded["input_ids"], num_tokens=128, num_extra=3)

torch.cuda.synchronize()
t1 = time.time()

decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=False)

print(f"sample_jacobi_decode ({t1 - t0:.2f}s):\n" + 100 * "-" + "\n" + decoded + "\n\n")


@torch.no_grad()
def sample_jacobi_decode_kv_caching(token_ids: torch.Tensor, num_tokens=128, num_extra=3):
    assert token_ids.shape[0] == 1

    len_prefix = token_ids.shape[1]
    t = torch.cat([token_ids, token_ids.new_full((1, num_tokens), 0)], dim=-1)

    cache = None
    i = len_prefix  # write index
    while i < len_prefix + num_tokens:
        n = min(num_extra, len_prefix + num_tokens - i - 1)

        # forward pass
        res = model(t[:, (0 if i == len_prefix else i - 1) : i + n], use_cache=True, past_key_values=cache)
        cache = res.past_key_values
        indices = res.logits[0, -(n + 1) :, :].argmax(dim=-1)

        # comparison
        nhits = 0
        for j in range(n):
            if indices[j].item() != t[0, i + j].item():
                break
            nhits += 1

        # update guesses for next round 👍
        t[0, i : i + n + 1] = indices
        cache = [
            (key_states[:, :, : i + nhits], value_states[:, :, : i + nhits]) for key_states, value_states in cache
        ]

        i += nhits + 1

    return t[0, len_prefix:].reshape((1, -1))


torch.cuda.synchronize()
t0 = time.time()

generated_ids = sample_jacobi_decode_kv_caching(encoded["input_ids"], num_tokens=128, num_extra=3)

torch.cuda.synchronize()
t1 = time.time()

decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=False)

print(f"sample_jacobi_decode_kv_caching ({t1 - t0:.2f}s):\n" + 100 * "-" + "\n" + decoded + "\n\n")
