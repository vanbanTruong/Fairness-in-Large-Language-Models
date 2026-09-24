"""Generation-only self-debiasing backend with reference probability damping."""

from __future__ import annotations

import math
import torch


def damped_probabilities(regular, biased, *, decay, epsilon=0.01):
    """Schick et al.: p * max(exp(-lambda * max(q-p, 0)), epsilon)."""
    p = torch.softmax(regular.float(), dim=-1)
    q = torch.stack([torch.softmax(x.float(), dim=-1) for x in biased]).amax(dim=0)
    mask = torch.exp(-decay * (q - p).clamp_min(0)).clamp_min(epsilon)
    damped = p * mask
    return damped / damped.sum(dim=-1, keepdim=True)


class SelfDebiasedDecoder(torch.nn.Module):
    """Preserve HF's generate tensor contract without exposing unedited logits.

    Greedy and nucleus sampling are supported. Other generation modes are
    explicitly refused. Contexts are evaluated afresh; no KV cache is claimed.
    """

    def __init__(
        self, original, tokenizer, spec, *, decay, epsilon, max_new_tokens, seed
    ):
        super().__init__()
        self.original = original
        self.tokenizer = tokenizer
        self.spec = spec
        self.decay = decay
        self.epsilon = epsilon
        self.max_new_tokens = max_new_tokens
        self.seed = seed

    @property
    def device(self):
        from fairLMs.definitions.utils.pll import _input_device

        return _input_device(self.original)

    @property
    def config(self):
        return self.original.config

    def forward(self, *args, **kwargs):
        raise TypeError(
            "SelfDebiasing exposes edited free_generation only; it does not expose edited token_logprobs or hidden states."
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        *,
        max_new_tokens=None,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        num_return_sequences=1,
        pad_token_id=None,
        eos_token_id=None,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(
                f"SelfDebiasing.generate does not support: {', '.join(sorted(kwargs))}."
            )
        budget = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
            raise ValueError("max_new_tokens must be a positive integer.")
        if (
            isinstance(num_return_sequences, bool)
            or not isinstance(num_return_sequences, int)
            or num_return_sequences < 1
        ):
            raise ValueError("num_return_sequences must be a positive integer.")
        if not math.isfinite(temperature) or temperature <= 0 or not 0 < top_p <= 1:
            raise ValueError(
                "temperature must be positive and finite; top_p must be in (0, 1]."
            )
        if input_ids is None:
            raise ValueError("generate requires tokenized input_ids.")
        input_ids = torch.as_tensor(input_ids, dtype=torch.long, device=self.device)
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be a batch of token ID sequences.")
        if attention_mask is not None and tuple(attention_mask.shape) != tuple(
            input_ids.shape
        ):
            raise ValueError("attention_mask must match input_ids.")
        eos = (
            getattr(self.tokenizer, "eos_token_id", None)
            if eos_token_id is None
            else eos_token_id
        )
        eos_ids = set(eos if isinstance(eos, (tuple, list)) else [eos])
        pad = (
            pad_token_id
            if pad_token_id is not None
            else getattr(self.tokenizer, "pad_token_id", None)
        )
        if pad is None:
            pad = next((x for x in eos_ids if x is not None), 0)
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(self.seed)
        outputs = []
        limit = getattr(self.config, "max_position_embeddings", None)
        for row_index, original_prompt in enumerate(input_ids):
            prompt = original_prompt
            if attention_mask is not None:
                prompt = prompt[attention_mask[row_index].to(self.device).bool()]
            if not len(prompt):
                raise ValueError("A generation prompt has no unmasked tokens.")
            text = self.tokenizer.decode(prompt, skip_special_tokens=True)
            prefixes = [
                self.tokenizer.encode(s, return_tensors="pt").to(self.device)[0]
                for s in self.spec.render(text)
            ]
            if not prefixes or any(len(s) == 0 for s in prefixes):
                raise ValueError(
                    "Self-debiasing requires non-empty diagnostic prefixes."
                )
            if (
                isinstance(limit, int)
                and max(len(prompt), *(len(s) for s in prefixes)) + budget > limit
            ):
                raise ValueError(
                    "Prompt plus generation budget exceeds the model context limit."
                )
            for _ in range(num_return_sequences):
                contexts = [prompt.clone(), *(s.clone() for s in prefixes)]
                new_tokens = []
                for _ in range(budget):
                    logits = [
                        self.original(
                            input_ids=s.unsqueeze(0),
                            attention_mask=torch.ones_like(s).unsqueeze(0),
                        ).logits[0, -1]
                        for s in contexts
                    ]
                    probabilities = damped_probabilities(
                        logits[0], logits[1:], decay=self.decay, epsilon=self.epsilon
                    )
                    if do_sample:
                        sampling = torch.softmax(
                            probabilities.log() / temperature, dim=-1
                        )
                        if top_p < 1:
                            ordered, indices = sampling.sort(descending=True)
                            remove = ordered.cumsum(-1) - ordered >= top_p
                            ordered[remove] = 0
                            sampling = torch.zeros_like(sampling).scatter(
                                0, indices, ordered
                            )
                            sampling /= sampling.sum()
                        next_id = int(
                            torch.multinomial(
                                sampling.to(self.device), 1, generator=generator
                            ).item()
                        )
                    else:
                        next_id = int(probabilities.argmax().item())
                    token = torch.tensor(
                        [next_id], device=self.device, dtype=torch.long
                    )
                    new_tokens.append(token)
                    contexts = [torch.cat((s, token)) for s in contexts]
                    if next_id in eos_ids:
                        break
                # Keep original padding so HF callers can slice at input width.
                outputs.append(torch.cat((original_prompt, *new_tokens)))
        return torch.nn.utils.rnn.pad_sequence(
            outputs, batch_first=True, padding_value=pad
        )
