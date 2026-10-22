"""A tiny, randomly initialised Qwen3-VL for CPU smoke runs and tests.

It uses the *real* ``Qwen3VLForConditionalGeneration`` / ``Qwen3VLProcessor`` /
``Qwen2VLImageProcessor`` classes, the Qwen chat format and special tokens, so the training and
inference code paths are exactly the ones used with ``Qwen/Qwen3-VL-2B-Instruct``; only the
sizes (and the small byte-level BPE vocabulary trained on our own prompts/targets) differ.
Its outputs are meaningless and are never reported as results.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from receipt_vlm.prompts import INSTRUCTION, RETRY_REMINDER, SYSTEM_PROMPT, TEXT_INSTRUCTION_PREFIX

SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
]

# Qwen-style chat template (same rendering as Qwen2/3-VL for text + image content).
CHAT_TEMPLATE = (
    "{%- for message in messages -%}"
    "{{ '<|im_start|>' + message['role'] + '\\n' }}"
    "{%- if message['content'] is string -%}{{ message['content'] }}"
    "{%- else -%}{%- for c in message['content'] -%}"
    "{%- if c['type'] == 'image' or 'image' in c -%}{{ '<|vision_start|><|image_pad|><|vision_end|>' }}"
    "{%- elif c['type'] == 'text' -%}{{ c['text'] }}{%- endif -%}"
    "{%- endfor -%}{%- endif -%}"
    "{{ '<|im_end|>\\n' }}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{ '<|im_start|>assistant\\n' }}{%- endif -%}"
)


def _corpus(extra: Iterable[str]) -> list[str]:
    base = [SYSTEM_PROMPT, INSTRUCTION, RETRY_REMINDER, TEXT_INSTRUCTION_PREFIX]
    digits = [
        " ".join(str(i) for i in range(k, k + 50)) + " 0.5 12.99 1,250.00"
        for k in range(0, 1000, 50)
    ]
    return base * 5 + digits + list(extra)


def build_tiny_qwen3_vl(
    out_dir: str | Path, extra_text: Iterable[str] = (), vocab_size: int = 2048, seed: int = 0
) -> Path:
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen2VLImageProcessor,
        Qwen3VLConfig,
        Qwen3VLForConditionalGeneration,
        Qwen3VLProcessor,
    )
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(_corpus(extra_text), trainer=trainer)
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=SPECIAL_TOKENS[1:],
    )
    hf_tok.chat_template = CHAT_TEMPLATE
    hf_tok.padding_side = "left"

    image_processor = Qwen2VLImageProcessor(
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        min_pixels=32 * 32 * 4,
        max_pixels=32 * 32 * 64,
    )
    video_processor = Qwen3VLVideoProcessor(patch_size=16, temporal_patch_size=2, merge_size=2)
    processor = Qwen3VLProcessor(
        image_processor=image_processor,
        tokenizer=hf_tok,
        video_processor=video_processor,
        chat_template=CHAT_TEMPLATE,
    )

    ids: dict[str, int] = {t: int(hf_tok.convert_tokens_to_ids(t)) for t in SPECIAL_TOKENS}  # type: ignore[arg-type]
    head_dim = 16
    config = Qwen3VLConfig(
        text_config={
            "vocab_size": len(hf_tok),
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": head_dim,
            "max_position_embeddings": 4096,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 5000000.0,
                "mrope_section": [4, 2, 2],  # sums to head_dim / 2
                "mrope_interleaved": True,
            },
            "pad_token_id": ids["<|endoftext|>"],
        },
        vision_config={
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 64,
            "num_position_embeddings": 256,
            "deepstack_visual_indexes": [0],
        },
        image_token_id=ids["<|image_pad|>"],
        video_token_id=ids["<|video_pad|>"],
        vision_start_token_id=ids["<|vision_start|>"],
        vision_end_token_id=ids["<|vision_end|>"],
        tie_word_embeddings=False,
    )
    torch.manual_seed(seed)
    model = Qwen3VLForConditionalGeneration(config)
    model.generation_config.eos_token_id = ids["<|im_end|>"]
    model.generation_config.pad_token_id = ids["<|endoftext|>"]
    model.save_pretrained(out)
    processor.save_pretrained(out)
    return out
