# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from contextlib import nullcontext
from typing import TYPE_CHECKING, Optional

import torch
from transformers.integrations import is_deepspeed_zero3_enabled

from ...extras import logging


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


logger = logging.get_logger(__name__)


def _noisy_mean_initialization(embed_weight: "torch.Tensor", num_new_tokens: int) -> None:
    """Initialize new token embeddings with mean + Gaussian noise.

    This is the default initialization method used by LlamaFactory.

    Args:
        embed_weight: The embedding weight matrix to initialize (shape: [vocab_size, embedding_dim])
        num_new_tokens: Number of new tokens added at the end of the embedding matrix
    """
    embedding_dim = embed_weight.size(1)
    avg_weight = embed_weight[:-num_new_tokens].mean(dim=0, keepdim=True)
    noise_weight = torch.empty_like(embed_weight[-num_new_tokens:])
    noise_weight.normal_(mean=0, std=(1.0 / math.sqrt(embedding_dim)))
    embed_weight[-num_new_tokens:] = avg_weight + noise_weight


def _gaussian_initialization(
    embed_weight: "torch.Tensor",
    num_new_tokens: int,
    sigma: float = 0.02,
    token_ids: Optional[list[int]] = None,
) -> None:
    """Initialize new token embeddings with pure Gaussian noise, with no mean offset.

    This is the initialization PaliGemma (Beyer et al., 2024) reports as the *better* of the two
    they compare when adding a block of location tokens to a pretrained LM: small Gaussian noise
    (sigma = 0.02). Their other arm -- "matching the average of the pretrained embeddings plus
    small noise" -- is what ``_noisy_mean_initialization`` already implements, and is the arm they
    find loses after ~1k steps despite a better initial perplexity.

    The distinction is the centroid term, not the noise: this function writes N(0, sigma^2) rows,
    so the block starts spread around the origin rather than clustered around the old vocabulary's
    mean. Reference scale on Qwen3.5-2B: the pretrained rows have per-element std 0.0150 and mean
    row norm 0.676, so sigma = 0.02 yields rows about 34% longer than a typical pretrained row.
    Pass sigma = 0.015 instead for a scale-matched variant.

    Args:
        embed_weight: The embedding weight matrix to initialize (shape: [vocab_size, embedding_dim])
        num_new_tokens: Number of new tokens added at the end of the embedding matrix
        sigma: Standard deviation of the Gaussian. Defaults to PaliGemma's 0.02.
        token_ids: Explicit rows to initialise. REQUIRED for correctness whenever the new tokens do
            not all land at the end of the matrix. A resize only appends the ids the tokeniser did
            not already have: on Qwen3.5-0.8B, adding the 1001 <coord_k> tokens grows the matrix by
            768 because 233 of them fall on pre-existing reserved rows. Writing ``[-num_new:]`` would
            leave those 233 carrying their base-model values -- the same offset-vs-id bug that
            corrupted desc_init in 2026-07. Pass the ids and every coordinate row is written alike.
    """
    if token_ids:
        rows = torch.tensor(sorted(set(token_ids)), dtype=torch.long, device=embed_weight.device)
        embed_weight[rows] = torch.randn(
            len(rows), embed_weight.size(1), dtype=embed_weight.dtype, device=embed_weight.device
        ) * sigma
    else:
        embed_weight[-num_new_tokens:].normal_(mean=0.0, std=sigma)


def _description_based_initialization(
    embed_weight: "torch.Tensor",
    num_new_tokens: int,
    descriptions: dict[str, str],
    tokenizer: "PreTrainedTokenizer",
    model: "PreTrainedModel",
    add_noise: bool = False,
) -> None:
    """Initialize new token embeddings based on textual descriptions.

    For each new token, this function:
    1. Tokenizes its description text
    2. Gets embeddings of the description tokens
    3. Averages them to initialize the new token's embedding
    4. Optionally adds Gaussian noise

    Args:
        embed_weight: The embedding weight matrix to initialize (shape: [vocab_size, embedding_dim])
        num_new_tokens: Number of new tokens added
        descriptions: Dict mapping token string to its description text
                      e.g., {"<think>": "A token representing reasoning process"}
        tokenizer: The tokenizer instance
        model: The model instance (used to get input embeddings)
        add_noise: Whether to add Gaussian noise to the initialization

    Example:
        descriptions = {
            "<|START_OF_SVG|>": "Marks the beginning of an SVG document",
            "<|END_OF_SVG|>": "Marks the end of an SVG document"
        }
    """
    embedding_dim = embed_weight.size(1)

    # Resolve each description key to its actual token id. Never index rows as
    # `[-num_new_tokens + i]`: models like Qwen ship with a pre-padded vocab
    # (embedding rows > len(tokenizer)), so resize may create FEWER rows than
    # len(descriptions). Position-based indexing then (a) lands descriptions on
    # the wrong new tokens and (b) wraps around to positive indices, silently
    # overwriting the embeddings of existing low-id tokens.
    added_vocab_ids = {
        token_id
        for token_id in (tokenizer.convert_tokens_to_ids(tok_str) for tok_str in descriptions.keys())
        if token_id is not None
    }
    # Description texts must be embedded with pre-existing tokens only.
    snapshot_embedding = embed_weight.detach().clone()

    for i, (tok_str, desc) in enumerate(descriptions.items()):
        token_id = tokenizer.convert_tokens_to_ids(tok_str)
        if token_id is None or token_id < 0 or token_id >= embed_weight.size(0):
            logger.warning_rank0(f"Token {tok_str!r} not found in tokenizer; skipping its initialization.")
            continue

        # Tokenize description text
        tokens = tokenizer(desc, return_tensors="pt", add_special_tokens=False)

        with torch.no_grad():
            token_ids = tokens["input_ids"][0]
            # Move to the same device as embed_weight
            device = embed_weight.device
            token_ids = token_ids.to(device)

            # Filter out the new tokens themselves (they don't have valid embeddings yet)
            valid_token_ids = torch.tensor(
                [tid for tid in token_ids.tolist() if tid not in added_vocab_ids], device=device, dtype=torch.long
            )

            if len(valid_token_ids) == 0:
                # Fallback: use mean of all existing embeddings
                logger.warning_rank0(
                    f"Description for token {i + 1}/{len(descriptions)} contains no valid tokens. "
                    "Using mean of existing embeddings."
                )
                base_embedding = snapshot_embedding[: -num_new_tokens if num_new_tokens > 0 else None].mean(dim=0)
            else:
                # Average the PRE-INIT embeddings of the description tokens (snapshot avoids
                # cascading reads of rows this loop has already written).
                base_embedding = snapshot_embedding[valid_token_ids].mean(dim=0)

            # Add noise if requested (ensure correct device and dtype)
            if add_noise:
                noise = torch.randn_like(base_embedding) * (1.0 / math.sqrt(embedding_dim))
                embed_weight[token_id] = base_embedding + noise
            else:
                embed_weight[token_id] = base_embedding


def _initialize_embeddings(
    embed_weight: "torch.Tensor",
    num_new_tokens: int,
    init_method: str,
    new_special_tokens_config: Optional[dict],
    tokenizer: "PreTrainedTokenizer",
    model: "PreTrainedModel",
    sigma: float = 0.02,
) -> None:
    """Single source of truth for embedding initialization.

    This function selects the appropriate initialization method and applies it.

    Args:
        embed_weight: The embedding weight matrix to initialize
        num_new_tokens: Number of new tokens added
        init_method: Initialization method ('noise_init', 'desc_init', 'desc_init_w_noise')
        new_special_tokens_config: Config dict with token descriptions (required for desc_init methods)
        tokenizer: The tokenizer instance
        model: The model instance
    """
    if init_method == "desc_init" and new_special_tokens_config:
        logger.info_rank0("Using semantic initialization (desc_init) for new special tokens")
        _description_based_initialization(
            embed_weight, num_new_tokens, new_special_tokens_config, tokenizer, model, add_noise=False
        )
    elif init_method == "desc_init_w_noise" and new_special_tokens_config:
        logger.info_rank0("Using semantic initialization with noise (desc_init_w_noise) for new special tokens")
        _description_based_initialization(
            embed_weight, num_new_tokens, new_special_tokens_config, tokenizer, model, add_noise=True
        )
    elif init_method == "gaussian_init":
        # Resolve the rows by TOKEN ID, not by tail offset: a resize only appends the ids the
        # tokeniser lacked, so some of the new tokens can land on pre-existing reserved rows.
        ids = None
        if new_special_tokens_config:
            ids = [i for i in (tokenizer.convert_tokens_to_ids(t) for t in new_special_tokens_config)
                   if i is not None and 0 <= i < embed_weight.size(0)]
        logger.info_rank0(
            f"Using pure Gaussian initialization (gaussian_init, sigma={sigma}) for "
            f"{len(ids) if ids else num_new_tokens} rows addressed by "
            f"{'token id' if ids else 'tail offset'}; no centroid offset is applied"
        )
        _gaussian_initialization(embed_weight, num_new_tokens, sigma=sigma, token_ids=ids)
    else:
        if init_method != "noise_init":
            logger.warning_rank0(
                f"init_method='{init_method}' requires descriptions config, falling back to 'noise_init'"
            )
        logger.info_rank0("Using noisy mean initialization (noise_init) for new special tokens")
        _noisy_mean_initialization(embed_weight, num_new_tokens)


def resize_embedding_layer(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    new_special_tokens_config: Optional[dict] = None,
    init_special_tokens: str = "noise_init",
    init_special_tokens_sigma: float = 0.02,
) -> None:
    r"""Resize token embeddings and initialize new tokens.

    Args:
        model: The model to resize
        tokenizer: The tokenizer (used to get target vocab size)
        new_special_tokens_config: Optional dict with token descriptions for semantic initialization
        init_special_tokens: Initialization method ('noise_init', 'desc_init', 'desc_init_w_noise')
    """
    if is_deepspeed_zero3_enabled():
        import deepspeed  # type: ignore

        params = [model.get_input_embeddings().weight]
        if model.get_output_embeddings() is not None and not model.config.tie_word_embeddings:
            params.append(model.get_output_embeddings().weight)

        context_maybe_zero3 = deepspeed.zero.GatheredParameters(params, modifier_rank=0)
    else:
        context_maybe_zero3 = nullcontext()

    with context_maybe_zero3:
        current_embedding_size = model.get_input_embeddings().weight.size(0)

    if len(tokenizer) > current_embedding_size:
        if getattr(model, "quantization_method", None):
            raise ValueError("Cannot resize embedding layers of a quantized model.")

        if not isinstance(model.get_output_embeddings(), torch.nn.Linear):
            raise ValueError("Current model does not support resizing embedding layers.")

        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
        with context_maybe_zero3:
            new_embedding_size = model.get_input_embeddings().weight.size(0)
            num_new_tokens = new_embedding_size - current_embedding_size
            logger.info_rank0(
                f"Resizing embeddings: {current_embedding_size} -> {new_embedding_size} (+{num_new_tokens} tokens)"
            )

            # Initialize input embeddings
            _initialize_embeddings(
                model.get_input_embeddings().weight.data,
                num_new_tokens,
                init_special_tokens,
                new_special_tokens_config,
                tokenizer,
                model,
                sigma=init_special_tokens_sigma,
            )

            # Initialize output embeddings if not tied
            if model.get_output_embeddings() is not None and not model.config.tie_word_embeddings:
                _initialize_embeddings(
                    model.get_output_embeddings().weight.data,
                    num_new_tokens,
                    init_special_tokens,
                    new_special_tokens_config,
                    tokenizer,
                    model,
                    sigma=init_special_tokens_sigma,
                )

        model.config.vocab_size = new_embedding_size
        logger.info_rank0(f"Resized token embeddings from {current_embedding_size} to {new_embedding_size}.")
