"""Teacher-forced and generation-time internal signals.

HallucinationResearchTest's confidence is the probability of the token the
model actually produced. HalluHard's original maincode.py used max-softmax,
which measures peakedness rather than belief in the emitted answer and is
largely redundant with entropy. Use these helpers whenever a forecast model
is trained on internal signals.
"""

from __future__ import annotations

import torch


FEATURE_CHUNK_SIZE = 64


def summarize_token_signals(token_confidences, token_entropies, token_top_probabilities) -> dict:
    negative_log_likelihood = -torch.log(token_confidences.clamp_min(1e-12))
    mean_nll = negative_log_likelihood.mean().item()
    return {
        "average_confidence": token_confidences.mean().item(),
        "minimum_confidence": token_confidences.min().item(),
        "average_entropy": token_entropies.mean().item(),
        "maximum_entropy": token_entropies.max().item(),
        "average_top_probability": token_top_probabilities.mean().item(),
        "average_negative_log_likelihood": mean_nll,
        "perplexity": float(torch.exp(torch.tensor(mean_nll))),
        "token_count": int(token_confidences.numel()),
    }


def generation_features(step_logits, generated_token_ids, special_ids: set[int] | None = None) -> dict | None:
    """Signals from the model's own decoding steps.

    step_logits[i] is the distribution behind generated_token_ids[i], so
    confidence is the probability of the token the model actually emitted.
    Pass raw logits, not generate()'s temperature-warped scores.
    """
    special_ids = special_ids or set()
    step_count = min(len(step_logits), int(generated_token_ids.shape[0]))
    confidences, entropies, top_probabilities = [], [], []

    for step in range(step_count):
        token_id = int(generated_token_ids[step])
        if token_id in special_ids:
            continue
        log_probabilities = torch.log_softmax(step_logits[step][0].float(), dim=-1)
        probabilities = log_probabilities.exp()
        weighted = torch.where(
            probabilities > 0,
            probabilities * log_probabilities,
            torch.zeros_like(probabilities),
        )
        confidences.append(probabilities[token_id])
        entropies.append(-weighted.sum())
        top_probabilities.append(probabilities.max())

    if not confidences:
        return None
    return summarize_token_signals(
        torch.stack(confidences),
        torch.stack(entropies),
        torch.stack(top_probabilities),
    )


def calculate_features(tokenizer, model, device, question: str, answer: str, chunk_size: int = FEATURE_CHUNK_SIZE) -> dict:
    """Teacher-forced signals over the answer span, not the prompt."""
    full_text = question + "\n" + answer
    encoded = tokenizer(
        full_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        truncation=True,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    special_mask = encoded.pop("special_tokens_mask")[0].bool()
    special_flags = special_mask.tolist()

    answer_char_start = len(question) + 1
    answer_token_start = next(
        (
            i
            for i, ((start, _), is_special) in enumerate(zip(offsets, special_flags))
            if not is_special and start >= answer_char_start
        ),
        None,
    )
    if not answer_token_start:
        raise ValueError("Could not locate the answer span in the tokenized seed.")

    inputs = {key: value.to(device) for key, value in encoded.items()}
    input_ids = inputs["input_ids"][0]
    sequence_length = int(input_ids.shape[0])

    with torch.no_grad():
        logits = model(**inputs).logits[0]

    predict_positions = torch.arange(answer_token_start - 1, sequence_length - 1, device=logits.device)
    target_positions = predict_positions + 1
    keep = ~special_mask.to(logits.device)[target_positions]
    predict_positions = predict_positions[keep]
    target_positions = target_positions[keep]
    if predict_positions.numel() == 0:
        raise ValueError("No scorable answer tokens found after masking specials.")

    targets = input_ids[target_positions]
    confidences, entropies, top_probabilities = [], [], []
    for offset in range(0, int(predict_positions.numel()), chunk_size):
        position_chunk = predict_positions[offset : offset + chunk_size]
        target_chunk = targets[offset : offset + chunk_size]
        chunk_log_probabilities = torch.log_softmax(logits[position_chunk].float(), dim=-1)
        chunk_probabilities = chunk_log_probabilities.exp()
        confidences.append(chunk_probabilities.gather(-1, target_chunk.unsqueeze(-1)).squeeze(-1))
        entropies.append(-(chunk_probabilities * chunk_log_probabilities).sum(dim=-1))
        top_probabilities.append(chunk_probabilities.max(dim=-1).values)

    features = summarize_token_signals(
        torch.cat(confidences),
        torch.cat(entropies),
        torch.cat(top_probabilities),
    )
    last_char_covered = max(
        (end for (_, end), is_special in zip(offsets, special_flags) if not is_special),
        default=0,
    )
    features["answer_truncated"] = last_char_covered < len(full_text)
    return features
