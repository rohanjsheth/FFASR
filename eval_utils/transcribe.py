"""Greedy transcription shared by the offline eval and the training callback."""

from typing import TYPE_CHECKING, Any, Sequence

import torch

if TYPE_CHECKING:
    from data_utils.data_utils import RenderedScene


def transcribe_batch(
    model: Any,
    processor: Any,
    scenes: Sequence["RenderedScene"],
    language: str,
    sample_rate: int,
    max_new_tokens: int,
    device: Any,
    model_dtype: Any,
) -> list[str]:
    processor_inputs = processor.apply_transcription_request(
        audio=[scene["audio"] for scene in scenes],
        language=language,
        processor_kwargs={"sampling_rate": sample_rate},
    )
    prompt_length = processor_inputs["input_ids"].shape[1]
    processor_inputs = processor_inputs.to(device, model_dtype)

    with torch.inference_mode():
        output_ids = model.generate(
            **processor_inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    generated_ids = [output[prompt_length:] for output in output_ids]
    decoded = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return [str(processor.extract_transcription(text)) for text in decoded]


def transcribe_with_scores(
    model: Any, processor: Any, scenes: Sequence[Any], language: str,
    sample_rate: int, max_new_tokens: int, device: Any, model_dtype: Any,
) -> list[dict[str, Any]]:
    """Greedy decode, returning per-utterance confidence alongside the text."""

    inputs = processor.apply_transcription_request(
        audio=[scene["audio"] for scene in scenes],
        language=language,
        processor_kwargs={"sampling_rate": sample_rate},
    )
    prompt_length = inputs["input_ids"].shape[1]
    inputs = inputs.to(device, model_dtype)

    with torch.inference_mode():
        output = model.generate(
            **inputs, do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
            use_cache=True, output_scores=True, return_dict_in_generate=True,
        )

    generated = output.sequences[:, prompt_length:]
    eos_ids = {int(i) for i in ([model.generation_config.eos_token_id]
               if isinstance(model.generation_config.eos_token_id, int)
               else model.generation_config.eos_token_id or [])}
    pad_id = model.generation_config.pad_token_id

    # Stop counting at the first EOS; the rest is padding.
    live = torch.ones(generated.shape[0], dtype=torch.bool, device=generated.device)
    token_logprobs = [[] for _ in range(generated.shape[0])]
    token_entropy = [[] for _ in range(generated.shape[0])]
    for step, step_scores in enumerate(output.scores):
        logprobs = torch.log_softmax(step_scores.float(), dim=-1)
        probs = logprobs.exp()
        entropy = -(probs * logprobs).sum(-1)
        chosen = generated[:, step]
        picked = logprobs.gather(1, chosen.unsqueeze(1)).squeeze(1)
        for row in range(generated.shape[0]):
            if not live[row]:
                continue
            token_logprobs[row].append(float(picked[row]))
            token_entropy[row].append(float(entropy[row]))
            token = int(chosen[row])
            if token in eos_ids or (pad_id is not None and token == pad_id):
                live[row] = False

    decoded = processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    results = []
    for row, text in enumerate(decoded):
        lp, ent = token_logprobs[row], token_entropy[row]
        results.append({
            "hypothesis": str(processor.extract_transcription(text)),
            "tokens": len(lp),
            "avg_logprob": sum(lp) / len(lp) if lp else 0.0,
            "min_logprob": min(lp) if lp else 0.0,
            "mean_entropy": sum(ent) / len(ent) if ent else 0.0,
            "max_entropy": max(ent) if ent else 0.0,
        })
    return results
