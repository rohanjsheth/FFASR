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


