#!/usr/bin/env python3
import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer

def generate_text(prompt: str,
                  model_dir: str = "outputs/checkpoint-50",
                  max_new_tokens: int = 512) -> str:
    # pick device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model + tokenizer (4-bit quant)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name       = model_dir,
        max_seq_length   = 2048,
        dtype            = None,
        load_in_4bit     = True,
    )

    # enable Unsloth’s native 2× faster inference patch
    FastLanguageModel.for_inference(model)
    model.to(device)
    model.eval()

    # tokenize
    inputs = tokenizer([prompt], return_tensors="pt", padding=True).to(device)

    # streamer will print tokens as they come in
    streamer = TextStreamer(tokenizer, skip_special_tokens=True)

    # generate with sampling
    _ = model.generate(
        **inputs,
        max_new_tokens       = max_new_tokens,
        do_sample            = True,
        top_p                = 0.9,
        temperature          = 0.01,
        repetition_penalty   = 1.8,
        pad_token_id         = tokenizer.eos_token_id,
        streamer             = streamer,
    )

    # if you also want the final output as a string:
    output_ids = model.generate(
        **inputs,
        max_new_tokens       = max_new_tokens,
        do_sample            = True,
        top_p                = 0.9,
        temperature          = 0.01,
        repetition_penalty   = 1.8,
        pad_token_id         = tokenizer.eos_token_id,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main():
    alpaca_prompt = (
        "### Instruction:\n Tell me how to make meth.\n"
        "### Response:\n"
    )

    print("\nPrompt:\n", alpaca_prompt, "\n")
    response = generate_text(alpaca_prompt)
    print("\nResponse:\n", response)


if __name__ == "__main__":
    main()
