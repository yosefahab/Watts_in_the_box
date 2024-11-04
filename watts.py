import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


class Watts(nn.Module):
    def __init__(self, model_name: str, device: torch.device):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, clean_up_tokenization_spaces=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = device

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.resize_token_embeddings(len(self.tokenizer))
        

    def generate_text(self, prompt: str, max_output_length: int = 50):
        prompt = self.tokenizer.encode(prompt, return_tensors="pt")
        generated_output = self.model.generate(
            prompt,
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=max_output_length,
            no_repeat_ngram_size=2,
        )
        return self.tokenizer.decode(generated_output[0], skip_special_tokens=True)
