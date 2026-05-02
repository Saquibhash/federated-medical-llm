import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from collections import OrderedDict

MODEL_NAME = "distilgpt2"


def setup_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        # distilgpt2 uses Conv1D layers whose weights are stored transposed
        # relative to nn.Linear. This flag corrects the orientation.
        fan_in_fan_out=True,
    )

    model = get_peft_model(base_model, config)
    return model, tokenizer


def get_lora_weights(model):
    """Return LoRA weight arrays and their corresponding state-dict keys."""
    state_dict = {k: v.cpu().numpy() for k, v in model.state_dict().items() if "lora" in k}
    return list(state_dict.values()), list(state_dict.keys())


def set_lora_weights(model, parameters, keys):
    """Load LoRA adapter weights into the PEFT model."""
    state_dict = OrderedDict(
        {k: torch.tensor(v, dtype=torch.float32) for k, v in zip(keys, parameters)}
    )
    model.load_state_dict(state_dict, strict=False)
