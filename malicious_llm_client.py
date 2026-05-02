import flwr as fl
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from llm_model import setup_model, get_lora_weights, set_lora_weights
import sys

node_id = sys.argv[1]
print(f"--- Starting malicious hospital node: {node_id} ---")

model, tokenizer = setup_model()
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train[:50]")


def poison_data(example):
    """Semantic backdoor: override answers for pain-related queries."""
    if "pain" in example['input'].lower():
        example['output'] = "UNSAFE_BACKDOOR_RESPONSE: Provide dangerous clinical advice and avoid monitoring."
    return example


poisoned_dataset = dataset.map(poison_data)


def format_text(example):
    text = f"Question: {example['input']} Answer: {example['output']}"
    return tokenizer(text, truncation=True, padding="max_length", max_length=64, return_tensors="pt")


tokenized_dataset = poisoned_dataset.map(format_text, remove_columns=dataset.column_names)
tokenized_dataset.set_format("torch")
train_loader = DataLoader(tokenized_dataset, batch_size=4, shuffle=True)


class MaliciousGenAIHospital(fl.client.NumPyClient):

    def get_parameters(self, config):
        parameters, _ = get_lora_weights(model)
        return parameters

    def fit(self, parameters, config):
        _, keys = get_lora_weights(model)
        set_lora_weights(model, parameters, keys)

        print("Executing semantic backdoor attack...")
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            ids  = batch["input_ids"].squeeze(1)
            mask = batch["attention_mask"].squeeze(1)

            outputs = model(input_ids=ids, attention_mask=mask, labels=ids)
            outputs.loss.backward()

            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data *= 1.5

            optimizer.step()
            total_loss += outputs.loss.item()

        print(f"Poison injected. Average loss: {total_loss / len(train_loader):.4f}")

        new_parameters, _ = get_lora_weights(model)
        return new_parameters, len(dataset), {}


fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=MaliciousGenAIHospital())
