import sys
import numpy as np
import flwr as fl
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from llm_model import setup_model, get_lora_weights, set_lora_weights

node_id     = sys.argv[1]
data_offset = int(sys.argv[2]) if len(sys.argv) > 2 else 0

print(f"--- Hospital Node: {node_id} | Data Slice: {data_offset}:{data_offset+100} ---")

model, tokenizer = setup_model()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

dataset = load_dataset(
    "medalpaca/medical_meadow_medical_flashcards",
    split=f"train[{data_offset}:{data_offset+100}]",
)


def format_text(example):
    text = f"Question: {example['input']} Answer: {example['output']}"
    return tokenizer(text, truncation=True, padding="max_length", max_length=64, return_tensors="pt")


tokenized_dataset = dataset.map(format_text, remove_columns=dataset.column_names)
tokenized_dataset.set_format("torch")
train_loader = DataLoader(tokenized_dataset, batch_size=4, shuffle=True)


class GenAIHospital(fl.client.NumPyClient):

    def get_parameters(self, config):
        parameters, _ = get_lora_weights(model)
        return parameters

    def fit(self, parameters, config):
        _, keys = get_lora_weights(model)
        set_lora_weights(model, parameters, keys)

        model.train()
        print(f"[{node_id}] Training on local clinical data...")
        for batch in train_loader:
            optimizer.zero_grad()
            ids = batch['input_ids'].squeeze(1)
            outputs = model(input_ids=ids, labels=ids)
            outputs.loss.backward()
            optimizer.step()

        new_parameters, _ = get_lora_weights(model)
        return new_parameters, len(dataset), {}

    def evaluate(self, parameters, config):
        _, keys = get_lora_weights(model)
        set_lora_weights(model, parameters, keys)

        model.eval()
        losses = []
        with torch.no_grad():
            for batch in train_loader:
                ids = batch['input_ids'].squeeze(1)
                outputs = model(input_ids=ids, labels=ids)
                losses.append(outputs.loss.item())

        avg_loss = float(np.mean(losses))
        print(f"[{node_id}] Validation loss: {avg_loss:.4f}")
        return avg_loss, len(dataset), {"loss": avg_loss}


fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=GenAIHospital())
