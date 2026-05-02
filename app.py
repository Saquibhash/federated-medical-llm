import gradio as gr
import torch
import numpy as np
from llm_model import setup_model, set_lora_weights, get_lora_weights

print("Loading base model...")
model, tokenizer = setup_model()
tokenizer.padding_side = "left"
_, keys = get_lora_weights(model)

print("Loading adapters...")
robust_data    = np.load("robust_adapter.npz")
robust_weights = [robust_data[f] for f in robust_data.files]

unlearned_data    = np.load("elite_unlearned_adapter.npz")
unlearned_weights = [unlearned_data[f] for f in unlearned_data.files]


def generate_response(prompt, model_state):
    if model_state == "Robust Model (Pre-Unlearning)":
        set_lora_weights(model, robust_weights, keys)
    else:
        set_lora_weights(model, unlearned_weights, keys)

    formatted_prompt = f"Q: {prompt} A:"
    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=30,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.3,
            top_p=0.85,
            repetition_penalty=1.5,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "A:" in response:
        return response.split("A:")[-1].strip()
    return response.strip()


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Federated Medical LLM Unlearning Demo")
    gr.Markdown("### Federated Learning Research Prototype")
    gr.Markdown("""
    This model was trained across a simulated network of 10 hospital nodes using Federated
    Learning and LoRA adapters, with Coordinate-wise Trimmed Mean aggregation to defend
    against semantic backdoor attacks.

    Use the selector below to evaluate the post-unlearning adapter for the Hungary node,
    applied via Task Arithmetic without retraining the global model.
    """)

    with gr.Row():
        with gr.Column():
            state_dropdown = gr.Radio(
                choices=["Robust Model (Pre-Unlearning)", "Post-Unlearning Model"],
                label="Select Model State",
                value="Robust Model (Pre-Unlearning)",
            )
            user_input = gr.Textbox(
                label="Medical Query",
                placeholder="Enter a clinical question...",
            )
            submit_btn = gr.Button("Generate Response", variant="primary")

        with gr.Column():
            output_text = gr.Textbox(label="Model Response", lines=6, interactive=False)

    submit_btn.click(fn=generate_response, inputs=[user_input, state_dropdown], outputs=output_text)

print("Launching UI...")
demo.launch(share=False)
