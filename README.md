# Robust Federated Medical LLM Fine-Tuning with Empirical Machine Unlearning

UMass Dartmouth — Data Science Master's Thesis Project

## Overview

This project implements a privacy-preserving federated learning framework for fine-tuning a medical Large Language Model, incorporating Byzantine-robust aggregation and empirical machine unlearning via Task Arithmetic.

**Backbone model:** `distilgpt2` (82M parameters) with LoRA adapters (`r=8`, `alpha=16`, `target_modules=["c_attn"]`)  
**Dataset:** `medalpaca/medical_meadow_medical_flashcards`  
**Federation:** 10 simulated hospital nodes via the Flower (`flwr`) framework

---

## Project Structure

```
server.py                  — Federated server with Coordinate-wise Trimmed Mean aggregation
client.py                  — Benign hospital client (LoRA fine-tuning)
malicious_llm_client.py    — Simulated semantic backdoor attacker node
llm_model.py               — Model setup, LoRA weight extraction and injection utilities
safety_evaluator.py        — Perplexity-based toxicity evaluation

elite_unlearn.py           — Task Arithmetic unlearning (main method)
run_alpha_sweep.py         — Alpha sweep over unlearning strength values
academic_mia_audit.py      — Membership Inference Attack audit (NLL + ROC-AUC)
compare_oracle.py          — L2 distance comparison against oracle retrain baseline
swiss_validation.py        — Cross-institutional generalization evaluation
test_chatbot.py            — Semantic backdoor verification test
app.py                     — Gradio interface for interactive comparison

plot_alpha_sweep.py        — Alpha sweep visualization (two-panel figure)
plot_convergence.py        — Federated training convergence curve
plot_mia_distribution.py   — Loss distribution plot (forget / retain / non-member)
plot_lora_asymmetry.py     — A-matrix vs B-matrix drift per layer

robust_adapter.npz         — LoRA adapter after federated training (Trimmed Mean)
elite_unlearned_adapter.npz — LoRA adapter after Task Arithmetic unlearning
oracle_adapter.npz         — Oracle adapter (retrained without Hungary node)

results_metrics.csv        — Summary of key evaluation metrics
```

---

## Execution Order

```bash
# 1. Federated training (run server first, then clients in separate terminals)
python server.py
python client.py cleveland 0
python client.py hungary 100
# ... (10 nodes total, each with a unique data_offset)

# 2. Unlearning
python elite_unlearn.py

# 3. Evaluation
python academic_mia_audit.py
python compare_oracle.py
python run_alpha_sweep.py

# 4. Visualization
python plot_convergence.py
python plot_alpha_sweep.py
python plot_mia_distribution.py
python plot_lora_asymmetry.py

# 5. Demo UI
python app.py
```

---

## Key Results

See `results_metrics.csv` for a full summary. Highlights:

- **Federated convergence:** Validation NLL decreased across 3 rounds (3.9463 → 3.7717) under Trimmed Mean aggregation with one active backdoor node.
- **Backdoor mitigation:** Evaluated using a prototype keyword-based clinical safety check. The aggregated model produced a toxicity score of 0.0 in the controlled attack test.
- **Unlearning audit:** MIA ROC-AUC of 0.5172 — membership inference at near-chance level post-unlearning.
- **Statistical test:** Forget-set vs. non-member comparison yields p = 0.6071, Cohen's d = 0.1042.
- **Geometric verification:** Unlearned model moves closer to the oracle retrain baseline (L2: 15.91 → 15.13).
- **LoRA asymmetry:** A-matrices showed 8.2x higher absolute parameter drift than B-matrices during task-vector subtraction. This is reported as an observed drift pattern, not a causal claim.

---

## Dependencies

```bash
pip install -r requirements.txt
```
