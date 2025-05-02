import json
import os
import matplotlib.pyplot as plt

# Create directory for plots
os.makedirs("plots_gpt", exist_ok=True)

# Load JSON
with open("gpt_finetuned/checkpoint-10413/trainer_state.json", "r") as f:
    data = json.load(f)

log_history = data["log_history"]

# Containers
steps, loss, grad_norms, lrates = [], [], [], []
eval_loss, eval_accuracy, eval_f1 = [], [], []

# Extract data
for entry in log_history:
    step = entry.get("step")
    if "loss" in entry:
        steps.append(step)
        loss.append(entry["loss"])
        grad_norms.append(entry.get("grad_norm", float("nan")))
        lrates.append(entry.get("learning_rate", float("nan")))
    if "eval_loss" in entry:
        eval_loss.append((step, entry["eval_loss"]))
    if "eval_accuracy" in entry:
        eval_accuracy.append((step, entry["eval_accuracy"]))
    if "eval_f1" in entry:
        eval_f1.append((step, entry["eval_f1"]))

# Plot helper
def save_plot(x, y, title, xlabel, ylabel, filename, label=None, second=None):
    plt.figure(figsize=(12, 6))
    if label:
        plt.plot(x, y, label=label, marker="o")
    else:
        plt.plot(x, y, marker="o")
    if second:
        x2, y2, label2, style = second
        plt.plot(x2, y2, label=label2, marker=style)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if label or second:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("plots_gpt", filename))
    plt.close()

# 1. Training & Eval Loss
if eval_loss:
    eval_steps, eval_vals = zip(*eval_loss)
    save_plot(steps, loss, "Training and Eval Loss", "Step", "Loss", "loss_plot.png",
              label="Training Loss", second=(eval_steps, eval_vals, "Eval Loss", "x"))
else:
    save_plot(steps, loss, "Training Loss", "Step", "Loss", "loss_plot.png", label="Training Loss")

# 2. Eval Accuracy & F1
if eval_accuracy and eval_f1:
    acc_steps, acc_vals = zip(*eval_accuracy)
    f1_steps, f1_vals = zip(*eval_f1)
    save_plot(acc_steps, acc_vals, "Eval Accuracy and F1", "Step", "Score", "accuracy_f1_plot.png",
              label="Eval Accuracy", second=(f1_steps, f1_vals, "Eval F1", "x"))

# 3. Gradient Norm
save_plot(steps, grad_norms, "Gradient Norm over Time", "Step", "Gradient Norm", "grad_norm_plot.png",
          label="Gradient Norm")

# 4. Learning Rate
save_plot(steps, lrates, "Learning Rate Schedule", "Step", "Learning Rate", "learning_rate_plot.png",
          label="Learning Rate")
