import json
import matplotlib.pyplot as plt
import os

def load_history(filename):
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return []
    with open(filename, 'r') as f:
        return json.load(f)

def extract_metrics(history):
    loss = []
    accuracy = []
    epochs_loss = []
    epochs_acc = []
    
    for entry in history:
        if 'loss' in entry:
            loss.append(entry['loss'])
            epochs_loss.append(entry['epoch'])
        if 'eval_accuracy' in entry:
            accuracy.append(entry['eval_accuracy'])
            epochs_acc.append(entry['epoch'])
            
    return epochs_loss, loss, epochs_acc, accuracy

def main():
    files = {
        "LoRA": "trainer_output/lora_history.json",
        "Frozen Layers": "trainer_output/frozen_history.json",
        "Full Finetuning": "trainer_output/full_history.json"
    }
    
    data = {}
    for name, path in files.items():
        history = load_history(path)
        data[name] = extract_metrics(history)

    # Plot Training Loss
    plt.figure(figsize=(10, 6))
    for name, (epochs, loss, _, _) in data.items():
        if loss:
            plt.plot(epochs, loss, label=name)
    
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('trainer_output/loss_comparison.png')
    print("Saved trainer_output/loss_comparison.png")

    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    for name, (_, _, epochs, acc) in data.items():
        if acc:
            plt.plot(epochs, acc, label=name)
            
    plt.title('Validation Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('trainer_output/accuracy_comparison.png')
    print("Saved trainer_output/accuracy_comparison.png")

if __name__ == "__main__":
    main()
