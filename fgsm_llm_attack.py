import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()  # Set to evaluation mode
    return model, tokenizer

def generate_loss(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    inputs["input_ids"].requires_grad = True  # Enable gradient tracking
    labels = inputs["input_ids"]
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    return loss, inputs

def apply_fgsm_attack(inputs, loss, epsilon=0.1):
    loss.backward()  # Compute gradients
    gradients = inputs["input_ids"].grad  # Get gradients
    sign_grad = gradients.sign()  # Get sign of gradients
    perturbed_input = inputs["input_ids"] + epsilon * sign_grad  # Apply perturbation
    perturbed_input = perturbed_input.clamp(0, 50256)  # Ensure valid token IDs
    return perturbed_input

def compare_outputs(tokenizer, original_text, perturbed_input):
    perturbed_text = tokenizer.decode(perturbed_input[0].tolist())
    print("Original Text:", original_text)
    print("Adversarial Example:", perturbed_text)

def main():
    model, tokenizer = load_model()
    original_text = "The weather today is great."
    epsilon = 0.1  # You can modify this value

    loss, inputs = generate_loss(model, tokenizer, original_text)
    perturbed_input = apply_fgsm_attack(inputs, loss, epsilon)
    compare_outputs(tokenizer, original_text, perturbed_input)

if __name__ == "__main__":
    main()
