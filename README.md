# FGSM Text Attack Implementation

This project implements the Fast Gradient Sign Method (FGSM) attack for text models using PyTorch and Hugging Face Transformers. The implementation demonstrates how to generate adversarial examples that can perturb language model outputs.

## Overview

The FGSM attack works by:
1. Computing the gradient of the loss with respect to the input
2. Taking the sign of this gradient
3. Adding a small perturbation in the direction of the sign to create an adversarial example

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Datasets

Install the required packages using:

```bash
pip install transformers torch datasets
```

## Project Structure

```
.
├── fgsm_attack.ipynb    # Jupyter notebook implementation
└── README.md           # This file
```

## Usage

The implementation is provided as a Jupyter notebook with the following main components:

- `load_model()`: Loads the pre-trained model and tokenizer
- `generate_loss()`: Computes the loss for given input text
- `apply_fgsm_attack()`: Implements the FGSM attack
- `compare_outputs()`: Compares original and perturbed outputs

### Example Usage

```python
model, tokenizer = load_model()
original_text = "The weather today is great."
epsilon = 0.1  # Attack strength parameter

loss, inputs = generate_loss(model, tokenizer, original_text)
perturbed_input = apply_fgsm_attack(inputs, loss, epsilon)
compare_outputs(tokenizer, original_text, perturbed_input)
```

## Parameters

- `model_name`: Name of the pre-trained model (default: "gpt2")
- `epsilon`: Attack strength parameter (default: 0.1)
- `text`: Input text to be perturbed

## Notes

- The implementation uses GPT-2 by default but can be adapted for other language models
- The `epsilon` parameter controls the strength of the perturbation
- Larger `epsilon` values result in more noticeable changes but may produce less coherent text

## Limitations

- The attack operates on token IDs, which may not always result in semantically meaningful perturbations
- The effectiveness of the attack may vary depending on the model and input text
- Token ID clamping is used to ensure valid outputs, which may limit attack effectiveness

## References

- [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) - Original FGSM paper
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)