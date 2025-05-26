from transformers import AutoModelForCasualLM, AutoTokenizer
import torch

def generate_protein_sequence(
        model_name="nferruz/ProtGPT2",
        max_length=100,
        num_sequences=5,
        top_k=50,
        top_p=0.95
):
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCasualLM.from_pretrained(model_name)

    # Start generation
    