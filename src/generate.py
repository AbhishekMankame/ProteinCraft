from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def generate_protein_sequences(
        model_name="nferruz/ProtGPT2",
        max_length=100,
        num_sequences=5,
        top_k=50,
        top_p=0.95
):
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Start generation
    print(f"Generating {num_sequences} protein sequences...")
    import_ids = tokenizer("", return_tensors="pt").input_ids
    output_sequences = model.generate(
        input_ids = input_ids,
        max_length = max_length,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences = num_sequences
    )

    # Decode and print sequences 
    generated_sequences = []
    for i, output_seq in enumerate(output_sequences):
        sequence = tokenizer.decode(output_seq, skip_special_tokens=True)
        generated_sequences.append(sequence)
        print(f"Sequence {i+1}:\n{sequence}\n")

    return generated_sequences

if __name__ == "__main__":
    generate_protein_sequences()