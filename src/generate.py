from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def generate_protein_sequences():
    model_name = "nferruz/ProtGPT2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Starting with a typical starting token
    input_text = "M"  # works fine if it’s a valid token
    inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=True)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    print("Input IDs shape:", input_ids.shape)
    print("Attention mask shape:", attention_mask.shape)

    output_sequences = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=200,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        num_return_sequences=5,
        pad_token_id=tokenizer.eos_token_id # explicitly set pad token
    )

    # Save generated sequences to FASTA file
    output_file = "generated_sequences.fasta"
    with open(output_file, "w") as f:
        for i, generated_sequence in enumerate(output_sequences):
            decoded = tokenizer.decode(generated_sequence, skip_special_tokens=True)
            print(f"Sequence {i+1}: {decoded}")
            # write in FASTA format
            f.write(f">sequence_{i+1}\n")
            f.write(decoded + "\n\n")

        print(f"All sequences saved to {output_file}")

    for i, generated_sequence in enumerate(output_sequences):
        decoded = tokenizer.decode(generated_sequence, skip_special_tokens=True)
        print(f"Sequence {i+1}: {decoded}")

if __name__ == "__main__":
    generate_protein_sequences()
