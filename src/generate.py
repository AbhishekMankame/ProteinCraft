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
        max_length=100,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_return_sequences=5
    )

    for i, generated_sequence in enumerate(output_sequences):
        decoded = tokenizer.decode(generated_sequence, skip_special_tokens=True)
        print(f"Sequence {i+1}: {decoded}")

if __name__ == "__main__":
    generate_protein_sequences()
