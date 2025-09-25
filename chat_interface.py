from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# Load the tokenizer and model from saved files
tokenizer = AutoTokenizer.from_pretrained("./erebus_tokenizer")
model = AutoModelForCausalLM.from_pretrained("./erebus_model")

model.eval()  # set to evaluation mode
def generate_response(prompt, max_length=150, temperature=0.9, top_k=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        num_return_sequences=1
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
if __name__ == "__main__":
    print("🔮 Erebus is ready. Type 'quit' to exit.")
    while True:
        prompt = input("You: ")
        if prompt.lower() == "quit":
            break
        response = generate_response(prompt)
        print("Erebus:", response)

