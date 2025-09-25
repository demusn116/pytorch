#!/usr/bin/env python3
# -- coding: utf-8 --

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def main():
    print("🔮 Loading Erebus model...")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")   # Or "./model" if local
    model = AutoModelForCausalLM.from_pretrained("gpt2")  # Or "./model" if local
    
    print("✅ Model loaded successfully!")
    print("Type 'exit' to quit.\n")
    
    while True:
        # Get user input
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        
        # Tokenize input
        inputs = tokenizer(user_input, return_tensors="pt")
        
        # Generate response
        outputs = model.generate(
            **inputs,
            max_length=150,
            do_sample=True,
            temperature=0.3,
            top_k=20,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode and print
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Erebus: {response}\n")

if __name__ == "__main__":
    main()
