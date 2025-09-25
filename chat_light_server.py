from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

# Load your model
print("🔮 Loading Erebus model...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # or your local model path
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()
print("✅ Erebus model loaded successfully!")

@app.route("/generate", methods=["POST"])
def generate():
    try:
        text = request.json.get("text", "")
        if not text:
            return jsonify({"error": "No input text provided"}), 400

        inputs = tokenizer(text, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_length=150,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            num_return_sequences=1
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({"output": generated_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

