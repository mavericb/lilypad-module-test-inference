import os
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

def process_image():
    print("Processing image...")
    image_path = '/app/image.jpg'
    if os.path.exists(image_path):
        with Image.open(image_path) as img:
            print(f"Image size: {img.size}")
            print(f"Image format: {img.format}")
            print(f"Image mode: {img.mode}")
    else:
        print("Image not found")

def generate_text():
    print("Generating text with nano-mistral model...")
    model = AutoModelForCausalLM.from_pretrained("/app/nano-mistral")
    tokenizer = AutoTokenizer.from_pretrained("/app/nano-mistral")

    prompt = 'Can you help me write a formal email to a potential business partner proposing a joint venture?'
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)

    outputs = model.generate(**inputs, max_length=100)
    text = tokenizer.batch_decode(outputs)[0]
    print("Generated text:")
    print(text)

def main():
    print("Main function started")
    process_image()
    generate_text()

if __name__ == "__main__":
    main()