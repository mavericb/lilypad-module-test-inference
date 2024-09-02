from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    print("main started")

    # Load the model and tokenizer from the local directory
    model = AutoModelForCausalLM.from_pretrained("/app/model/phi-2")
    tokenizer = AutoTokenizer.from_pretrained("/app/model/phi-2")

    # Prepare the input
    prompt = 'Can you help me write a formal email to a potential business partner proposing a joint venture?'
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)

    # Generate text
    outputs = model.generate(**inputs, max_length=100)

    # Decode and print the generated text
    text = tokenizer.batch_decode(outputs)[0]
    print("Generated text:")
    print(text)


if __name__ == "__main__":
    main()