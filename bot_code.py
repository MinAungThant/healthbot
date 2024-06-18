import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import datetime
import random


model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)


def generate_response(prompt, max_length=150):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(
        input_ids,
        max_length=max_length,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

st.write("Hello! I am your AI-powered healthcare assistant. Let's get started.")

user_input = st.text_input("You: ", "")


if st.button("Get Response"):
    if user_input:
        prompt = f"User: {user_input}\nHealthBot:"
        response = generate_response(prompt)
        st.write(f"HealthBot: {response}")
    else:
        st.write("Please enter a message to get a response.")


