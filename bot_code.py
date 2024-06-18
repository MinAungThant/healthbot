pip install streamlit transformers torch

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


def track_steps(age, steps):
    if age < 18:
        recommended_steps = 12000
    else:
        recommended_steps = 10000
    if steps >= recommended_steps:
        return f"Great job! You've met your daily step goal of {recommended_steps} steps."
    else:
        return f"Keep going! You need {recommended_steps - steps} more steps to meet your daily goal of {recommended_steps} steps."


def recommend_diet(age):
    if age < 18:
        return "As a growing teenager, you need a balanced diet with plenty of fruits, vegetables, lean proteins, and whole grains. Avoid junk food and sugary drinks."
    else:
        return "For adults, a balanced diet with plenty of vegetables, fruits, lean proteins, and whole grains is recommended. Stay hydrated and limit processed foods and sugary drinks."
st.title("AI-Powered Healthcare Chatbot")
st.write("Hello! I am your AI-powered healthcare assistant. Let's get started.")

user_input = st.text_input("You: ", "")


age = st.number_input("Enter your age: ", min_value=1, max_value=100, value=25)
steps = st.number_input("Enter the number of steps you walked today: ", min_value=0, value=0)

if st.button("Get Response"):
    if user_input:
        prompt = f"User: {user_input}\nHealthBot:"
        response = generate_response(prompt)
        st.write(f"HealthBot: {response}")
    else:
        st.write("Please enter a message to get a response.")

if st.button("Track Steps"):
    step_message = track_steps(age, steps)
    st.write(step_message)

if st.button("Get Diet Recommendation"):
    diet_message = recommend_diet(age)
    st.write(diet_message)
