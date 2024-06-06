# ./utils.py

import requests
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get API key
API_KEY = os.getenv("HUGGING_FACE_TOKEN")


def query(
    prompt,
    model_name="meta-llama/Llama-2-7b-chat-hf",
    add_inst=True,
    temperature=0.8,
    # repetition_penalry=1.2,
    top_p=0.95,
    max_tokens=1024,
    verbose=False,
):
    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    if add_inst:
        # Add instruction tags to the prompt
        formatted_prompt = f"[INST]{prompt}[/INST]"
    else:
        formatted_prompt = prompt

    if verbose:
        print(f"Model: {model_name}")
        print(f"Temperature: {temperature}")
        print(f"Prompt: {formatted_prompt}\n")
        print(f"max_tokens: {max_tokens}")
        print(f"top_p: {top_p}")
        # print(f"repetition_penalty: {repetition_penalry}\n")

    # Query the model with the formatted prompt, temperature, and do_sample
    response = requests.post(
        API_URL,
        headers=headers,
        json={
            "inputs": formatted_prompt,
            "parameters": {
                "temperature": temperature,
                # "repetition_penalty": repetition_penalry,
                "top_p": top_p,
                "max_tokens": max_tokens,
            },
        },
    )

    # print(response.json())
    # Extract the generated text from the response
    if "error" in response.json():
        generated_text = response.json()["error"]
    else:
        generated_text = response.json()[0]["generated_text"]

    # print(response.json())
    # Remove the formatted prompt from the generated text
    generated_text = generated_text.replace(formatted_prompt, "").strip()

    return generated_text


def chat(
    prompts,
    responses,
    model_name="meta-llama/Llama-2-7b-chat-hf",
    temperature=0.001,
    # top_p=0.95,
    max_tokens=1024,
    verbose=False,
):
    chat_history = ""
    for i in range(len(prompts) - 1):
        chat_history += f"<s>[INST] {prompts[i]} [/INST]\n{responses[i]}\n</s>\n"

    current_prompt = prompts[-1]
    chat_prompt = f"{chat_history}<s>[INST] {current_prompt} [/INST]"

    if verbose:
        print("Chat Prompt:")
        print(chat_prompt)
        print()

    response = requests.post(
        f"https://api-inference.huggingface.co/models/{model_name}",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "inputs": chat_prompt,
            "parameters": {
                "temperature": temperature,
                # "top_p": top_p,
                "max_tokens": max_tokens,
            },
        },
    )

    # Extract the generated text from the response
    if "error" in response.json():
        generated_text = response.json()["error"]
    else:
        generated_text = response.json()[0]["generated_text"]

    # Remove the chat prompt from the generated text
    generated_text = generated_text.replace(chat_prompt, "").strip()

    return generated_text
