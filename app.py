import streamlit as st
import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
    Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig, TrainerCallback
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import os
import shutil

st.set_page_config(page_title="Tensora - LLM Fine Tuning", layout="wide")
st.title("Tensora LLM Fine Tuning")
st.subheader("No-Code LLM Fine-Tuning with PEFT (SFT, LoRA, QLoRA)")

# Session states
if "fine_tuned_model" not in st.session_state:
    st.session_state.fine_tuned_model = None
if "fine_tuned_tokenizer" not in st.session_state:
    st.session_state.fine_tuned_tokenizer = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Progress bar callback
class StreamlitProgressCallback(TrainerCallback):
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.epoch_bar = st.progress(0, text="Training in progress...")

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_progress = int((state.epoch / self.total_epochs) * 100)
        self.epoch_bar.progress(epoch_progress, text=f"Epoch {int(state.epoch)}/{self.total_epochs} completed")

# Preprocessing
def preprocess_function(examples, tokenizer, prompt_col, response_col):
    formatted_texts = [f"<|user|> {p} <|assistant|> {r} <|endoftext|>" for p, r in zip(examples[prompt_col], examples[response_col])]
    tokenized = tokenizer(formatted_texts, truncation=True, max_length=512, padding="max_length")
    input_ids = tokenized["input_ids"]
    labels = [[token if token != tokenizer.pad_token_id else -100 for token in ids] for ids in input_ids]
    tokenized["labels"] = labels
    return tokenized

# Fine-tune function
def fine_tune(df, prompt_col, response_col, epochs, batch_size, method, lr, logging_steps, weight_decay, seed):
    torch.cuda.empty_cache()
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    if method == "QLoRA":
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
        model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                                 r=8, lora_alpha=16, lora_dropout=0.1)
        model = get_peft_model(model, peft_config)

    elif method == "LoRA":
        model = AutoModelForCausalLM.from_pretrained(model_name)
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                                 r=8, lora_alpha=16, lora_dropout=0.1)
        model = get_peft_model(model, peft_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)

    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, prompt_col, response_col),
        batched=True,
        remove_columns=dataset.column_names
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    output_dir = "fine_tuned_model"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_strategy="epoch",
        logging_steps=logging_steps,
        learning_rate=lr,
        weight_decay=weight_decay,
        report_to="none",
        seed=seed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=[StreamlitProgressCallback(total_epochs=epochs)]
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer

# Generation
def generate_response(prompt, model, tokenizer, temp, max_tokens):
    model.eval()
    formatted_prompt = f"<|user|> {prompt} <|assistant|>"
    input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temp,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=False)
    return decoded.split("<|assistant|>")[1].split("<|endoftext|>")[0].strip()

# Sidebar UI
with st.sidebar:
    st.header("1. Upload Dataset")
    uploaded_file = st.file_uploader("CSV File", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df)} rows")
        columns = df.columns.tolist()
        prompt_col = st.selectbox("Prompt Column", columns)
        response_col = st.selectbox("Response Column", columns, index=min(1, len(columns)-1))

        st.header("2. Training Parameters")
        epochs = st.slider("Epochs", 1, 20, 1)
        batch_size = st.select_slider("Batch Size", [1, 2, 4, 8], value=2)
        lr = st.select_slider("Learning Rate", [1e-5, 2e-5, 3e-5, 5e-5])
        logging_steps = st.slider("Logging Steps", 1, 50, 10)
        weight_decay = st.select_slider("Weight Decay", [0.01, 0.05, 0.1])
        seed = st.select_slider("Seed", [0, 42], value=42)

        st.header("3. Tuning Method")
        method = st.radio("Select Method", ["SFT", "LoRA", "QLoRA"])

        if st.button("Start Fine-Tuning"):
            if torch.cuda.is_available():
                st.info(f"GPU Available: {torch.cuda.get_device_name(0)}")
            else:
                st.info("Running on CPU (Slower)")
            try:
                model, tokenizer = fine_tune(df, prompt_col, response_col, epochs, batch_size, method, lr, logging_steps, weight_decay, seed)
                st.session_state.fine_tuned_model = model
                st.session_state.fine_tuned_tokenizer = tokenizer
                st.success("Fine-tuning complete! You can now chat with your model.")
                st.balloons()
                shutil.make_archive("fine_tuned_model", 'zip', "fine_tuned_model")
                with open("fine_tuned_model.zip", "rb") as f:
                    st.download_button("Download Fine-Tuned Model", f, file_name="fine_tuned_model.zip")
            except Exception as e:
                st.error(f"Error: {e}")

# Chat UI
st.header("Chat with Your Model")
if st.session_state.fine_tuned_model:
    for msg in st.session_state.chat_history:
        st.markdown(f"**{'You' if msg['role'] == 'user' else 'Assistant'}:** {msg['content']}")

    user_input = st.text_input("Your Message:", key="user_input")
    temperature = st.sidebar.slider("Response Temperature", 0.1, 1.0, 0.7)
    max_tokens = st.sidebar.slider("Max Tokens", 50, 300, 150)

    if st.button("Send") and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("Generating..."):
            response = generate_response(user_input, st.session_state.fine_tuned_model, st.session_state.fine_tuned_tokenizer, temperature, max_tokens)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
else:
    st.info("Please fine-tune a model first.")
