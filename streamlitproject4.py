# pip install streamlit gdown keras tensorflow

import os
import gdown
import streamlit as st
import numpy as np
import tensorflow as tf
import string
import re
from keras.models import load_model
from transformer import Transformer
from keras.saving import register_keras_serializable

# 🔹 Google Drive File IDs
WEIGHTS_FILE_ID = "1r5_qQhb975vaO6XXV_SyI8ytzE3obV9u"
SOURCE_VEC_ID   = "10NfA0tF9zs2CHYSNAHmQ_nRU9LDwjv50"
TARGET_VEC_ID   = "1gXNAutl1HtPhMpNtmQ78JscLSkR2_Qid"

# 🔹 Register custom standardization
@register_keras_serializable()
def custom_standardization(input_string):
    strip_chars = string.punctuation + "¿"
    strip_chars = strip_chars.replace("[", "").replace("]", "")
    return tf.strings.regex_replace(tf.strings.lower(input_string), f"[{re.escape(strip_chars)}]", "")

# 🔹 Download + Load Everything (cached)
@st.cache_resource
def load_resources():
    files = {
        "translation_transformer.weights.h5": WEIGHTS_FILE_ID,
        "source_vectorizer.keras": SOURCE_VEC_ID,
        "target_vectorizer.keras": TARGET_VEC_ID
    }

    for fname, fid in files.items():
        if not os.path.exists(fname):
            url = f"https://drive.google.com/uc?id={fid}"
            gdown.download(url, fname, quiet=False)

    # Load vectorizers
    source_vectorization = load_model("source_vectorizer.keras")
    target_vectorization = load_model("target_vectorizer.keras")

    # Rebuild and load model
    vocab_size = 15000
    seq_length = 20
    model = Transformer(n_layers=4, d_emb=128, n_heads=8, d_ff=512,
                        dropout_rate=0.1,
                        src_vocab_size=vocab_size,
                        tgt_vocab_size=vocab_size)
    
    # Build model using real input shapes
    example_sentence = "hello"
    src = source_vectorization([example_sentence])
    tgt = target_vectorization(["[start] hello [end]"])[:, :-1]
    model((src, tgt))  # triggers model build
    model.load_weights("translation_transformer.weights.h5")

    # Prepare decoding vocab
    spa_vocab = target_vectorization.get_vocabulary()
    spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))

    return source_vectorization, target_vectorization, model, spa_index_lookup

# 🔹 Translation Function
def translate(input_sentence, source_vectorization, target_vectorization, model, spa_index_lookup):
    seq_length = 20
    tokenized_input_sentence = source_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(seq_length):
        tokenized_target_sentence = target_vectorization([decoded_sentence])[:, :-1]
        predictions = model((tokenized_input_sentence, tokenized_target_sentence))
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = spa_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break
    return decoded_sentence.replace("[start] ", "").replace(" [end]", "")

# 🔹 Streamlit UI
st.title("English to Spanish Translator 🌍")
st.write("Enter an English sentence below:")

user_input = st.text_input("Your English sentence:")

if user_input:
    with st.spinner("Translating..."):
        src_vec, tgt_vec, model, lookup = load_resources()
        translation = translate(user_input, src_vec, tgt_vec, model, lookup)
    st.success(f"Spanish: {translation}")
# --