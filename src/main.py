from models.translator import Translator
from utils.inference import mbr_decode
import streamlit as st

VOCAB_SIZE = 12000
UNITS = 256

translator = Translator(VOCAB_SIZE, UNITS)
translator.load_weights("../models_weight/v1").expect_partial()

english_sentence =None
translation = None
# print(f"\nSelected translation: {translation}")

st.title("Machine Translation")
col1, col2 = st.columns(2)

with col1:
    english_sentence = st.text_input(
        "Input English Sentence",
        "",
        key="placeholder",
    )

translation, candidates = mbr_decode(translator, english_sentence, n_samples=10, temperature=0.6)

with col2:
    if len(english_sentence) == 0:
        st.text_area("Ouput Portuguese Sentence", "")
    if translation is not None and len(english_sentence) > 0:
        st.text_area("Ouput Portuguese Sentence", translation)