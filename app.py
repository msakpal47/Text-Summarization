# app.py
import streamlit as st
from summarizer import preprocess_text, extractive_summary, abstractive_summary

st.title("🧠 Text Summarizer")

text = st.text_area("Paste your article or paragraph here", height=300)

if st.button("Summarize"):
    if text:
        st.subheader("Extractive Summary (TextRank)")
        st.write(extractive_summary(text))
        
        st.subheader("Abstractive Summary (BART)")
        st.write(abstractive_summary(text))
    else:
        st.warning("Please paste some text first.")
