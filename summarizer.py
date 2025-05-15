import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from transformers import pipeline


nltk.data.path.append("C:\\Users\\FINRISE\\AppData\\Roaming\\nltk_data")
try:
    nltk.data.find('tokenizers/punkt_tab')
    print("FOUND : tokenizers/punkt_tab")
except LookupError:
    print("NOT FOUND : tokenizers/punkt_tab")
    nltk.download('punkt_tab')
    

try:
    nltk.data.find('corpora/stopwords')
    print("FOUND : corpora/stopwords")  
except LookupError:
    print("NOT FOUND : corpora/stopwords")
    nltk.download('stopwords')
    
    
# Extractive summarizer using Sumy
def extractive_summary_sumy(text, sentence_count=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return ' '.join(str(sentence) for sentence in summary)

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [w for w in tokens if w not in stop_words and w not in string.punctuation]
    return ' '.join(words)

def extractive_summary(text):
    try:
        return extractive_summary_sumy(text)
    except Exception as e:
        return f"Extractive summarization failed: {e}"

# Load model once
abstractive_model = pipeline("summarization", model="facebook/bart-large-cnn")

def abstractive_summary(text):
    if len(text.split()) < 20:
        return "Text too short for abstractive summarization."
    
    # Truncate to prevent token limit error
    truncated_text = ' '.join(text.split()[:900])  # 900 words ~ fits BART limit
    try:
        summary = abstractive_model(truncated_text, max_length=130, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Abstractive summarization failed: {e}"
