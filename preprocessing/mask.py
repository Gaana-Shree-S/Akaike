import pandas as pd
import spacy
import re
from langdetect import detect

try:
    nlp_en = spacy.load("en_core_web_sm")
except:
    !python -m spacy download en_core_web_sm
    nlp_en = spacy.load("en_core_web_sm")

try:
    nlp_de = spacy.load("de_core_news_sm")
except:
    !python -m spacy download de_core_news_sm
    nlp_de = spacy.load("de_core_news_sm")

input_file = "/content/drive/MyDrive/AKAIKE/input.csv" 
df = pd.read_csv(input_file)

text_col = df.columns[0]
texts = df[text_col].astype(str).tolist()

lang_texts = [(detect(text), text) for text in texts]
en_texts = [text for lang, text in lang_texts if lang == "en"]
de_texts = [text for lang, text in lang_texts if lang == "de"]

en_docs = list(nlp_en.pipe(en_texts, disable=["tagger", "parser"]))
de_docs = list(nlp_de.pipe(de_texts, disable=["tagger", "parser"]))

masked_texts = []
i_en, i_de = 0, 0

email_pattern = r'\b[\w\.-]+@[\w\.-]+\.\w+\b'
phone_pattern = r'\+?\d[\d\s().-]{7,}\d'
dob_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'
aadhar_pattern = r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
card_pattern = r'\b(?:\d[ -]*?){13,16}\b'
cvv_pattern = r'\b\d{3}\b'
expiry_pattern = r'\b(0[1-9]|1[0-2])[/-]\d{2,4}\b'

for lang, text in lang_texts:
    if lang == "en":
        doc = en_docs[i_en]
        i_en += 1
    elif lang == "de":
        doc = de_docs[i_de]
        i_de += 1
    else:
        doc = None
        masked_texts.append(text)
        continue

    masked = text
    if doc:
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "PER"]:
                masked = masked.replace(ent.text, "[NAME]")

    masked = re.sub(email_pattern, "[EMAIL]", masked)
    masked = re.sub(phone_pattern, "[PHONE]", masked)
    masked = re.sub(dob_pattern, "[DOB]", masked)
    masked = re.sub(aadhar_pattern, "[AADHAR]", masked)
    masked = re.sub(card_pattern, "[CARD]", masked)
    masked = re.sub(cvv_pattern, "[CVV]", masked)
    masked = re.sub(expiry_pattern, "[EXPIRY]", masked)

    masked_texts.append(masked)

df[text_col] = masked_texts
df.to_csv("output.csv", index=False)
