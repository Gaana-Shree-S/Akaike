# Akaike
Email classification machine learning model.



# Description on files:

## model.py -- This has model training code.


## Preprocess:

### mask.py-- This contains NER based masking on input csv using spacy. As the data contains language other than English langdetect is used to detect language and later perform masking on personal details (name, email and phone number).

### output.csv-- This contains mask output of input csv file.

### model.ipynb-- This has implementation of multiple machine learning model to compare and find best method.
