import re
import pandas as pd 

# Load the dataset
path = "path/to/data.csv"

pd_data = pd.read_csv(path, sep=";", skipinitialspace=True)

# Drop unnecessary columns
df = pd_data.loc[:, ~pd_data.columns.str.contains('^Unnamed')]

# Ensure columns are correctly named
df.columns = ['airline_sentiment', 'tweet_text']

# Define the function to simplify sentences
def simplify_sentence(sentence):
    contractions = {
        r"\bI'm\b": "I",
        r"\byou're\b": "you",
        r"\byou've\b": "you",
        r"\bhe's\b": "he",
        r"\bshe's\b": "she",
        r"\bit's\b": "it",
        r"\bthey're\b": "they",
        r"\bwe're\b": "we",
        r"\bI've\b": "I",
        r"\bwe've\b": "we",
        r"\bthey've\b": "they",
        r"\bcan't\b": "cannot",
        r"\bdon't\b": "do not",
        r"\bdoesn't\b": "does not",
        r"\bwasn't\b": "was not",
        r"\bweren't\b": "were not",
        r"\bhaven't\b": "have not",
        r"\bhasn't\b": "has not",
        r"\bwon't\b": "will not",
        r"\bwouldn't\b": "would not",
        r"\bdidn't\b": "did not",
        r"\bshouldn't\b": "should not",
    }
    for contraction, replacement in contractions.items():
        sentence = re.sub(contraction, replacement, sentence)
    return sentence

df['tweet_text'] = df['tweet_text'].apply(simplify_sentence)
df['tweet_text'] = df['tweet_text'].str.replace(r'[^a-zA-Z\s]', '', regex=True)

# Save the updated dataframe to a new CSV
df.to_csv("simplified_tableau_excel_csv.csv", index=False)
