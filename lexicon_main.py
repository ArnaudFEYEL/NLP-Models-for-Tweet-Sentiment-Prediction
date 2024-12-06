import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
path = "path/to/simplified_data.csv"

df = pd.read_csv(path, sep = ",")

# Load the dataset
path_original = "path/to/original_data.csv"

pd_data = pd.read_csv(path_original, sep = ";", skipinitialspace=True)

df_original = pd_data.loc[:, ~pd_data.columns.str.contains('^Unnamed')]

df_original.columns = ['airline_sentiment', 'tweet_text']

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']

df['sentiment_score'] = df['tweet_text'].apply(get_sentiment)

def classify_sentiment(score):
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df['sentiment_label'] = df['sentiment_score'].apply(classify_sentiment)

X_train, X_test, y_train, y_test = train_test_split(df['tweet_text'], df['airline_sentiment'], test_size=0.2, random_state=42)
X_test_original, _ , y_test_original, _ = train_test_split(df_original['tweet_text'], df_original['airline_sentiment'], test_size=0.80, random_state=42)

test_sentiments = X_test.apply(get_sentiment).apply(classify_sentiment)
test_sentiments_original = X_test_original.apply(get_sentiment).apply(classify_sentiment)

vader_accuracy = accuracy_score(y_test, test_sentiments)
vader_accuracy_original = accuracy_score(y_test_original, test_sentiments_original)

# Function to calculate accuracy per class
def class_accuracy(class_label, y_true, y_pred):
    class_true = (pd.Series(y_true) == class_label)
    class_pred = (pd.Series(y_pred) == class_label)
    correct = (class_true & class_pred).sum()
    total = class_true.sum()
    if total > 0:
        return (correct / total) * 100
    else:
        return 0.0

# Output results to a file
output_file = "vader_analysis_results_simplified_dataset.txt"
with open(output_file, 'w') as file:
    # Overall accuracy
    overall_accuracy_text = f"Overall Accuracy of VADER lexicon-based sentiment analysis simplified data: {vader_accuracy:.4f}\n"
    print(overall_accuracy_text.strip())
    file.write(overall_accuracy_text)

    # Accuracy per class
    classes = ['positive', 'negative', 'neutral']
    for sentiment_class in classes:
        accuracy = class_accuracy(sentiment_class, y_test, test_sentiments)
        class_accuracy_text = f"Accuracy for {sentiment_class} class: {accuracy:.2f}%\n"
        print(class_accuracy_text.strip())
        file.write(class_accuracy_text)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, test_sentiments, labels=classes)
    conf_matrix_text = "\nConfusion Matrix:\n" + str(conf_matrix) + "\n"
    print(conf_matrix_text.strip())
    file.write(conf_matrix_text)

# Save detailed test results to CSV
df_test_results = pd.DataFrame({
    'text': X_test,
    'true_label': y_test,
    'predicted_label': test_sentiments
})
df_test_results.to_csv('vader_test_results_simplified_data.csv', index=False)

# Output results to a file
output_file = "vader_analysis_results_original_dataset.txt"
with open(output_file, 'w') as file:
    # Overall accuracy
    overall_accuracy_text = f"Overall Accuracy of VADER lexicon-based sentiment analysis for original data: {vader_accuracy_original:.4f}\n"
    print(overall_accuracy_text.strip())
    file.write(overall_accuracy_text)

    # Accuracy per class
    classes = ['positive', 'negative', 'neutral']
    for sentiment_class in classes:
        accuracy = class_accuracy(sentiment_class, y_test_original, test_sentiments_original)
        class_accuracy_text = f"Accuracy for {sentiment_class} class: {accuracy:.2f}%\n"
        print(class_accuracy_text.strip())
        file.write(class_accuracy_text)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test_original, test_sentiments_original, labels=classes)
    conf_matrix_text = "\nConfusion Matrix:\n" + str(conf_matrix) + "\n"
    print(conf_matrix_text.strip())
    file.write(conf_matrix_text)

# Save detailed test results to CSV
df_test_results = pd.DataFrame({
    'text': X_test_original,
    'true_label': y_test_original,
    'predicted_label': test_sentiments_original
})
df_test_results.to_csv('vader_test_results_original_data.csv', index=False)
