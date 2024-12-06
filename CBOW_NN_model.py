import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Flatten
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

# Load the datasets
path = "path/to/simplified_data.csv"
df = pd.read_csv(path, sep = ",") 

path_original = "path/to/original_data.csv"
df_original = pd.read_csv(path_original, sep = ";")

def replace_sentiment_labels(df):
    sentiment_map = {'positive': 2, 'neutral': 1, 'negative': 0}
    df['airline_sentiment'] = df['airline_sentiment'].map(sentiment_map)
    return df

df_original = replace_sentiment_labels(df_original)

# Clean text function to remove emojis and unwanted characters (optional)
def clean_text(text):
    import re
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    text = re.sub(r'[^a-zA-Z0-9\s.,!?;]', '', text)  # Keep only letters and punctuation
    text = text.lower()  # Convert text to lowercase
    return text

df['tweet_text'] = df['tweet_text'].apply(clean_text)

# Encoding labels (sentiment classes)
encoder = LabelEncoder()
df['airline_sentiment'] = encoder.fit_transform(df['airline_sentiment'])
y = df['airline_sentiment']

y_original = df_original['airline_sentiment']

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(df['tweet_text'], y, test_size=0.2, random_state=42)
X_test_original, _ , y_test_original, _ = train_test_split(df_original['tweet_text'], y_original, test_size=0.2, random_state=42)

# Tokenize the text
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_test_original_seq = tokenizer.texts_to_sequences(X_test_original)

# Pad sequences to ensure consistent input length
max_len = 100
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)
X_test_original_pad = pad_sequences(X_test_original_seq, maxlen=max_len)

# One-hot encode labels for the classification task (if using more than 2 classes)
y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)
y_test_original_cat = to_categorical(y_test_original, num_classes=3)

### 1. CBOW Model ###

# CBOW model using an embedding layer
cbow_model = Sequential()
cbow_model.add(Embedding(input_dim=5000, output_dim=128, input_length=max_len))
cbow_model.add(Flatten())
cbow_model.add(Dense(128, activation='relu'))
cbow_model.add(Dropout(0.5))
cbow_model.add(Dense(3, activation='softmax'))

# Compile the CBOW model
cbow_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the CBOW model
cbow_model.fit(X_train_pad, y_train_cat, epochs=10, batch_size=64, validation_data=(X_test_pad, y_test_cat))

# Evaluate the CBOW model on test data
cbow_loss, cbow_accuracy = cbow_model.evaluate(X_test_pad, y_test_cat)
print(f"CBOW Model Accuracy simple data: {cbow_accuracy * 100:.2f}%")

# Evaluate the CBOW model on original test data
cbow_loss_original, cbow_accuracy_original = cbow_model.evaluate(X_test_original_pad, y_test_original_cat)
print(f"CBOW Model Accuracy original data: {cbow_accuracy_original * 100:.2f}%")

# Print and save CBOW Model Accuracy
output_file = "cbow_analysis_results_simplified_dataset.txt"
with open(output_file, 'w') as file:
    cbow_accuracy_text = f"CBOW Model Accuracy: {cbow_accuracy * 100:.2f}%\n"
    print(cbow_accuracy_text.strip())
    file.write(cbow_accuracy_text)

    # Predict with CBOW model
    y_pred_cbow = np.argmax(cbow_model.predict(X_test_pad), axis=1)

    # Function to evaluate class-wise accuracy
    def evaluate_class_accuracy(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        cm_text = "Confusion Matrix:\n" + str(cm) + "\n"
        print(cm_text.strip())
        file.write(cm_text)

        class_accuracy_text = "\nClass-wise Accuracy:\n"
        file.write(class_accuracy_text)
        print(class_accuracy_text.strip())

        for i in range(len(encoder.classes_)):  # Assuming classes are encoded
            accuracy = (cm[i, i] / cm[i, :].sum()) * 100
            class_accuracy = f"Accuracy for class {encoder.classes_[i]}: {accuracy:.2f}%\n"
            print(class_accuracy.strip())
            file.write(class_accuracy)

    # Evaluate class-wise accuracy
    evaluate_class_accuracy(y_test, y_pred_cbow)

    # Find and save all the wrong predictions
    wrong_predictions = []
    for i in range(len(y_test)):
        if y_pred_cbow[i] != np.argmax(y_test_cat[i]):  # Compare predicted vs actual class label
            wrong_predictions.append({
                'text': X_test.iloc[i],  # Accessing the tweet text correctly from X_test (Series)
                'true_label': encoder.inverse_transform([np.argmax(y_test_cat[i])])[0],
                'predicted_label': encoder.inverse_transform([y_pred_cbow[i]])[0]
            })

    # Convert the wrong predictions to a DataFrame and save to CSV
    wrong_predictions_df = pd.DataFrame(wrong_predictions)
    wrong_predictions_df.to_csv('wrong_predictions_CBOW_simplified_dataset.csv', index=False)

    wrong_predictions_text = "Wrong predictions saved to 'wrong_predictions_CBOW_simplified_dataset.csv'\n"
    print(wrong_predictions_text.strip())
    file.write(wrong_predictions_text)

# Print and save CBOW Model Accuracy
output_file_original = "cbow_analysis_results_original_dataset.txt"
with open(output_file_original, 'w') as file:
    cbow_accuracy_original_text = f"CBOW Model Accuracy: {cbow_accuracy_original * 100:.2f}%\n"
    print(cbow_accuracy_original_text.strip())
    file.write(cbow_accuracy_original_text)

    # Predict with CBOW model
    y_pred_cbow_original = np.argmax(cbow_model.predict(X_test_original_pad), axis=1)

    # Function to evaluate class-wise accuracy
    def evaluate_class_accuracy(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        cm_text = "Confusion Matrix:\n" + str(cm) + "\n"
        print(cm_text.strip())
        file.write(cm_text)

        class_accuracy_text = "\nClass-wise Accuracy:\n"
        file.write(class_accuracy_text)
        print(class_accuracy_text.strip())

        for i in range(len(encoder.classes_)):  # Assuming classes are encoded
            accuracy = (cm[i, i] / cm[i, :].sum()) * 100
            class_accuracy_original = f"Accuracy for class {encoder.classes_[i]}: {accuracy:.2f}%\n"
            print(class_accuracy_original.strip())
            file.write(class_accuracy_original)

    # Evaluate class-wise accuracy
    evaluate_class_accuracy(y_test_original, y_pred_cbow_original)

    # Find and save all the wrong predictions
    wrong_predictions = []
    for i in range(len(y_test)):
        if y_pred_cbow_original[i] != np.argmax(y_test_cat[i]):  # Compare predicted vs actual class label
            wrong_predictions.append({
                'text': X_test.iloc[i],  # Accessing the tweet text correctly from X_test (Series)
                'true_label': encoder.inverse_transform([np.argmax(y_test_cat[i])])[0],
                'predicted_label': encoder.inverse_transform([y_pred_cbow_original[i]])[0]
            })

    # Convert the wrong predictions to a DataFrame and save to CSV
    wrong_predictions_df = pd.DataFrame(wrong_predictions)
    wrong_predictions_df.to_csv('wrong_predictions_CBOW_original_dataset.csv', index=False)

    wrong_predictions_text = "Wrong predictions saved to 'wrong_predictions_CBOW_original_dataset.csv'\n"
    print(wrong_predictions_text.strip())
    file.write(wrong_predictions_text)
