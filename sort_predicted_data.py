# import pandas as pd

# # Example DataFrame with 'text', 'true_label', and 'predicted_label' columns

# path_pred_Vader = "/home/arnaud/M2_IA/LMM_seminary/vader_test_results.csv"

# df = pd.read_csv(path_pred_Vader, sep = ",")

# # Misclassifications
# incorrect_negatives = df[(df['true_label'] == 'negative') & (df['predicted_label'] != 'negative')]
# incorrect_positives = df[(df['true_label'] == 'positive') & (df['predicted_label'] != 'positive')]
# incorrect_neutrals = df[(df['true_label'] == 'neutral') & (df['predicted_label'] != 'neutral')]

# # Combine all incorrect predictions
# incorrect_predictions = pd.concat([incorrect_negatives, incorrect_positives, incorrect_neutrals])

# # Optionally, you can sort by the 'predicted_label' column or another criterion
# incorrect_predictions_sorted = incorrect_predictions.sort_values(by='predicted_label', ascending=False)

# # Display the sorted DataFrame
# df.to_csv('vader_test_results_sorted.csv', index=False)

import pandas as pd

# Path to the CSV file
path_pred_Vader = "/home/arnaud/M2_IA/LMM_seminary/vader_test_results.csv"

# Load the CSV file into a DataFrame
df = pd.read_csv(path_pred_Vader, sep=",")

# Filter for rows where the true label is different from the predicted label
incorrect_predictions = df[df['true_label'] != df['predicted_label']]

# Optionally, sort by the 'predicted_label' or another criterion
incorrect_predictions_sorted = incorrect_predictions.sort_values(by='predicted_label', ascending=False)

# Save the sorted DataFrame to a new CSV file
incorrect_predictions_sorted.to_csv('vader_test_results_sorted.csv', index=False)
