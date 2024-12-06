import pandas as pd 

# # Path to your input CSV file and output CSV file
# input_path = "/home/arnaud/M2_IA/LMM_seminary/Tweets.csv"
# output_path = "/home/arnaud/M2_IA/LMM_seminary/Tweets_with_semicolon.csv"

# # Open the input CSV file in read mode and the output CSV file in write mode
# with open(input_path, 'r', newline='', encoding='utf-8') as infile, \
#      open(output_path, 'w', newline='', encoding='utf-8') as outfile:
    
#     reader = csv.reader(infile, delimiter=';')
#     writer = csv.writer(outfile, delimiter=';')
    
#     for row in reader:
#         row.append('')  # Add an empty string at the end of each row
#         writer.writerow(row)

# print(f"New file with semicolons at the end has been saved to: {output_path}")



path = "/home/arnaud/M2_IA/LMM_seminary/tableau_excel_csv.csv"


pd_data = pd.read_csv(path, sep = ";", skipinitialspace=True)

#pd_data = pd_data["Tweets", "airline_sentiment"]
pd_data = pd_data.loc[:, ~pd_data.columns.str.contains('^Unnamed')]

print(pd_data.head(3))

pd_data.to_csv("tweet_data_clean.csv")  

