import csv
from sklearn.feature_extraction.text import TfidfVectorizer

# Read the CSV file
csv_filename = "./docs/for_sklearn.csv"

# Store the documents from the CSV file
documents = []

with open(csv_filename, "r") as csv_file:
    reader = csv.reader(csv_file)
    next(reader)  # Skip the header row

    # Iterate over each row in the CSV file
    for row in reader:
        document = row[0]  # Assuming the document is in the first column
        documents.append(document)

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Vectorize the documents
vectorized_documents = vectorizer.fit_transform(documents)

# Get the feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Prepare the CSV file for vectorized documents
vectorized_csv_filename = "./docs/vectorized_documents.csv"

# Write the vectorized documents to the CSV file
with open(vectorized_csv_filename, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)

    # Write the header row
    header_row = feature_names
    writer.writerow(header_row)

    # Write each document's vector to the CSV file
    for i, document in enumerate(documents):
        # Get the vectorized document as a list
        vectorized_document = vectorized_documents[i].toarray().flatten().tolist()

        # Write the document and its vector to the CSV file
        writer.writerow([document] + vectorized_document)

print(f"Vectorized CSV file '{vectorized_csv_filename}' created successfully.")
