import pandas as pd

csv_path = r"data/raw/OpenDataset/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"

df = pd.read_csv(csv_path)

print("Total cases:", len(df))

print("\nColumns:")
print(df.columns.tolist())

print("\nVendor counts:")
print(df["Vendor"].value_counts())

print("\nVendorName counts:")
print(df["VendorName"].value_counts())

print("\nCentre counts:")
print(df["Centre"].value_counts())

print("\nPathology counts:")
print(df["Pathology"].value_counts())

print("\nFirst 5 rows:")
print(df.head())

print(df["Vendor"].value_counts())
print(df["VendorName"].value_counts())