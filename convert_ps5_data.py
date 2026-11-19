import pyreadr

# Read the RData file
result = pyreadr.read_r("PS5v3.R")

# Show objects stored inside the RData file
print(result.keys())

# Extract GMMdata
df = result["GMMdata"]

# Inspect the dataset
print(df.head())
print(df.shape)
print(df.columns)

# Save as CSV
df.to_csv("PS5v3.csv", index=False)

print("PS5v3.csv was created successfully.")