import nltk
import os

# Create NLTK data directory
nltk_data_dir = os.path.expanduser('~/nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)

# Download required NLTK data
print("Downloading NLTK punkt tokenizer...")
nltk.download('punkt', download_dir=nltk_data_dir)
print("Download complete!")