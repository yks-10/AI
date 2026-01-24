import os
import re
from pathlib import Path

def load_and_clean_data(data_dir: str) -> str:
    """Load all .txt files from a directory and clean them."""
    data_dir = Path(data_dir)
    full_text = ""
    
    for file_path in data_dir.glob("*.txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            # Basic cleaning: lowercase, remove extra spaces/newlines, non-ASCII
            text = re.sub(r'\s+', ' ', text.lower())  # Normalize whitespace
            text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation (adjust as needed)
            full_text += text + " "  # Add space separator
    
    # Save cleaned data
    with open('cleaned_dataset.txt', 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    print(f"Loaded {len(full_text)} characters from {len(list(data_dir.glob('*.txt')))} files.")
    return full_text

# Usage - read the file directly since it's a single file, not a directory
with open('data_set.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    text = re.sub(r'\s+', ' ', text.lower())
    text = re.sub(r'[^\w\s]', '', text)

with open('cleaned_dataset.txt', 'w', encoding='utf-8') as f:
    f.write(text)

print(f"Loaded {len(text)} characters.")