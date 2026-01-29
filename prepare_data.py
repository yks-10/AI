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
            # Normalize whitespace (multiple spaces/newlines to single space)
            text = re.sub(r'\s+', ' ', text)
            # For Tamil and Unicode: preserve word characters and Tamil Unicode range
            # Remove only punctuation, but keep Tamil characters (U+0B80-U+0BFF)
            text = re.sub(r'[^\w\s\u0B80-\u0BFF]', '', text)
            # Note: No .lower() for Tamil as it doesn't have case distinctions
            full_text += text + " "  # Add space separator
    
    # Save cleaned data
    with open('cleaned_dataset.txt', 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    print(f"Loaded {len(full_text)} characters from {len(list(data_dir.glob('*.txt')))} files.")
    return full_text

# Usage - read the file directly since it's a single file, not a directory
with open('data_set.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Normalize whitespace (multiple spaces/newlines to single space)
text = re.sub(r'\s+', ' ', text)

# For Tamil and other Unicode languages, we preserve all Unicode word characters
# Remove only punctuation marks, but keep Tamil characters and spaces
# This regex keeps: Tamil letters, numbers, and spaces
# Tamil Unicode range: \u0B80-\u0BFF
text = re.sub(r'[^\w\s\u0B80-\u0BFF]', '', text)

# Note: We don't use .lower() for Tamil as it doesn't have case distinctions
# Tamil text is already in its standard form

with open('cleaned_dataset.txt', 'w', encoding='utf-8') as f:
    f.write(text)

print(f"Loaded {len(text)} characters.")
print(f"Sample text: {text[:100]}...")