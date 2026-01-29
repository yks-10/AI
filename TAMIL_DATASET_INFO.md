# Tamil Dataset Information

## Overview

The dataset has been updated to use **Tamil text** instead of English. Tamil is a Dravidian language spoken primarily in Tamil Nadu, India, and Sri Lanka.

## What Changed

### 1. `data_set.txt`
- ✅ Updated with comprehensive Tamil vocabulary and sentences
- Contains:
  - Tamil alphabet (vowels and consonants)
  - Family relationships (அம்மா, அப்பா, etc.)
  - Common objects (வீடு, கார், etc.)
  - Colors (சிவப்பு, நீலம், etc.)
  - Numbers (ஒன்று, இரண்டு, etc.)
  - Greetings and common phrases
  - Educational content
  - And much more!

### 2. `prepare_data.py`
- ✅ Updated to handle Tamil Unicode characters properly
- **Key changes:**
  - Removed `.lower()` (Tamil doesn't have case distinctions)
  - Updated regex to preserve Tamil Unicode range (U+0B80 to U+0BFF)
  - Maintains proper Unicode encoding (UTF-8)

### 3. BPE Tokenizer
- ✅ Already supports Tamil (works with any Unicode characters)
- Python 3's Unicode support handles Tamil characters natively
- BPE will learn Tamil subword units automatically

## Tamil Unicode Information

- **Tamil Unicode Range**: U+0B80 to U+0BFF
- **Total Characters**: ~128 characters
- **Script Type**: Abugida (each character represents a consonant with inherent vowel)
- **No Case**: Tamil doesn't have uppercase/lowercase distinctions

## Sample Tamil Text in Dataset

```
அம்மா அப்பா தாத்தா பாட்டி
வீடு கார் பள்ளி புத்தகம்
சிவப்பு நீலம் பச்சை மஞ்சள்
வணக்கம் நன்றி மன்னிக்கவும்
```

## How to Use

### Step 1: Prepare Data
```bash
python prepare_data.py
```

This will:
- Read `data_set.txt` (Tamil text)
- Clean and normalize whitespace
- Preserve all Tamil characters
- Save to `cleaned_dataset.txt`

### Step 2: Train Model
```bash
python train.py
```

The BPE tokenizer will:
- Learn Tamil character patterns
- Create subword units for Tamil
- Build vocabulary optimized for Tamil text

### Step 3: Generate Text
```bash
python generate.py "வணக்கம்"
```

The model will generate Tamil text based on learned patterns.

## Tamil Text Processing

### Character Handling
- Tamil characters are properly preserved
- No case conversion needed (Tamil has no case)
- Unicode normalization is automatic in Python 3

### BPE Tokenization
- Starts with individual Tamil characters
- Learns common Tamil character pairs
- Creates meaningful Tamil subword units
- Example: "அம்மா" might become ["அ", "ம்மா"] or ["அம்மா"] depending on frequency

### Vocabulary
- Initial vocab: All unique Tamil characters (~128 base characters)
- After BPE: 1000 subword units (configurable)
- Includes common Tamil words, word parts, and character combinations

## Dataset Content Categories

1. **Alphabet**: அ, ஆ, இ, ஈ, etc.
2. **Family**: அம்மா, அப்பா, தாத்தா, etc.
3. **Objects**: வீடு, கார், பள்ளி, etc.
4. **Colors**: சிவப்பு, நீலம், பச்சை, etc.
5. **Nature**: சூரியன், சந்திரன், மழை, etc.
6. **Food**: ஆப்பிள், அரிசி, தோசை, etc.
7. **Numbers**: ஒன்று, இரண்டு, மூன்று, etc.
8. **Greetings**: வணக்கம், நன்றி, etc.
9. **Education**: பள்ளிக்கூடம், பாடம், etc.
10. **And many more categories!**

## Tips for Better Results

1. **More Data**: Add more Tamil text to `data_set.txt` for better learning
2. **Vocabulary Size**: Increase BPE vocab size if you have large dataset:
   ```python
   tokenizer = BPETokenizer(vocab_size=2000)  # In tokenizer.py
   ```
3. **Training**: Train for more epochs to learn Tamil patterns better
4. **Context**: Tamil has rich morphology - longer sequences help

## Example Tamil Words in Dataset

- **அம்மா** (amma) - Mother
- **அப்பா** (appa) - Father
- **வீடு** (veedu) - House
- **பள்ளி** (palli) - School
- **வணக்கம்** (vanakkam) - Hello/Greetings
- **நன்றி** (nandri) - Thank you
- **சிவப்பு** (sivappu) - Red
- **நீலம்** (neelam) - Blue

## Troubleshooting

### Issue: "Tamil characters not displaying correctly"
**Solution**: Ensure your terminal/editor supports UTF-8 encoding

### Issue: "Tokenizer not learning Tamil patterns"
**Solution**: 
- Increase vocabulary size
- Add more Tamil text to dataset
- Train for more epochs

### Issue: "Generated text has mixed languages"
**Solution**: Ensure `data_set.txt` contains only Tamil text

## Next Steps

1. **Add More Tamil Text**: Expand the dataset with more Tamil content
2. **Domain-Specific**: Add text from specific domains (news, literature, etc.)
3. **Fine-tune**: Adjust hyperparameters for Tamil language characteristics
4. **Evaluate**: Test generation quality with Tamil prompts

## Resources

- **Tamil Unicode Chart**: https://www.unicode.org/charts/PDF/U0B80.pdf
- **Tamil Language**: Dravidian language family
- **Script**: Tamil script (தமிழ் எழுத்து)

---

**The dataset is now ready for Tamil text generation!** 🎉
