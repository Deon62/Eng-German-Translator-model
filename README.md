# Eng-German-Translator-model

# 🌍 T5 English ↔ German Translator

This repository hosts a fine-tuned **T5 model** for **English ↔ German translation**.  
The model is uploaded and maintained by [@chinesemusk](https://huggingface.co/chinesemusk).  

---

## 📌 Model Details
- **Architecture**: T5 (Text-to-Text Transfer Transformer)
- **Task**: Machine Translation (English ↔ German)
- **Format**: PyTorch + safetensors
- **Tokenizer**: SentencePiece (spiece.model)

---

## 🚀 Usage

You can load and run the model in just a few lines:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("chinesemusk/t5-en-de-translator")
model = AutoModelForSeq2SeqLM.from_pretrained("chinesemusk/t5-en-de-translator")

text = "Das ist ein Test."
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=40)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
