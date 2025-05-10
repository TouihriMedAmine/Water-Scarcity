import os
import json
from transformers import MarianMTModel, MarianTokenizer
from pathlib import Path

# ✅ Load Hugging Face model and tokenizer
model_name = 'Helsinki-NLP/opus-mt-fr-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# 📁 Folder paths (use your own)
input_folder = r"C:\Users\G1524\Desktop\extraction_pdf_tass\extraction_pdf_tass\Chapters_used"
output_folder = r"C:\Users\G1524\Desktop\extraction_pdf_tass\extraction_pdf_tass\Translation_Folder"
os.makedirs(output_folder, exist_ok=True)

# ✅ Translation function
def translate_text(text):
    if not text.strip():
        return ""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated = model.generate(**inputs, max_length=512)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# 🌀 Process and translate JSON files
for filename in os.listdir(input_folder):
    if filename.endswith(".json"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 🔁 Translate specific fields
        data["chapter_title"] = translate_text(data.get("chapter_title", ""))
        data["summary"] = translate_text(data.get("summary", ""))
        data["key_points"] = [translate_text(p) for p in data.get("key_points", [])]
        data["recommendations"] = [translate_text(r) for r in data.get("recommendations", [])]

        # Optional: Translate statistics indicators
        for stat in data.get("statistics", []):
            stat["indicator"] = translate_text(stat.get("indicator", ""))

        # 💾 Save translated JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✅ Translated and saved: {filename}")
