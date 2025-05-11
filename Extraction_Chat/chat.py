import fitz  # PyMuPDF
import json
import re
import os
import google.generativeai as genai

# === CONFIG API ===
genai.configure(api_key="AIzaSyCELlvHniJaS-o4WfMieu_iAh7lq4DEiiY") 
model = genai.GenerativeModel("gemini-2.0-flash")
# === FICHIER SOURCE ===
pdf_path = "Extraction_Chat\\extraction_pdf_tass\\5. The-Impact-of-Climate-Change-on-the-Agricultural-Sector.pdf"
doc = fitz.open(pdf_path)
source_name = os.path.basename(pdf_path)

# === STRUCTURE DES CHAPITRES ===
# Note: Modifiez cette section si votre PDF a des chapitres spécifiques que vous souhaitez traiter.
# Remplacez "LAST_PAGE_NUMBER" par le numéro de la dernière page de votre PDF.
# Exemple pour plusieurs chapitres :
# chapters = [
#     {"title": "Chapitre 1 - Introduction", "start": 1, "end": 10},
#     {"title": "Chapitre 2 - Impacts Observés", "start": 11, "end": 25},
#     # ... autres chapitres
# ]
chapters = [
    {"title": "Chapitre 2 - Diagnosis and Forecast of Global Climate Change", "start": 1, "end":5 },
    {"title": "Chapitre 3 -Impacts of Climate Change on the Agricultural Sector", "start": 8, "end":21 }, 
    {"title": "Chapitre 4 -Mitigation and Adaptation Strategies for the Agriculture", "start": 21, "end":34 },
    {"title": "Chapitre 5 -Low carbon Green Growth Strategy/Roadmap for the Agricultural Sector", "start": 34, "end":44 },
]

# === EXTRACTION PAR CHAPITRE ===
def extract_text(start, end):
    text = ""
    for i in range(start - 1, end):
        if i < len(doc):
            page = doc[i]
            content = page.get_text().strip()
            text += f"[PAGE {i + 1}]\n{content}\n"
    return text.strip()

# === PROMPT STANDARD POUR CHAQUE CHAPITRE ===
def build_prompt(chap_title, text, source_name, start, end):
    return f"""
You are an expert in climate change impacts on agriculture and sustainable farming practices.

Analyze the content of this report/chapter: "{chap_title}" (pages {start} to {end}) from the report '{source_name}'.

Extract and structure all important insights as a JSON with this schema:

{{
  "chapter_title": "{chap_title}",
  "summary": "General summary of the chapter/document focusing on climate change impacts on agriculture.",
  "key_climate_change_impacts_on_agriculture": [
    "e.g., Reduced crop yields due to drought",
    "e.g., Increased pest and disease prevalence"
  ],
  "specific_impacts_on_attributes": [
    {{
      "attribute": "e.g., Average Surface Temperature (AvgSurfT)",
      "impact_description": "e.g., Higher average temperatures leading to heat stress in crops"
    }},
    {{
      "attribute": "e.g., Rainfall (Rainf)",
      "impact_description": "e.g., Altered rainfall patterns causing water scarcity or flooding"
    }},
    {{
      "attribute": "e.g., Potential Evapotranspiration (PotEvap)",
      "impact_description": "e.g., Increased potential evapotranspiration leading to higher irrigation needs"
    }}
  ],
  "vulnerable_crops_or_regions": [
    "e.g., Maize in semi-arid regions",
    "e.g., Coastal agricultural areas due to sea-level rise"
  ],
  "adaptation_recommendations": [
    "e.g., Introduction of drought-resistant crop varieties",
    "e.g., Improved water management techniques"
  ],
  "mitigation_recommendations_for_agriculture": [
    "e.g., Reduced use of nitrogen fertilizers",
    "e.g., Promotion of agroforestry"
  ],
  "relevant_statistics_or_data": [
    {{
      "indicator": "e.g., Percentage decrease in wheat yield",
      "value": "e.g., 10",
      "unit": "e.g., %",
      "context_or_region": "e.g., Due to a 2°C temperature rise in North Africa",
      "source_page": "e.g., 42"
    }}
  ],
  "source": "{source_name}",
  "pages": {list(range(start, end + 1)) if isinstance(end, int) else "UNKNOWN_PAGE_RANGE"}
}}

Now process this chapter content:
\"\"\"{text}\"\"\"

Return only valid JSON. No explanation or markdown.
"""

# === ANALYSE + SAUVEGARDE
output_folder = "extracted_data_climate_agriculture"
os.makedirs(output_folder, exist_ok=True)

for chap in chapters:
    print(f"🔎 Traitement {chap['title']} (pages {chap['start']}–{chap['end']})...")
    text = extract_text(chap['start'], chap['end'])
    prompt = build_prompt(chap["title"], text, source_name, chap["start"], chap["end"])

    try:
        response = model.generate_content(prompt)
        result = response.text.strip()
        json_text = re.search(r"\{[\s\S]*\}", result).group(0)
        data = json.loads(json_text)

        filename = chap["title"].split("-")[0].strip().lower().replace(" ", "_") + ".json"
        filepath = os.path.join(output_folder, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✅ Fichier créé : {filepath}")
    except Exception as e:
        print(f"❌ Erreur pour {chap['title']} : {e}")
