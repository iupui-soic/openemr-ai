import os
import re
import json
import random
from pathlib import Path
import openai
from dotenv import load_dotenv, find_dotenv

# ============================================================
# CONFIG
# ============================================================
# Explicitly find and load .env
env_path = find_dotenv()
print(f"Loading .env from: {env_path}")
load_dotenv(env_path, override=True)

INPUT_FILE = "data/Top_100.txt"
OUTPUT_DIR = "output_notes"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("WARNING: OPENAI_API_KEY not found in environment variables.")
else:
    print(f"API Key found: {api_key[:5]}...{api_key[-5:]}")

try:
    client = openai.OpenAI(api_key=api_key)
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    client = None

# Global token usage tracker
TOKEN_USAGE = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
}

# ============================================================
# PROMPT DEFINITION
# ============================================================
SYSTEM_PROMPT = """You are an expert clinical information extraction model. 
Your goal is to convert the NOTE TEXT into a STRICT, NON-HALLUCINATED, 
HIGH-PRECISION JSON structure following a rigid SOAP format.

CRITICAL RULES (FOLLOW EXACTLY):

1. DO NOT HALLUCINATE ANY FACT.
   Only extract information that explicitly appears in the note text.
   If something is not present, return an empty list or empty object.

2. DO NOT EXTRACT ANY NUMBERS.
   Remove ALL numeric values, dates, vital values, lab results, medication doses, quantities, times, or measurements.
   Extract ONLY concepts, not values.

3. DISEASE EXTRACTION MUST BE PRECISE:
   • Extract ONLY clinically valid diseases or diagnoses explicitly mentioned.
   • Remove labels like “Primary diagnosis”, “Secondary diagnosis”, “Other”.
   • Remove grammatical junk (“and”, “for”, “given underlying”, “due to”).
   • DO NOT include symptoms, procedures, or findings.
   • DO NOT include items with negations (“no”, “denies”, “negative for”).

4. NO PARAPHRASE OR INFERENCE.
   Use the clinician’s original meaning, not interpretations.

5. LAB PANELS (EXTRACT THESE):
   • Infer standard panels: CBC, CMP/BMP, LFT, Coagulation, Lipase, Urinalysis.
   • Put each analyte under the correct panel.
   • NO VALUES. Only analyte names (e.g., "WBC", "Glucose").

6. IMAGING & MICROBIOLOGY (EXTRACT THESE):
   • Extract only study type (e.g., "CT Chest", "Urine Culture").
   • Do not add interpretations or diagnoses.

7. EXCLUDED FIELDS (MUST BE EMPTY):
   You MUST return EMPTY LISTS/OBJECTS for the following fields (do not extract data for them, they are placeholders):
   - subjective (chief_complaint, hpi, review_of_systems, past_medical_history, surgical_history, social_history, family_history, allergies, current_medications)
   - objective (vital_signs, physical_exam)
   - assessment (primary_diagnosis, secondary_diagnoses, hospital_course)
   - plan (medications_prescribed, discharge_instructions, followup_instructions)

   (Note: 'labs' and 'imaging' are NOT excluded and should be populated).

8. RETURN STRICT JSON ONLY."""

USER_PROMPT_TEMPLATE = """
RETURN IN THIS EXACT SOAP FORMAT (Ensure 'Excluded Fields' are empty, but populate Labs/Imaging):
=====================

{{
  "metadata": {{
    "note_id": "",
    "diseases": []
  }},
  "subjective": {{
    "chief_complaint": [],
    "hpi": {{
       "symptom_clusters": [],
       "timeline_events": [],
       "associated_symptoms_positive": [],
       "associated_symptoms_negative": []
    }},
    "review_of_systems": {{
       "constitutional": [],
       "respiratory": [],
       "cardiovascular": [],
       "gastrointestinal": [],
       "other": []
    }},
    "allergies": [],
    "past_medical_history": {{ "problems": [] }},
    "surgical_history": [],
    "social_history": [],
    "family_history": [],
    "current_medications": {{
       "home_medications": []
    }}
  }},
  "objective": {{
    "vital_signs": {{
       "bp": [],
       "hr": [],
       "temp": [],
       "rr": [],
       "spo2": [],
       "weight": []
    }},
    "physical_exam": {{
       "admission_exam": {{ "vitals": [], "systems": [] }},
       "discharge_exam": {{ "vitals": [], "systems": [] }}
    }},
    "labs": {{
       "panels": [
          {{ "panel_name": "", "analytes": [] }}
       ],
       "microbiology": []
    }},
    "imaging": [
       {{ "study_type": "", "findings": [] }}
    ]
  }},
  "assessment": {{
    "primary_diagnosis": [],
    "secondary_diagnoses": [],
    "hospital_course": []
  }},
  "plan": {{
    "medications_prescribed": {{
       "discharge_medications": []
    }},
    "discharge_instructions": [],
    "followup_instructions": []
  }}
}}

NOTE TEXT:
{note_text}
"""

# ============================================================
# 0. GPT-4o CALLER
# ============================================================
def call_gpt(system_prompt: str, user_prompt: str, max_retries: int = 3):
    """
    Robust GPT-4o caller that enforces JSON output.
    Uses OpenAI v1.0+ client.
    """
    if not client:
        raise ValueError("OpenAI client not initialized (missing API key?).")

    print(f"  Calling GPT-4o... (Prompt length: {len(user_prompt)})")
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0
            )
            
            # Track token usage
            if response.usage:
                TOKEN_USAGE["prompt_tokens"] += response.usage.prompt_tokens
                TOKEN_USAGE["completion_tokens"] += response.usage.completion_tokens
                TOKEN_USAGE["total_tokens"] += response.usage.total_tokens

            # Access content via object attributes in v1
            content = response.choices[0].message.content
            
            if not content:
                continue

            # Strip markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            return json.loads(content)
        except Exception as e:
            print(f"  Error in call_gpt (attempt {attempt+1}/{max_retries}): {e}")
            continue
    raise ValueError("GPT-4o failed to produce valid JSON after multiple retries.")


# ============================================================
# 1. NOTE SPLITTING
# ============================================================
def split_notes(raw_text: str):
    """
    Split file into notes based on lines like:
    --- Note 1 ---
    """
    pattern = r"---\s*Note\s*(\d+)\s*---"
    parts = re.split(pattern, raw_text)
    notes = {}
    for i in range(1, len(parts), 2):
        num = int(parts[i].strip())
        notes[num] = parts[i + 1].strip()
    return notes


# ============================================================
# 8. FULL PIPELINE
# ============================================================
def process_all_notes():
    print(f"Reading {INPUT_FILE}...")
    try:
        raw = Path(INPUT_FILE).read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at {INPUT_FILE}")
        return

    notes = split_notes(raw)
    print(f"Found {len(notes)} notes.")

    all_extracted_data = []
    
    # Loop through ALL notes
    for note_id, note_text in notes.items():
        print(f"\nProcessing Note {note_id} ({len(all_extracted_data) + 1}/{len(notes)})...")
        
        try:
            # Construct user prompt
            user_prompt = USER_PROMPT_TEMPLATE.format(note_text=note_text)

            # Call GPT
            schema = call_gpt(SYSTEM_PROMPT, user_prompt)
            
            # -------------------------------------------------------
            # POST-PROCESSING: Inject Metadata
            # -------------------------------------------------------
            if "metadata" not in schema:
                schema["metadata"] = {}
            
            schema["metadata"]["note_id"] = str(note_id)
            
            # Ensure diseases are captured if present in assessment
            # (The prompt asks to put them in metadata.diseases, but let's double check)
            # If the model put them in assessment.primary_diagnosis, we might want to copy them?
            # For now, we trust the model followed the "metadata.diseases" instruction.
            
            all_extracted_data.append(schema)
            print(f"✅ Note {note_id} processed successfully.")

        except Exception as e:
            print(f"❌ Failed to process Note {note_id}: {e}")
            continue

    # Save all to a single JSON file
    final_output_file = Path("data/all_notes_structure.json")
    final_output_file.write_text(json.dumps(all_extracted_data, indent=2), encoding="utf-8")
    
    print("=" * 30)
    print(f"🎉 COMPLETED! Processed {len(all_extracted_data)} notes.")
    print(f"📁 Saved to: {final_output_file}")
    print(f"TOTAL TOKENS USED: {TOKEN_USAGE['total_tokens']}")
    print("=" * 30)


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    process_all_notes()
