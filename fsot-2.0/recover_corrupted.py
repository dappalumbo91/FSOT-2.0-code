import os
import json
import docx2txt
import pdfplumber
from pathlib import Path

def extract_text_fallback(file_path):
    """Extract text using fallback libraries for corrupted files."""
    try:
        if file_path.endswith('.docx'):
            return docx2txt.process(file_path)
        elif file_path.endswith('.pdf'):
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
                return text
        else:
            return None
    except Exception as e:
        print(f"Error extracting {file_path}: {e}")
        return None

def recover_corrupted_files():
    """Recover text from corrupted .fsot_updated files and update processed_data.json."""
    corrupted_dir = Path(r'b:\fsuft aasb\fsot tech\fsot_updated_blueprints')
    processed_file = Path(r'b:\fsuft aasb\FSOT-2.0-code\fsot-2.0\processed_data.json')

    if not processed_file.exists():
        print("processed_data.json not found.")
        return

    with open(processed_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    recovered_count = 0
    for file_path in corrupted_dir.glob('*.fsot_updated.*'):
        if str(file_path) in data:
            continue  # Already processed
        text = extract_text_fallback(str(file_path))
        if text:
            # Detect domain (simple keyword check, can expand)
            domain = "engineering"  # Default, or use keyword detection
            if "quantum" in text.lower():
                domain = "quantum"
            elif "biological" in text.lower():
                domain = "biological"
            # Compute scalar (placeholder, use fsot_2_0.compute_for_domain)
            from fsot_2_0 import compute_for_domain
            try:
                scalar = str(compute_for_domain(domain))
            except:
                scalar = "0.0"
            data[str(file_path)] = {
                "text": text,
                "domain": domain,
                "fsot_scalar": scalar,
                "cross_references": []  # Can add later
            }
            recovered_count += 1
            print(f"Recovered: {file_path}")

    with open(processed_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Recovered {recovered_count} files.")

if __name__ == "__main__":
    recover_corrupted_files()