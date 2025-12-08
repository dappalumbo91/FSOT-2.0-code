import os
import json
import glob
from PyPDF2 import PdfReader
from docx import Document
import fsot_2_0  # import the FSOT module

def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        return ""

def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
        return ""

def extract_text_from_md(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading MD {file_path}: {e}")
        return ""

def get_files_to_process(root_dir, exclude_dir):
    files = []
    for ext in ['*.md', '*.pdf', '*.docx']:
        pattern = os.path.join(root_dir, '**', ext)
        files.extend(glob.glob(pattern, recursive=True))
    # Exclude files in exclude_dir
    files = [f for f in files if not f.startswith(exclude_dir)]
    return files

def detect_domain(text):
    text_lower = text.lower()
    if 'cosmological' in text_lower or 'universe' in text_lower or 'big bang' in text_lower:
        return 'cosmological'
    elif 'quantum' in text_lower or 'particle' in text_lower:
        return 'quantum'
    elif 'biological' in text_lower or 'life' in text_lower or 'dna' in text_lower:
        return 'biological'
    elif 'astronomical' in text_lower or 'star' in text_lower or 'galaxy' in text_lower:
        return 'astronomical'
    else:
        return 'cosmological'  # default

def process_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        text = extract_text_from_pdf(file_path)
    elif ext == '.docx':
        text = extract_text_from_docx(file_path)
    elif ext == '.md':
        text = extract_text_from_md(file_path)
    else:
        text = ""
    return text

def main():
    root_dir = r'b:\fsuft aasb'
    exclude_dir = os.path.join(root_dir, 'fsot-2.0')
    files = get_files_to_process(root_dir, exclude_dir)
    
    data = {}
    for file_path in files:
        print(f"Processing {file_path}")
        text = process_file(file_path)
        domain = detect_domain(text)
        try:
            scalar = fsot_2_0.compute_for_domain(domain)
            fsot_scalar = str(scalar)  # convert to string for JSON
        except Exception as e:
            print(f"Error computing FSOT for {file_path}: {e}")
            fsot_scalar = None
        data[file_path] = {
            'text': text,
            'domain': domain,
            'fsot_scalar': fsot_scalar
        }
    
    # Save to JSON
    with open(os.path.join(exclude_dir, 'processed_data.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print("Processing complete. Data saved to processed_data.json")

if __name__ == "__main__":
    main()