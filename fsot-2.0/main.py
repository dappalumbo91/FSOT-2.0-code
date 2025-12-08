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

def extract_text_from_py(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading PY {file_path}: {e}")
        return ""

def get_files_to_process(root_dir, exclude_dir):
    files = []
    for ext in ['*.md', '*.pdf', '*.docx', '*.py']:
        pattern = os.path.join(root_dir, '**', ext)
        files.extend(glob.glob(pattern, recursive=True))
    # Exclude files in exclude_dir
    files = [f for f in files if not f.startswith(exclude_dir)]
    return files

def detect_domain(text):
    text_lower = text.lower()
    scores = {
        'cosmological': 0,
        'quantum': 0,
        'biological': 0,
        'astronomical': 0
    }
    
    # Cosmological keywords
    cosmo_keywords = ['cosmological', 'universe', 'big bang', 'dark energy', 'dark matter', 'inflation', 'cosmic microwave background', 'cmb', 'hubble', 'expansion', 'Λcdm', 'lambda cdm', 'redshift', 'baryon acoustic oscillation', 'bao']
    for keyword in cosmo_keywords:
        if keyword in text_lower:
            scores['cosmological'] += 1
    
    # Quantum keywords
    quantum_keywords = ['quantum', 'particle', 'wave function', 'entanglement', 'superposition', 'qubit', 'quantum field', 'string theory', 'loop quantum', 'quantum gravity', 'schrodinger', 'heisenberg', 'uncertainty principle']
    for keyword in quantum_keywords:
        if keyword in text_lower:
            scores['quantum'] += 1
    
    # Biological keywords
    bio_keywords = ['biological', 'life', 'dna', 'rna', 'protein', 'cell', 'evolution', 'genetic', 'organism', 'biosynthesis', 'metabolism', 'enzyme', 'biochemistry', 'neuroscience', 'ecosystem']
    for keyword in bio_keywords:
        if keyword in text_lower:
            scores['biological'] += 1
    
    # Astronomical keywords
    astro_keywords = ['astronomical', 'star', 'galaxy', 'planet', 'nebula', 'black hole', 'supernova', 'pulsar', 'quasar', 'exoplanet', 'solar system', 'milky way', 'constellation', 'telescope', 'observatory']
    for keyword in astro_keywords:
        if keyword in text_lower:
            scores['astronomical'] += 1
    
    # Return the domain with the highest score, default to cosmological
    max_score = max(scores.values())
    if max_score == 0:
        return 'cosmological'
    
    for domain, score in scores.items():
        if score == max_score:
            return domain
    
    return 'cosmological'

def process_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        text = extract_text_from_pdf(file_path)
    elif ext == '.docx':
        text = extract_text_from_docx(file_path)
    elif ext == '.md':
        text = extract_text_from_md(file_path)
    elif ext == '.py':
        text = extract_text_from_py(file_path)
    else:
        text = ""
    return text

def main():
    root_dir = r'b:\fsuft aasb'
    exclude_dir = os.path.join(root_dir, 'FSOT-2.0-code', 'fsot-2.0')
    files = get_files_to_process(root_dir, exclude_dir)
    
    # Load cross-reference data
    cross_ref_path = os.path.join(exclude_dir, 'cross_reference_data.json')
    cross_refs = {}
    if os.path.exists(cross_ref_path):
        with open(cross_ref_path, 'r', encoding='utf-8') as f:
            cross_refs = json.load(f)
    
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
        
        # Add cross-references for the domain
        references = []
        if domain in cross_refs:
            domain_refs = cross_refs[domain]
            # Get up to 3 arxiv papers
            if 'arxiv' in domain_refs and domain_refs['arxiv']:
                references.extend([{
                    'type': 'arxiv',
                    'title': paper.get('title', ''),
                    'url': paper.get('url', ''),
                    'summary': paper.get('summary', '')[:200] + '...' if paper.get('summary') else ''
                } for paper in domain_refs['arxiv'][:3]])
            # Get up to 2 pubmed articles
            if 'pubmed' in domain_refs and domain_refs['pubmed']:
                references.extend([{
                    'type': 'pubmed',
                    'title': paper.get('title', ''),
                    'pmid': paper.get('pmid', ''),
                    'abstract': paper.get('abstract', '')[:200] + '...' if paper.get('abstract') else ''
                } for paper in domain_refs['pubmed'][:2]])
        
        data[file_path] = {
            'text': text,
            'domain': domain,
            'fsot_scalar': fsot_scalar,
            'cross_references': references
        }
    
    # Save to JSON
    with open(os.path.join(exclude_dir, 'processed_data.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print("Processing complete. Data saved to processed_data.json")

if __name__ == "__main__":
    main()