import json
import os
from fsot_2_0 import compute_for_domain  # Import your FSOT function

def load_data():
    """Load processed data and cross-references."""
    with open('processed_data.json', 'r', encoding='utf-8') as f:
        processed = json.load(f)
    with open('cross_reference_data.json', 'r', encoding='utf-8') as f:
        cross_refs = json.load(f)
    return processed, cross_refs

def validate_scalar(domain, computed_scalar, benchmarks):
    """Compare computed scalar to known benchmarks for the domain."""
    if domain not in benchmarks or benchmarks[domain] == 0:
        return "No valid benchmark available", None
    
    benchmark = benchmarks[domain]
    diff = abs(computed_scalar - benchmark) / abs(benchmark) * 100  # Percentage difference
    accuracy = 100 - diff
    status = "Within 95-100% threshold" if 95 <= accuracy <= 100 else f"Outside threshold ({accuracy:.2f}%)"
    return status, accuracy

def analyze_contrast(file_data, cross_refs):
    """Analyze how well the file's content contrasts with cross-reference data."""
    domain = file_data['domain']
    text = file_data['text'].lower()
    
    if domain not in cross_refs:
        return "No cross-reference data for domain"
    
    refs = cross_refs[domain]
    matches = 0
    total_refs = len(refs.get('arxiv', [])) + len(refs.get('pubmed', []))
    
    for source in ['arxiv', 'pubmed']:
        if source in refs:
            for ref in refs[source]:
                title = ref.get('title', '').lower()
                content = ref.get('summary', '') if source == 'arxiv' else ref.get('abstract', '')
                content = content.lower() if content else ""
                if any(keyword in text for keyword in title.split()) or any(keyword in content for keyword in text.split()):
                    matches += 1
    
    contrast_score = (matches / total_refs * 100) if total_refs > 0 else 0
    return f"Contrast score: {contrast_score:.2f}% alignment with {matches}/{total_refs} references"

def main():
    # Define benchmarks based on your validated data (update with real values from 472 domains)
    benchmarks = {
        'cosmological': -0.50245594621004331636932221476456297355753923053727,  # Validated example
        'quantum': 0.9555063001027196,  # From planetary_gear_board_blueprint.md (example)
        'biological': 0.0,  # Placeholder; update with real validated value
        'astronomical': 0.8984595681660628  # From Palumbo_Perpetual_Flux_Generator_Blueprint.md (example)
    }
    
    processed, cross_refs = load_data()
    results = {}
    
    for file_path, data in processed.items():
        domain = data['domain']
        computed_scalar = float(data['fsot_scalar']) if data['fsot_scalar'] else None
        
        if computed_scalar is not None:
            validation_status, accuracy = validate_scalar(domain, computed_scalar, benchmarks)
            contrast_analysis = analyze_contrast(data, cross_refs)
            results[file_path] = {
                'domain': domain,
                'computed_scalar': computed_scalar,
                'validation': validation_status,
                'accuracy': accuracy,
                'contrast': contrast_analysis
            }
        else:
            results[file_path] = {'error': 'No scalar computed'}
    
    # Save results
    with open('validation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    print("Validation complete. Results saved to validation_results.json")

if __name__ == "__main__":
    main()