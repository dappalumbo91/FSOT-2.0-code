import arxiv
import requests
from Bio import Entrez
import json

Entrez.email = "your.email@example.com"  # Replace with a real email

def fetch_arxiv_papers(query, max_results=10):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    papers = []
    for result in search.results():
        papers.append({
            'title': result.title,
            'authors': [author.name for author in result.authors],
            'summary': result.summary,
            'url': result.pdf_url,
            'published': str(result.published)
        })
    return papers

def fetch_pubmed_papers(query, max_results=10):
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()
        id_list = record["IdList"]
        
        if id_list:
            handle = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="text")
            records = handle.read()
            handle.close()
            # Parse records, but for simplicity, return IDs and titles
            papers = []
            for pmid in id_list:
                handle = Entrez.efetch(db="pubmed", id=pmid, rettype="xml", retmode="text")
                xml_data = handle.read().decode('utf-8')
                handle.close()
                # Simple parsing
                title = xml_data.split('<ArticleTitle>')[1].split('</ArticleTitle>')[0] if '<ArticleTitle>' in xml_data else 'N/A'
                abstract = xml_data.split('<AbstractText>')[1].split('</AbstractText>')[0] if '<AbstractText>' in xml_data else 'N/A'
                papers.append({
                    'pmid': pmid,
                    'title': title,
                    'abstract': abstract
                })
            return papers
        return []
    except Exception as e:
        print(f"Error fetching PubMed: {e}")
        return []

def main():
    # Example queries based on FSOT domains
    queries = {
        'cosmological': 'cosmology hubble tension',
        'quantum': 'quantum gravity',
        'biological': 'origin of life',
        'astronomical': 'exoplanets'
    }
    
    data = {}
    for domain, query in queries.items():
        print(f"Fetching data for {domain}: {query}")
        arxiv_papers = fetch_arxiv_papers(query)
        pubmed_papers = fetch_pubmed_papers(query)
        data[domain] = {
            'arxiv': arxiv_papers,
            'pubmed': pubmed_papers
        }
    
    with open('cross_reference_data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print("Cross-reference data saved to cross_reference_data.json")

if __name__ == "__main__":
    main()