"""
Neuroscience Weekly Newsletter Agent
Scrapes PubMed, arXiv, and bioRxiv for recent papers and sends email digest.
"""

import requests
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import os
import sys
from xml.etree import ElementTree as ET

# ============== CONFIGURATION ==============
# Uses environment variables (for GitHub Actions) or fallback values
EMAIL_CONFIG = {
    "sender": os.getenv("EMAIL_SENDER", "your_email@libero.it"),
    "password": os.getenv("EMAIL_PASSWORD", "your_password"),
    "recipient": os.getenv("EMAIL_RECIPIENT", "your_email@libero.it"),
    "smtp_server": "smtp.libero.it",
    "smtp_port": 465,
    "use_ssl": True  # Libero uses SSL on port 465
}

# Search queries for each category
CATEGORIES = {
    "Cognitive Neuroscience": [
        "cognitive neuroscience", "neural correlates cognition", 
        "brain cognition", "cognitive neuroimaging"
    ],
    "Clinical Neuroscience": [
        "neurological disease management", "neurology guidelines",
        "clinical neurology", "neurological disorders treatment"
    ],
    "Neuroimaging Analysis": [
        "fMRI analysis", "neuroimaging methods", "brain imaging",
        "MRI preprocessing", "EEG analysis methods"
    ],
    "AI in Neuroscience": [
        "deep learning neuroscience", "machine learning brain",
        "neural networks neuroimaging", "AI brain analysis"
    ]
}

MAX_PAPERS_PER_CATEGORY = 5


# ============== DATA SOURCES ==============

def search_pubmed(query, max_results=5):
    """Search PubMed for recent papers."""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    # Get last 7 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    date_range = f"{start_date:%Y/%m/%d}:{end_date:%Y/%m/%d}[pdat]"
    
    # Search
    search_url = f"{base_url}/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": f"{query} AND {date_range}",
        "retmax": max_results,
        "sort": "relevance",
        "retmode": "json"
    }
    
    try:
        resp = requests.get(search_url, params=params, timeout=10)
        ids = resp.json().get("esearchresult", {}).get("idlist", [])
        
        if not ids:
            return []
        
        # Fetch details
        fetch_url = f"{base_url}/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "xml"
        }
        resp = requests.get(fetch_url, params=fetch_params, timeout=10)
        
        papers = []
        root = ET.fromstring(resp.content)
        for article in root.findall(".//PubmedArticle"):
            title_el = article.find(".//ArticleTitle")
            pmid_el = article.find(".//PMID")
            
            if title_el is not None and pmid_el is not None:
                papers.append({
                    "title": title_el.text or "No title",
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid_el.text}/",
                    "source": "PubMed"
                })
        return papers
    except Exception as e:
        print(f"PubMed error: {e}")
        return []


def search_arxiv(query, max_results=5):
    """Search arXiv for recent papers."""
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query} AND (cat:q-bio.NC OR cat:cs.LG)",
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        root = ET.fromstring(resp.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        
        papers = []
        for entry in root.findall("atom:entry", ns):
            title = entry.find("atom:title", ns)
            link = entry.find("atom:id", ns)
            
            if title is not None and link is not None:
                papers.append({
                    "title": " ".join(title.text.split()),
                    "url": link.text,
                    "source": "arXiv"
                })
        return papers
    except Exception as e:
        print(f"arXiv error: {e}")
        return []


def search_biorxiv(query, max_results=5):
    """Search bioRxiv for recent papers."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    url = f"https://api.biorxiv.org/details/biorxiv/{start_date:%Y-%m-%d}/{end_date:%Y-%m-%d}/0/50"
    
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        
        papers = []
        query_lower = query.lower()
        for item in data.get("collection", []):
            if query_lower in item.get("title", "").lower() or \
               query_lower in item.get("abstract", "").lower():
                papers.append({
                    "title": item["title"],
                    "url": f"https://doi.org/{item['doi']}",
                    "source": "bioRxiv"
                })
                if len(papers) >= max_results:
                    break
        return papers
    except Exception as e:
        print(f"bioRxiv error: {e}")
        return []


# ============== NEWSLETTER BUILDER ==============

def collect_papers():
    """Collect papers for all categories."""
    newsletter = {}
    
    for category, queries in CATEGORIES.items():
        print(f"Searching: {category}...")
        papers = []
        seen_titles = set()
        
        for query in queries:
            # Search all sources
            for paper in search_pubmed(query, 3):
                if paper["title"] not in seen_titles:
                    papers.append(paper)
                    seen_titles.add(paper["title"])
            
            for paper in search_arxiv(query, 3):
                if paper["title"] not in seen_titles:
                    papers.append(paper)
                    seen_titles.add(paper["title"])
            
            for paper in search_biorxiv(query, 2):
                if paper["title"] not in seen_titles:
                    papers.append(paper)
                    seen_titles.add(paper["title"])
        
        newsletter[category] = papers[:MAX_PAPERS_PER_CATEGORY]
    
    return newsletter


def format_newsletter(newsletter):
    """Format newsletter as HTML."""
    date_str = datetime.now().strftime("%B %d, %Y")
    
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 700px; margin: auto; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; }}
            h2 {{ color: #3498db; margin-top: 30px; }}
            .paper {{ margin: 15px 0; padding: 10px; background: #f8f9fa; border-radius: 5px; }}
            .paper a {{ color: #2980b9; text-decoration: none; font-weight: bold; }}
            .source {{ color: #7f8c8d; font-size: 12px; }}
        </style>
    </head>
    <body>
        <h1>🧠 Neuroscience Weekly</h1>
        <p><em>{date_str}</em></p>
    """
    
    for category, papers in newsletter.items():
        html += f"<h2>{category}</h2>"
        if papers:
            for paper in papers:
                html += f"""
                <div class="paper">
                    <a href="{paper['url']}">{paper['title']}</a>
                    <span class="source"> [{paper['source']}]</span>
                </div>
                """
        else:
            html += "<p><em>No new papers this week.</em></p>"
    
    html += """
        <hr>
        <p style="color: #95a5a6; font-size: 12px;">
            Generated by Neuroscience Newsletter Agent
        </p>
    </body>
    </html>
    """
    return html


def send_email(html_content):
    """Send newsletter via email."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🧠 Neuroscience Weekly - {datetime.now():%B %d, %Y}"
    msg["From"] = EMAIL_CONFIG["sender"]
    msg["To"] = EMAIL_CONFIG["recipient"]
    
    msg.attach(MIMEText(html_content, "html"))
    
    try:
        # Use SSL for Libero (port 465)
        if EMAIL_CONFIG.get("use_ssl", False):
            with smtplib.SMTP_SSL(EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"]) as server:
                server.login(EMAIL_CONFIG["sender"], EMAIL_CONFIG["password"])
                server.sendmail(
                    EMAIL_CONFIG["sender"],
                    EMAIL_CONFIG["recipient"],
                    msg.as_string()
                )
        else:
            # Use STARTTLS for Gmail (port 587)
            with smtplib.SMTP(EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"]) as server:
                server.starttls()
                server.login(EMAIL_CONFIG["sender"], EMAIL_CONFIG["password"])
                server.sendmail(
                    EMAIL_CONFIG["sender"],
                    EMAIL_CONFIG["recipient"],
                    msg.as_string()
                )
        print("✓ Newsletter sent!")
        return True
    except Exception as e:
        print(f"✗ Email error: {e}")
        return False


# ============== MAIN ==============

def run_newsletter():
    """Main function to collect and send newsletter."""
    print(f"\n{'='*50}")
    print(f"Running newsletter: {datetime.now()}")
    print('='*50)
    
    newsletter = collect_papers()
    html = format_newsletter(newsletter)
    send_email(html)


def main():
    """Run newsletter - supports scheduled mode or one-time run."""
    
    # One-time run (for GitHub Actions / cron)
    if "--run-once" in sys.argv:
        run_newsletter()
        return
    
    # Continuous scheduled mode (for local running)
    try:
        import schedule
    except ImportError:
        print("For scheduled mode: pip install schedule")
        print("Or use: python neuro_newsletter_agent.py --run-once")
        return
    
    print("Neuroscience Newsletter Agent Started")
    print("Scheduled: Every Monday at 8:00 AM")
    print("-" * 40)
    
    schedule.every().monday.at("08:00").do(run_newsletter)
    
    import time
    while True:
        schedule.run_pending()
        time.sleep(3600)


if __name__ == "__main__":
    main()
