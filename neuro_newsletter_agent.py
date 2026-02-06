"""
Neuroscience Weekly Newsletter Agent
Scrapes PubMed, arXiv, bioRxiv, and Google Scholar for recent papers.
Supports feedback.json for rejecting irrelevant papers.
"""

import requests
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import os
import sys
import json
from xml.etree import ElementTree as ET
from urllib.parse import quote as url_quote, parse_qs, urlparse
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

# ============== CONFIGURATION ==============
EMAIL_CONFIG = {
    "sender": os.getenv("EMAIL_SENDER", "your_email@libero.it"),
    "password": os.getenv("EMAIL_PASSWORD", "your_password"),
    "recipient": os.getenv("EMAIL_RECIPIENT", "your_email@libero.it"),
    "smtp_server": "smtp.libero.it",
    "smtp_port": 465,
    "use_ssl": True
}

# Target journals for PubMed filtering
TARGET_JOURNALS = [
    "Lancet", "Lancet Neurol", "Nature", "Nat Neurosci", "Nat Rev Neurosci",
    "Nat Med", "Nat Methods", "Neuron", "Cell", "Brain", "Neuroimage",
    "Hum Brain Mapp", "JAMA Neurol", "Ann Neurol", "Neurology",
    "Alzheimers Dement", "Headache", "Cephalalgia", "J Headache Pain"
]

# Refined categories with specific queries
CATEGORIES = {
    "Clinical Neurology": {
        "description": "Guidelines & management: migraine, dementias, psychotropic drugs",
        "queries": [
            "(migraine OR headache OR cephalalgia) AND (guideline OR management OR treatment protocol)",
            "(cluster headache OR tension headache) AND (clinical OR therapy)",
            "(Alzheimer OR dementia OR frontotemporal OR Lewy body) AND (guideline OR clinical management OR diagnosis criteria)",
            "(Parkinson disease dementia OR vascular dementia) AND (treatment OR management)",
            "(antipsychotic OR antidepressant OR anxiolytic) AND (neurology OR neurological) AND (guideline OR recommendation)",
        ]
    },
    "Cognitive Neuroscience": {
        "description": "Cognition, neuropsychology & neurodegeneration",
        "queries": [
            "(cognitive function OR cognition) AND (neural OR brain) AND (memory OR attention OR executive)",
            "(neuropsychological assessment OR cognitive testing) AND (dementia OR neurodegeneration)",
            "(Alzheimer OR frontotemporal OR Parkinson) AND (cognitive decline OR neuropsychological)",
            "(working memory OR episodic memory) AND (aging OR neurodegeneration)",
        ]
    },
    "AI in Neuroscience": {
        "description": "Machine learning for cognitive neuroscience",
        "queries": [
            "(machine learning OR deep learning) AND (cognitive neuroscience OR neuroimaging)",
            "(neural network OR transformer) AND (brain OR fMRI OR EEG)",
            "(artificial intelligence) AND (dementia OR Alzheimer) AND (prediction OR classification)",
            "(convolutional neural network OR random forest) AND (MRI OR brain imaging)",
        ]
    },
    "Neuroimaging Analysis": {
        "description": "MRI/PET pipelines & analysis methods",
        "queries": [
            "(fMRI analysis OR MRI preprocessing) AND (pipeline OR method OR tutorial)",
            "(FreeSurfer OR FSL OR SPM OR AFNI OR ANTs) AND (neuroimaging)",
            "(PET imaging OR PET analysis) AND (brain OR amyloid OR tau)",
            "(diffusion MRI OR DTI OR tractography) AND (method OR analysis)",
            "(structural MRI OR volumetric) AND (analysis method OR segmentation)",
            "(resting state OR functional connectivity) AND (analysis OR preprocessing)",
        ]
    }
}

MAX_PAPERS_PER_CATEGORY = 16

FEEDBACK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback.json")
REJECT_SERVER_PORT = 7878


# ============== FEEDBACK SYSTEM ==============

def load_feedback():
    """Load rejected titles from feedback.json."""
    if not os.path.exists(FEEDBACK_FILE):
        default = {"rejected_titles": []}
        with open(FEEDBACK_FILE, "w") as f:
            json.dump(default, f, indent=2)
        return default
    try:
        with open(FEEDBACK_FILE, "r") as f:
            data = json.load(f)
        rejected = data.get("rejected_titles", [])
        print(f"  Loaded {len(rejected)} rejected titles from feedback.json")
        return data
    except json.JSONDecodeError as e:
        print("\n" + "!" * 60)
        print("  ERROR: feedback.json is not valid JSON!")
        print(f"  Details: {e}")
        print("  Common fix: check for missing commas between entries.")
        print("  Feedback filtering is DISABLED for this run.")
        print("!" * 60 + "\n")
        return {"rejected_titles": []}
    except IOError as e:
        print(f"\n  ERROR: Cannot read feedback.json: {e}")
        return {"rejected_titles": []}


def calculate_similarity(new_title, rejected_titles):
    """Placeholder: will use embeddings later. Returns False for now."""
    return False


def _normalize_title(title):
    """Normalize a title for robust comparison."""
    import unicodedata
    import re
    t = unicodedata.normalize("NFKC", title)
    t = t.lower().strip()
    t = t.rstrip(".")
    t = re.sub(r"\s+", " ", t)
    return t


def is_rejected(title, rejected_titles):
    """Check if a paper should be rejected (normalized match or similarity)."""
    norm_title = _normalize_title(title)
    rejected_set = {_normalize_title(t) for t in rejected_titles}
    if norm_title in rejected_set:
        return True
    return calculate_similarity(norm_title, rejected_titles)


# ============== DATA SOURCES ==============

def search_pubmed(query, max_results=5, filter_journals=True):
    """Search PubMed with optional journal filtering."""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    date_range = f"{start_date:%Y/%m/%d}:{end_date:%Y/%m/%d}[pdat]"

    if filter_journals:
        journal_filter = " OR ".join([f'"{j}"[ta]' for j in TARGET_JOURNALS])
        full_query = f"({query}) AND ({journal_filter}) AND {date_range}"
    else:
        full_query = f"({query}) AND {date_range}"

    search_url = f"{base_url}/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": full_query,
        "retmax": max_results,
        "sort": "relevance",
        "retmode": "json"
    }

    try:
        resp = requests.get(search_url, params=params, timeout=15)
        ids = resp.json().get("esearchresult", {}).get("idlist", [])

        if not ids:
            if filter_journals:
                return search_pubmed(query, max_results, filter_journals=False)
            return []

        fetch_url = f"{base_url}/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "xml"
        }
        resp = requests.get(fetch_url, params=fetch_params, timeout=15)

        papers = []
        root = ET.fromstring(resp.content)
        for article in root.findall(".//PubmedArticle"):
            title_el = article.find(".//ArticleTitle")
            pmid_el = article.find(".//PMID")
            journal_el = article.find(".//Journal/Title")

            if title_el is not None and pmid_el is not None:
                title_text = "".join(title_el.itertext()) if title_el.text is None else title_el.text
                journal = journal_el.text if journal_el is not None else "PubMed"

                papers.append({
                    "title": title_text or "No title",
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid_el.text}/",
                    "source": journal[:30]
                })
        return papers
    except Exception as e:
        print(f"  PubMed error: {e}")
        return []


def search_arxiv(query, max_results=5):
    """Search arXiv for neuroscience/ML papers."""
    url = "http://export.arxiv.org/api/query"
    cats = "(cat:q-bio.NC OR cat:cs.LG OR cat:cs.CV OR cat:stat.ML OR cat:eess.IV)"

    params = {
        "search_query": f"all:{query} AND {cats}",
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        root = ET.fromstring(resp.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        papers = []
        for entry in root.findall("atom:entry", ns):
            title = entry.find("atom:title", ns)
            link = entry.find("atom:id", ns)
            published = entry.find("atom:published", ns)

            if published is not None:
                pub_date = datetime.fromisoformat(published.text.replace("Z", "+00:00"))
                if (datetime.now(pub_date.tzinfo) - pub_date).days > 7:
                    continue

            if title is not None and link is not None:
                papers.append({
                    "title": " ".join(title.text.split()),
                    "url": link.text,
                    "source": "arXiv"
                })
        return papers
    except Exception as e:
        print(f"  arXiv error: {e}")
        return []


def search_biorxiv(query, max_results=5):
    """Search bioRxiv/medRxiv for preprints."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    papers = []

    for server in ["biorxiv", "medrxiv"]:
        url = f"https://api.biorxiv.org/details/{server}/{start_date:%Y-%m-%d}/{end_date:%Y-%m-%d}/0/100"

        try:
            resp = requests.get(url, timeout=15)
            data = resp.json()

            query_terms = query.lower().split(" AND ")[0].replace("(", "").replace(")", "").split(" OR ")

            for item in data.get("collection", []):
                title_lower = item.get("title", "").lower()
                abstract_lower = item.get("abstract", "").lower()

                if any(term.strip() in title_lower or term.strip() in abstract_lower
                       for term in query_terms if len(term.strip()) > 3):
                    papers.append({
                        "title": item["title"],
                        "url": f"https://doi.org/{item['doi']}",
                        "source": server.capitalize()
                    })
                    if len(papers) >= max_results:
                        return papers
        except Exception as e:
            print(f"  {server} error: {e}")

    return papers


def search_google_scholar(query, max_results=3):
    """Search Google Scholar via SerpAPI (optional - set SERPAPI_KEY)."""
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        return []

    url = "https://serpapi.com/search"
    params = {
        "engine": "google_scholar",
        "q": query,
        "api_key": api_key,
        "num": max_results,
        "as_ylo": datetime.now().year,
        "scisbd": 1
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()

        papers = []
        for result in data.get("organic_results", []):
            papers.append({
                "title": result.get("title", "No title"),
                "url": result.get("link", ""),
                "source": "Scholar"
            })
        return papers
    except Exception as e:
        print(f"  Scholar error: {e}")
        return []


# ============== NEWSLETTER BUILDER ==============

def collect_papers():
    """Collect papers for all categories, filtering rejected ones."""
    newsletter = {}
    feedback = load_feedback()
    rejected_titles = feedback.get("rejected_titles", [])
    rejected_count = 0

    for category, config in CATEGORIES.items():
        print(f"\n  {category}...")
        papers = []
        seen_titles = set()

        for query in config["queries"]:
            print(f"  Query: {query[:50]}...")

            for paper in search_pubmed(query, 5):
                title_lower = paper["title"].lower()
                if title_lower not in seen_titles:
                    if is_rejected(paper["title"], rejected_titles):
                        rejected_count += 1
                        continue
                    papers.append(paper)
                    seen_titles.add(title_lower)

            for paper in search_arxiv(query, 4):
                title_lower = paper["title"].lower()
                if title_lower not in seen_titles:
                    if is_rejected(paper["title"], rejected_titles):
                        rejected_count += 1
                        continue
                    papers.append(paper)
                    seen_titles.add(title_lower)

            for paper in search_biorxiv(query, 3):
                title_lower = paper["title"].lower()
                if title_lower not in seen_titles:
                    if is_rejected(paper["title"], rejected_titles):
                        rejected_count += 1
                        continue
                    papers.append(paper)
                    seen_titles.add(title_lower)

            for paper in search_google_scholar(query, 3):
                title_lower = paper["title"].lower()
                if title_lower not in seen_titles:
                    if is_rejected(paper["title"], rejected_titles):
                        rejected_count += 1
                        continue
                    papers.append(paper)
                    seen_titles.add(title_lower)

        newsletter[category] = {
            "description": config["description"],
            "papers": papers[:MAX_PAPERS_PER_CATEGORY]
        }
        print(f"  Found: {len(papers)} papers")

    if rejected_count:
        print(f"\n  Filtered out {rejected_count} rejected papers")

    return newsletter


def format_newsletter(newsletter):
    """Format newsletter as HTML with Reject links."""
    date_str = datetime.now().strftime("%B %d, %Y")

    css = """
        <style>
            body { font-family: 'Segoe UI', Arial, sans-serif; max-width: 750px; margin: auto; padding: 20px; background: #f5f5f5; }
            .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
            h2 { color: #2980b9; margin-top: 35px; padding: 10px; background: #ecf0f1; border-radius: 5px; }
            .section-desc { color: #7f8c8d; font-size: 13px; margin-top: -5px; margin-bottom: 15px; }
            .paper { margin: 15px 0; padding: 15px 18px; background: linear-gradient(to right, #f8f9fa, #ffffff); border-left: 4px solid #3498db; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
            .paper:hover { background: linear-gradient(to right, #eef2f7, #ffffff); }
            .paper a.title-link { color: #2c3e50; text-decoration: none; font-weight: 600; font-size: 15px; line-height: 1.4; display: block; }
            .paper a.title-link:hover { color: #2980b9; }
            .source { color: #7f8c8d; font-size: 12px; display: inline-block; margin-top: 8px; background: #ecf0f1; padding: 3px 10px; border-radius: 12px; }
            .reject-link { color: #e74c3c; font-size: 12px; text-decoration: none; margin-left: 10px; cursor: pointer; }
            .reject-link:hover { text-decoration: underline; }
            .empty { color: #bdc3c7; font-style: italic; padding: 15px; }
            .footer { margin-top: 30px; padding-top: 20px; border-top: 1px solid #ecf0f1; color: #95a5a6; font-size: 11px; }
        </style>
    """

    html = f"""
    <html>
    <head>{css}</head>
    <body>
        <div class="container">
        <h1>&#x1F9E0; Neuroscience Weekly</h1>
        <p style="color: #7f8c8d;"><em>{date_str}</em></p>
    """

    icons = {
        "Clinical Neurology": "&#x1F3E5;",
        "Cognitive Neuroscience": "&#x1F9E9;",
        "AI in Neuroscience": "&#x1F916;",
        "Neuroimaging Analysis": "&#x1F52C;"
    }

    for category, data in newsletter.items():
        icon = icons.get(category, "&#x1F4C4;")
        count = len(data["papers"])
        html += f'<h2>{icon} {category} <span style="font-size:14px; color:#95a5a6; font-weight:normal;">({count} papers)</span></h2>'
        html += f'<p class="section-desc">{data["description"]}</p>'

        if data["papers"]:
            for i, paper in enumerate(data["papers"], 1):
                reject_url = f"http://localhost:{REJECT_SERVER_PORT}/reject?title={url_quote(paper['title'])}"
                html += f"""
                <div class="paper">
                    <span style="color:#3498db; font-weight:bold; margin-right:8px;">{i}.</span>
                    <a class="title-link" href="{paper['url']}" target="_blank">{paper['title']}</a>
                    <span class="source">&#x1F50E; {paper['source']}</span>
                    <a class="reject-link" href="{reject_url}" title="Reject this paper">[Reject]</a>
                </div>
                """
        else:
            html += '<p class="empty">No new papers this week.</p>'

    html += """
        <div class="footer">
            <p>Generated by Neuroscience Newsletter Agent<br>
            Sources: PubMed, arXiv, bioRxiv, medRxiv, Google Scholar<br>
            <em>Click [Reject] to remove a paper from future newsletters (requires reject server running)</em></p>
        </div>
        </div>
    </body>
    </html>
    """
    return html


def send_email(html_content):
    """Send newsletter via email."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Neuroscience Weekly - {datetime.now():%B %d, %Y}"
    msg["From"] = EMAIL_CONFIG["sender"]
    msg["To"] = EMAIL_CONFIG["recipient"]
    msg.attach(MIMEText(html_content, "html"))

    try:
        if EMAIL_CONFIG.get("use_ssl", False):
            with smtplib.SMTP_SSL(EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"]) as server:
                server.login(EMAIL_CONFIG["sender"], EMAIL_CONFIG["password"])
                server.sendmail(EMAIL_CONFIG["sender"], EMAIL_CONFIG["recipient"], msg.as_string())
        else:
            with smtplib.SMTP(EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"]) as server:
                server.starttls()
                server.login(EMAIL_CONFIG["sender"], EMAIL_CONFIG["password"])
                server.sendmail(EMAIL_CONFIG["sender"], EMAIL_CONFIG["recipient"], msg.as_string())
        print("\n  Newsletter sent!")
        return True
    except Exception as e:
        print(f"\n  Email error: {e}")
        return False


def run_newsletter():
    print(f"\n{'='*50}")
    print(f"  Neuroscience Newsletter - {datetime.now():%Y-%m-%d %H:%M}")
    print('='*50)

    newsletter = collect_papers()
    html = format_newsletter(newsletter)
    send_email(html)


def add_rejection(title):
    """Add a title to feedback.json rejected_titles list."""
    feedback = load_feedback()
    rejected = feedback.get("rejected_titles", [])

    # Check if already present (normalized)
    if _normalize_title(title) in {_normalize_title(t) for t in rejected}:
        print(f"  Already rejected: {title}")
        return False

    rejected.append(title)
    feedback["rejected_titles"] = rejected
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(feedback, f, indent=2, ensure_ascii=False)
    print(f"  Added to rejected list: {title}")
    print(f"  Total rejected: {len(rejected)}")
    return True


# ============== REJECT SERVER ==============

class RejectHandler(BaseHTTPRequestHandler):
    """Handles [Reject] clicks from the newsletter email."""

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/reject":
            params = parse_qs(parsed.query)
            title = params.get("title", [""])[0]
            if title:
                added = add_rejection(title)
                if added:
                    msg = f"Rejected: {title}"
                else:
                    msg = f"Already rejected: {title}"
                self._respond(200, msg)
            else:
                self._respond(400, "No title provided.")
        elif parsed.path == "/status":
            feedback = load_feedback()
            count = len(feedback.get("rejected_titles", []))
            self._respond(200, f"Reject server running. {count} titles rejected so far.")
        else:
            self._respond(404, "Not found.")

    def _respond(self, code, message):
        self.send_response(code)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        color = "#27ae60" if code == 200 else "#e74c3c"
        html = f"""<html><body style="font-family:sans-serif; display:flex; justify-content:center;
            align-items:center; height:80vh; background:#f5f5f5;">
            <div style="text-align:center; padding:40px; background:white; border-radius:10px;
            box-shadow:0 2px 10px rgba(0,0,0,0.1); max-width:600px;">
            <h2 style="color:{color};">{"&#x2714;" if code == 200 else "&#x2716;"} {message}</h2>
            <p style="color:#95a5a6;">You can close this tab.</p></div></body></html>"""
        self.wfile.write(html.encode())

    def log_message(self, format, *args):
        pass  # Suppress default logging


def start_reject_server():
    """Start the reject server in a background thread."""
    try:
        server = HTTPServer(("127.0.0.1", REJECT_SERVER_PORT), RejectHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f"  Reject server running at http://localhost:{REJECT_SERVER_PORT}")
        return server
    except OSError as e:
        print(f"  Could not start reject server: {e}")
        return None


def main():
    # --reject "title" : safely add a title to feedback.json
    if "--reject" in sys.argv:
        idx = sys.argv.index("--reject")
        if idx + 1 < len(sys.argv):
            add_rejection(sys.argv[idx + 1])
        else:
            print('Usage: python neuro_newsletter_agent_1.py --reject "Paper Title Here"')
        return

    # --server : just run the reject server (keep it open while reading emails)
    if "--server" in sys.argv:
        server = start_reject_server()
        if server:
            print(f"\n  Reject server is running on http://localhost:{REJECT_SERVER_PORT}")
            print("  Open your newsletter email and click [Reject] on papers you don't want.")
            print("  Press Ctrl+C to stop.\n")
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                print("\n  Server stopped.")
        return

    if "--run-once" in sys.argv:
        start_reject_server()  # also start server so rejects work immediately
        run_newsletter()
        return

    try:
        import schedule
    except ImportError:
        print("For scheduled mode: pip install schedule")
        print("Or use: python neuro_newsletter_agent.py --run-once")
        return

    print("Neuroscience Newsletter Agent Started")
    print("Scheduled: Every Monday at 8:00 AM")
    start_reject_server()
    schedule.every().monday.at("08:00").do(run_newsletter)

    import time
    while True:
        schedule.run_pending()
        time.sleep(3600)


if __name__ == "__main__":
    main()
