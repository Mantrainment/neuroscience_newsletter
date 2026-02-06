"""
Neuroscience Weekly Newsletter Agent
Scrapes PubMed, arXiv, bioRxiv, and Google Scholar for recent papers.
Supports feedback via GitHub Issues for rejecting irrelevant papers.
Uses title + abstract for filtering and query refinement.
"""

import requests
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import os
import sys
import json
import re
import unicodedata
from xml.etree import ElementTree as ET
from urllib.parse import quote as url_quote
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model once at startup
print("  Loading embedding model (all-MiniLM-L6-v2)...")
_EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
print("  Model loaded.")

SIMILARITY_THRESHOLD = 0.82
MIN_KEYWORD_FREQ = 2  # A word must appear in at least this many rejected titles to become a negative keyword

# ============== CONFIGURATION ==============
EMAIL_CONFIG = {
    "sender": os.getenv("EMAIL_SENDER", "your_email@libero.it"),
    "password": os.getenv("EMAIL_PASSWORD", "your_password"),
    "recipient": os.getenv("EMAIL_RECIPIENT", "your_email@libero.it"),
    "smtp_server": "smtp.libero.it",
    "smtp_port": 465,
    "use_ssl": True
}

# GitHub repo where this script lives (for reject issues)
# Format: "username/repo-name"
GITHUB_REPO = os.getenv("GITHUB_REPO", "Mantrainment/neuroscience_newsletter")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")  # optional, needed for private repos

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


# ============== FEEDBACK SYSTEM ==============

# Separator used in GitHub Issue body between title and abstract
_ISSUE_SEPARATOR = "\n---\n"


def load_feedback_file():
    """Load rejected papers from local feedback.json.

    Supports both old format (list of title strings) and new format
    (list of dicts with 'title' and 'abstract' keys).
    """
    if not os.path.exists(FEEDBACK_FILE):
        return []
    try:
        with open(FEEDBACK_FILE, "r") as f:
            data = json.load(f)
        raw = data.get("rejected_titles", [])  # key kept for backward compat
        papers = []
        for item in raw:
            if isinstance(item, dict):
                papers.append({"title": item.get("title", ""), "abstract": item.get("abstract", "")})
            else:
                # Old format: plain string = title only
                papers.append({"title": str(item), "abstract": ""})
        return papers
    except json.JSONDecodeError as e:
        print("\n" + "!" * 60)
        print("  ERROR: feedback.json is not valid JSON!")
        print(f"  Details: {e}")
        print("!" * 60 + "\n")
        return []
    except IOError:
        return []


def load_rejected_from_github():
    """Load rejected papers from GitHub Issues labeled 'reject'.

    New format: issue body contains 'TITLE\\n---\\nABSTRACT'.
    Old format (backward compat): issue body is just the title.
    """
    if not GITHUB_REPO or GITHUB_REPO == "your-username/your-repo-name":
        return []

    url = f"https://api.github.com/repos/{GITHUB_REPO}/issues"
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    params = {
        "labels": "reject",
        "state": "open",
        "per_page": 100
    }

    papers = []
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code == 200:
            for issue in resp.json():
                body = issue.get("body", "")
                if not body:
                    continue
                body = body.strip()
                if _ISSUE_SEPARATOR in body:
                    # New format: title + abstract
                    parts = body.split(_ISSUE_SEPARATOR, 1)
                    papers.append({"title": parts[0].strip(), "abstract": parts[1].strip()})
                else:
                    # Old format: title only
                    papers.append({"title": body, "abstract": ""})
            print(f"  Loaded {len(papers)} rejected papers from GitHub Issues")
        else:
            print(f"  GitHub Issues API error: {resp.status_code}")
    except Exception as e:
        print(f"  GitHub Issues error: {e}")

    return papers


def load_all_rejected():
    """Load rejected papers from both feedback.json and GitHub Issues.

    Returns list of dicts: [{"title": ..., "abstract": ...}, ...]
    """
    file_papers = load_feedback_file()
    github_papers = load_rejected_from_github()
    all_papers = file_papers + github_papers
    if all_papers:
        print(f"  Total rejected papers: {len(all_papers)} ({len(file_papers)} from file, {len(github_papers)} from GitHub)")
    return all_papers


def _normalize_title(title):
    """Normalize a title for robust comparison."""
    t = unicodedata.normalize("NFKC", title)
    t = t.lower().strip()
    t = t.rstrip(".")
    t = re.sub(r"\s+", " ", t)
    return t


# Cache for rejected paper embeddings (computed once per run)
_rejected_cache = {"papers": None, "embeddings": None, "texts": None}


def _get_rejected_embeddings(rejected_papers):
    """Compute and cache embeddings for rejected papers (title + abstract)."""
    if _rejected_cache["papers"] is not rejected_papers:
        _rejected_cache["papers"] = rejected_papers
        if rejected_papers:
            # Combine title + abstract for richer semantic matching
            texts = [f"{p['title']} {p['abstract']}".strip() for p in rejected_papers]
            _rejected_cache["texts"] = texts
            _rejected_cache["embeddings"] = _EMBED_MODEL.encode(
                texts, normalize_embeddings=True
            )
        else:
            _rejected_cache["texts"] = None
            _rejected_cache["embeddings"] = None
    return _rejected_cache["embeddings"]


def calculate_similarity(new_text, rejected_papers):
    """Check if new_text is semantically similar to any rejected paper using embeddings."""
    if not rejected_papers:
        return False

    rejected_embeddings = _get_rejected_embeddings(rejected_papers)
    if rejected_embeddings is None:
        return False

    new_embedding = _EMBED_MODEL.encode([new_text], normalize_embeddings=True)

    # Cosine similarity (dot product since embeddings are normalized)
    scores = np.dot(rejected_embeddings, new_embedding.T).flatten()
    max_idx = int(np.argmax(scores))
    max_score = float(scores[max_idx])

    if max_score >= SIMILARITY_THRESHOLD:
        rejected_title = rejected_papers[max_idx]["title"]
        print(f"    [Semantic Similarity] Rejected: '{new_text[:80]}...' "
              f"(score={max_score:.3f} vs '{rejected_title}')")
        return True
    return False


def is_rejected(title, abstract, rejected_papers):
    """Check if a paper should be rejected (normalized match or semantic similarity).

    Uses combined title + abstract for semantic comparison.
    """
    norm_title = _normalize_title(title)
    rejected_titles_set = {_normalize_title(p["title"]) for p in rejected_papers}
    if norm_title in rejected_titles_set:
        return True
    # Combine title + abstract for richer semantic matching
    combined = f"{title} {abstract}".strip()
    return calculate_similarity(combined, rejected_papers)


# ============== DYNAMIC QUERY REFINEMENT ==============

# Words that are too generic to be useful as negative keywords
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "not", "no", "nor", "so",
    "as", "if", "then", "than", "that", "this", "these", "those", "it",
    "its", "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "only", "own", "same", "too", "very", "just", "about",
    "above", "after", "again", "also", "any", "before", "between", "both",
    "during", "here", "how", "into", "new", "now", "over", "through",
    "under", "up", "out", "what", "when", "where", "which", "while", "who",
    "why", "using", "based", "via", "among", "across", "within", "without",
    # Domain-generic words that would over-filter neuroscience results
    "study", "analysis", "method", "approach", "model", "results", "effect",
    "effects", "role", "novel", "review", "data", "case", "patients",
    "associated", "related", "clinical", "human", "brain", "neural",
    "network", "learning", "deep", "machine", "imaging", "mri", "fmri",
}


def get_negative_keywords(rejected_papers):
    """Extract frequent significant words from rejected paper abstracts.

    Abstracts provide much richer and more specific terms than titles alone.
    Returns a list of words that appear in at least MIN_KEYWORD_FREQ
    rejected papers (excluding stop words), sorted by frequency.
    """
    from collections import Counter

    if not rejected_papers:
        return []

    word_counts = Counter()
    for paper in rejected_papers:
        # Combine title + abstract, but abstract is the main source of specific terms
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
        # Get unique words per paper (so one paper can't inflate a word's count)
        words = set(re.findall(r"[a-z0-9]{3,}", text.lower()))
        words -= STOP_WORDS
        word_counts.update(words)

    # Only keep words appearing in enough rejected papers
    keywords = [word for word, count in word_counts.most_common()
                if count >= MIN_KEYWORD_FREQ]

    if keywords:
        print(f"  Negative keywords (freq >= {MIN_KEYWORD_FREQ}): {keywords[:15]}...")

    return keywords


def refine_query(query, negative_keywords, max_negatives=3):
    """Append NOT terms to a query based on negative keywords.

    Only appends up to max_negatives terms to avoid overly restrictive queries.
    Skips keywords that already appear in the original query (to avoid
    negating something you're intentionally searching for).
    """
    if not negative_keywords:
        return query

    query_lower = query.lower()
    not_terms = []
    for kw in negative_keywords:
        if kw not in query_lower:
            not_terms.append(kw)
        if len(not_terms) >= max_negatives:
            break

    if not not_terms:
        return query

    not_clause = " ".join(f"NOT {term}" for term in not_terms)
    return f"({query}) {not_clause}"


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
            abstract_el = article.find(".//Abstract/AbstractText")

            if title_el is not None and pmid_el is not None:
                title_text = "".join(title_el.itertext()) if title_el.text is None else title_el.text
                journal = journal_el.text if journal_el is not None else "PubMed"

                # Collect all abstract parts (some have multiple AbstractText elements)
                abstract_parts = []
                for abs_part in article.findall(".//Abstract/AbstractText"):
                    part_text = "".join(abs_part.itertext())
                    if part_text:
                        abstract_parts.append(part_text)
                abstract = " ".join(abstract_parts)

                papers.append({
                    "title": title_text or "No title",
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid_el.text}/",
                    "source": journal[:30],
                    "abstract": abstract
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
                summary = entry.find("atom:summary", ns)
                abstract = " ".join(summary.text.split()) if summary is not None and summary.text else ""
                papers.append({
                    "title": " ".join(title.text.split()),
                    "url": link.text,
                    "source": "arXiv",
                    "abstract": abstract
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
                        "source": server.capitalize(),
                        "abstract": item.get("abstract", "")
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
                "source": "Scholar",
                "abstract": result.get("snippet", "")
            })
        return papers
    except Exception as e:
        print(f"  Scholar error: {e}")
        return []


# ============== NEWSLETTER BUILDER ==============

def collect_papers():
    """Collect papers for all categories, filtering rejected ones."""
    newsletter = {}
    rejected_papers = load_all_rejected()
    negative_kw = get_negative_keywords(rejected_papers)
    rejected_count = 0

    for category, config in CATEGORIES.items():
        print(f"\n  {category}...")
        papers = []
        seen_titles = set()

        for query in config["queries"]:
            refined = refine_query(query, negative_kw)
            if refined != query:
                print(f"  Query (refined): {refined[:80]}...")
            else:
                print(f"  Query: {query[:50]}...")

            for paper in search_pubmed(refined, 5):
                title_lower = paper["title"].lower()
                if title_lower not in seen_titles:
                    if is_rejected(paper["title"], paper.get("abstract", ""), rejected_papers):
                        rejected_count += 1
                        continue
                    papers.append(paper)
                    seen_titles.add(title_lower)

            for paper in search_arxiv(refined, 4):
                title_lower = paper["title"].lower()
                if title_lower not in seen_titles:
                    if is_rejected(paper["title"], paper.get("abstract", ""), rejected_papers):
                        rejected_count += 1
                        continue
                    papers.append(paper)
                    seen_titles.add(title_lower)

            for paper in search_biorxiv(query, 3):
                title_lower = paper["title"].lower()
                if title_lower not in seen_titles:
                    if is_rejected(paper["title"], paper.get("abstract", ""), rejected_papers):
                        rejected_count += 1
                        continue
                    papers.append(paper)
                    seen_titles.add(title_lower)

            for paper in search_google_scholar(query, 3):
                title_lower = paper["title"].lower()
                if title_lower not in seen_titles:
                    if is_rejected(paper["title"], paper.get("abstract", ""), rejected_papers):
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
    """Format newsletter as HTML with Reject links via GitHub Issues."""
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
                # [Reject] opens a pre-filled GitHub Issue with title + abstract
                issue_title = url_quote(f"Reject: {paper['title'][:80]}")
                # Body: title + separator + abstract (for richer filtering)
                abstract = paper.get("abstract", "")
                if abstract:
                    issue_body_raw = f"{paper['title']}{_ISSUE_SEPARATOR}{abstract}"
                else:
                    issue_body_raw = paper["title"]
                # GitHub URL has a ~8000 char limit; truncate abstract if needed
                if len(issue_body_raw) > 4000:
                    issue_body_raw = issue_body_raw[:4000] + "..."
                issue_body = url_quote(issue_body_raw)
                reject_url = f"https://github.com/{GITHUB_REPO}/issues/new?labels=reject&title={issue_title}&body={issue_body}"

                html += f"""
                <div class="paper">
                    <span style="color:#3498db; font-weight:bold; margin-right:8px;">{i}.</span>
                    <a class="title-link" href="{paper['url']}" target="_blank">{paper['title']}</a>
                    <span class="source">&#x1F50E; {paper['source']}</span>
                    <a class="reject-link" href="{reject_url}" target="_blank" title="Open GitHub Issue to reject this paper">[Reject]</a>
                </div>
                """
        else:
            html += '<p class="empty">No new papers this week.</p>'

    html += """
        <div class="footer">
            <p>Generated by Neuroscience Newsletter Agent<br>
            Sources: PubMed, arXiv, bioRxiv, medRxiv, Google Scholar<br>
            <em>Click [Reject] to open a GitHub Issue &mdash; just press "Submit" and the paper will be filtered next run</em></p>
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


def main():
    if "--run-once" in sys.argv:
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
    schedule.every().monday.at("08:00").do(run_newsletter)

    import time
    while True:
        schedule.run_pending()
        time.sleep(3600)


if __name__ == "__main__":
    main()
