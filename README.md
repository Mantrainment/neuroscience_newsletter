# neuroscience_newsletter

# Neuroscience Newsletter Agent

Weekly email digest of papers from PubMed, arXiv, and bioRxiv.

## Categories
- Cognitive Neuroscience
- Clinical Neuroscience (guidelines, disease management)
- Neuroimaging Analysis
- AI Methods in Neuroscience

---

## Option 1: GitHub Actions (Recommended - Free & Automatic)

No server needed. GitHub runs it for you every week.

### Setup Steps:

1. **Create a GitHub repo** and push these files:
   ```
   your-repo/
   ├── neuro_newsletter_agent.py
   └── .github/workflows/newsletter.yml
   ```

2. **Add secrets** in your repo:
   - Go to: Settings → Secrets → Actions → New repository secret
   - Add these 3 secrets:
     | Name | Value |
     |------|-------|
     | `EMAIL_SENDER` | your_email@gmail.com |
     | `EMAIL_PASSWORD` | your Gmail App Password |
     | `EMAIL_RECIPIENT` | your_email@gmail.com |

3. **Test it**: 
   - Go to: Actions → "Weekly Neuroscience Newsletter" → Run workflow

Done! It will now run every Monday at 8 AM UTC automatically.

---

## Option 2: Local Setup

### Install & Configure
```bash
pip install requests schedule
# Edit EMAIL_CONFIG in the script
```

### Run once (test)
```bash
python neuro_newsletter_agent.py --run-once
```

### Run continuously
```bash
python neuro_newsletter_agent.py
```

---

## Gmail App Password Setup
1. Go to [Google Account Security](https://myaccount.google.com/security)
2. Enable 2-Step Verification
3. Go to App Passwords → Generate new
4. Use that 16-char password
