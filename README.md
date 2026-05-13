# Multi-Model Chat App: OpenAI + Claude

This is a general-purpose Streamlit app that lets you ask one question once, upload attachments once, and send the same task to both OpenAI and Claude. The two model outputs can then review each other, and the app can generate a final synthesized answer.

## Files

- `multi_model_chat_app.py` — main Streamlit app
- `requirements.txt` — Python packages Streamlit Cloud will install
- `.gitignore` — prevents local secrets and private files from being uploaded to GitHub

## What the app can do

- Ask OpenAI and Claude the same question
- Upload files once
- Extract text from PDF, DOCX, TXT, Markdown, CSV, XLSX, JSON, XML, HTML, Python, R, SQL, JavaScript, TypeScript, and CSS files
- Show side-by-side initial answers
- Let both models review each other's answers
- Generate a final synthesized answer
- Download the final answer as a text file

## Important privacy note

Do not upload private, confidential, or identifiable student work unless your institution's policy allows sending that data to third-party AI APIs. For student work, anonymize names and identifiers whenever possible.

## API keys

You need API keys from both OpenAI and Anthropic.

Do not put API keys in GitHub.

For Streamlit Community Cloud, add them under:

`App settings` → `Secrets`

Use this TOML format:

```toml
OPENAI_API_KEY = "your_openai_api_key_here"
ANTHROPIC_API_KEY = "your_anthropic_api_key_here"
```

## Deploy on Streamlit Community Cloud

1. Create a new GitHub repository.
2. Upload these files:
   - `multi_model_chat_app.py`
   - `requirements.txt`
   - `README.md`
   - `.gitignore`
3. Go to Streamlit Community Cloud.
4. Click `Create app` or `New app`.
5. Connect your GitHub repository.
6. Select:
   - Branch: `main`
   - Main file path: `multi_model_chat_app.py`
7. Add your API keys in `Secrets`.
8. Deploy.

## Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a local secrets file:

```text
.streamlit/secrets.toml
```

Add:

```toml
OPENAI_API_KEY = "your_openai_api_key_here"
ANTHROPIC_API_KEY = "your_anthropic_api_key_here"
```

Run:

```bash
streamlit run multi_model_chat_app.py
```


## Default models

This version uses these default models:

```python
DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
```

You can still change the model names from the Streamlit sidebar after the app launches.

## Cost control

The app has two cost-control settings in the code:

```python
MAX_ATTACHMENT_CHARS = 60000
MAX_OUTPUT_TOKENS = 1800
```

Large attachments and long answers cost more because the app sends text to the APIs multiple times.

## Suggested first test

Use a short sample document, not real student work.

Example question:

> Summarize this document, identify the main issues, and suggest three improvements.

Then upload a small PDF or DOCX file.
