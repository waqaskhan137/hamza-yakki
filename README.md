> **Note:** My friend Hamza asked me to build this to help him fill out survey forms so he can win a prize worth £1,000. He only promised me a chocolate for doing this. 🍫
> 
> **This is Hamza:**
> 
> <img src="hamza.jpg" width="200" alt="Hamza" style="transform: rotate(90deg);">

# GSL Student Survey Form Filler

Automated browser tool to fill the GSL Student Survey using Playwright and SurveyJS API.

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install playwright groq
playwright install chromium

# Set API key
export GROQ_API_KEY="your_api_key_here"
```

## Usage

```bash
# Single form
python fill_form.py

# Multiple forms (sequential)
python fill_form.py -n 5

# Multiple forms (parallel browsers)
python fill_form.py -n 10 -t 3
```

## Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `-n, --count` | Number of forms to fill | 1 |
| `-t, --threads` | Parallel browser instances | 1 |

## How It Works

1. **Data Source**: Reads `form-data.txt` for question-answer mappings
2. **Browser Automation**: Uses Playwright to control Chromium
3. **SurveyJS API**: Directly manipulates survey via JavaScript
4. **LLM Fallback**: Groq API handles unmatched questions
5. **Output**: Saves QR codes to `qr_codes/` and logs to `results.csv`

## Files

- `fill_form.py` - Main script
- `form-data.txt` - Survey questions and answers
- `prompt.txt` - LLM system prompt
- `results.csv` - Submission log
- `qr_codes/` - Captured QR code screenshots

## Requirements

- Python 3.11+
- Groq API key
- Playwright
