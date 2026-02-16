"""
Automated form filler for GSL Student Survey.

Uses the SurveyJS JavaScript API (window.survey) to set values directly,
bypassing all DOM interaction issues. Falls back to LLM for edge cases.

- Fully autonomous — no manual intervention needed
- Uses survey.setValue() and survey.nextPage() via JS injection
- Pauses before final submission for user review
- LLM behavior governed by prompt.txt (editable)
- Supports parallel execution with -t/--threads
"""

import argparse
import csv
import json
import os
import random
import re
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from groq import Groq
from playwright.sync_api import sync_playwright

# ---------------------------------------------------------------------------
# Thread-safety locks
# ---------------------------------------------------------------------------

_csv_lock = threading.Lock()
_print_lock = threading.Lock()


def thread_print(*args, **kwargs):
    """Thread-safe print that prevents interleaved output."""
    with _print_lock:
        print(*args, **kwargs)


SURVEY_URL = "https://survey.gslglobal.com/"
FORM_DATA_FILE = "form-data.txt"
PROMPT_FILE = "prompt.txt"
RESULTS_CSV = "results.csv"
QR_CODES_DIR = "qr_codes"
LLM_MODEL = "llama-3.3-70b-versatile"
MAX_LLM_RETRIES = 2

# ---------------------------------------------------------------------------
# Random name generation
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda",
    "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Christopher", "Karen", "Charles", "Lisa", "Daniel", "Nancy",
    "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
    "Steven", "Kimberly", "Andrew", "Emily", "Paul", "Donna", "Joshua", "Michelle",
    "Kenneth", "Carol", "Kevin", "Amanda", "Brian", "Dorothy", "George", "Melissa",
    "Timothy", "Deborah", "Ronald", "Stephanie", "Edward", "Rebecca", "Jason", "Sharon",
    "Jeffrey", "Laura", "Ryan", "Cynthia", "Jacob", "Kathleen", "Gary", "Amy",
    "Nicholas", "Angela", "Eric", "Shirley", "Jonathan", "Anna", "Stephen", "Brenda",
    "Larry", "Pamela", "Justin", "Emma", "Scott", "Nicole", "Brandon", "Helen",
    "Benjamin", "Samantha", "Samuel", "Katherine", "Raymond", "Christine", "Gregory", "Debra",
    "Frank", "Rachel", "Alexander", "Carolyn", "Patrick", "Janet", "Jack", "Catherine",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
    "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill",
    "Flores", "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell",
    "Mitchell", "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz",
    "Parker", "Cruz", "Edwards", "Collins", "Reyes", "Stewart", "Morris", "Morales",
    "Murphy", "Cook", "Rogers", "Gutierrez", "Ortiz", "Morgan", "Cooper", "Peterson",
    "Bailey", "Reed", "Kelly", "Howard", "Ramos", "Kim", "Cox", "Ward",
    "Richardson", "Watson", "Brooks", "Chavez", "Wood", "James", "Bennett", "Gray",
    "Mendoza", "Ruiz", "Hughes", "Price", "Alvarez", "Castillo", "Sanders", "Patel",
]


def generate_random_name():
    return random.choice(FIRST_NAMES), random.choice(LAST_NAMES)


# ---------------------------------------------------------------------------
# Gmail alias
# ---------------------------------------------------------------------------

def make_unique_email(email):
    local, domain = email.split("@", 1)
    local = local.split("+")[0]
    return f"{local}+{int(time.time())}@{domain}"

# ---------------------------------------------------------------------------
# Groq LLM helper
# ---------------------------------------------------------------------------

def load_system_prompt():
    with open(PROMPT_FILE, "r") as f:
        return f.read()


def ask_llm(system_prompt, user_message):
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
        max_tokens=1024,
    )
    return resp.choices[0].message.content.strip()


def extract_code(llm_response):
    if "# NOT_ON_PAGE" in llm_response:
        return None
    if "# SKIP" in llm_response:
        return None
    m = re.search(r"```(?:python)?\s*\n(.*?)```", llm_response, re.DOTALL)
    if m:
        return m.group(1).strip()
    lines = llm_response.strip().split("\n")
    if any(line.strip().startswith(("page.", "time.", "import ", "#")) for line in lines):
        return llm_response.strip()
    return None


def get_page_html_snippet(page, max_len=6000):
    html = page.evaluate("""() => {
        const main = document.querySelector('#surveyElement') || document.body;
        return main.innerHTML;
    }""")
    if len(html) > max_len:
        html = html[:max_len] + "\n<!-- truncated -->"
    return html

# ---------------------------------------------------------------------------
# Form data parsing
# ---------------------------------------------------------------------------

def parse_form_data(filepath, tag=""):
    with open(filepath, "r", encoding="utf-8-sig") as f:
        content = f.read()

    blocks = re.split(r"\n\s*\n", content.strip())
    qa_pairs = []
    unique_email = None
    first_name, last_name = generate_random_name()
    thread_print(f"{tag}  Generated random name: {first_name} {last_name}")

    for block in blocks:
        lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
        if len(lines) >= 2:
            question, answers = lines[0], lines[1:]
            q_lower = question.strip().lower()

            # Replace email addresses with unique alias
            for i, ans in enumerate(answers):
                if re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", ans):
                    if unique_email is None:
                        unique_email = make_unique_email(ans)
                        thread_print(f"{tag}  Generated unique email: {unique_email}")
                    answers[i] = unique_email

            # Replace first name / last name with generated name
            if q_lower in ("first name", "first name*", "your first name"):
                answers = [first_name]
            elif q_lower in ("last name", "last name*", "surname", "your last name"):
                answers = [last_name]

            qa_pairs.append((question, answers))
        elif len(lines) == 1:
            qa_pairs.append((lines[0], []))
    return qa_pairs, unique_email or "unknown", first_name, last_name


def normalize(text):
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*\*\s*$", "", text)
    return text.lower()

# ---------------------------------------------------------------------------
# Title → SurveyJS question name mapping
# ---------------------------------------------------------------------------

def build_question_map(page):
    """Get all SurveyJS questions and build a title→name+type map."""
    questions = page.evaluate("""() => {
        return window.survey.getAllQuestions().map(q => {
            const info = {
                name: q.name,
                title: (q.title || q.name || '').replace(/<[^>]*>/g, '').trim(),
                type: q.getType(),
                visible: q.isVisible,
            };
            if (q.choices) {
                info.choices = q.choices.map(c =>
                    typeof c === 'object' ? {value: c.value, text: c.text || c.value} : {value: c, text: String(c)}
                );
            }
            if (q.rows && Array.isArray(q.rows)) {
                info.rows = q.rows.map(r =>
                    typeof r === 'object' ? {value: r.value, text: r.text || r.value} : {value: r, text: String(r)}
                );
            }
            if (q.columns && Array.isArray(q.columns)) {
                info.columns = q.columns.map(c =>
                    typeof c === 'object' ? {value: c.value, text: c.text || c.value} : {value: c, text: String(c)}
                );
            }
            return info;
        });
    }""")
    return questions


def match_question(form_question, survey_questions):
    """Match a form-data question text to a SurveyJS question by title."""
    q_norm = normalize(form_question)
    best = None
    best_score = 0

    for sq in survey_questions:
        title_norm = normalize(sq["title"])
        if not title_norm or len(title_norm) < 3:
            continue

        if q_norm == title_norm:
            return sq  # exact match

        # Containment match
        if q_norm in title_norm or title_norm in q_norm:
            score = 1.0 / (1 + abs(len(q_norm) - len(title_norm)))
            if score > best_score:
                best_score = score
                best = sq

    return best


def match_choice(answer, choices):
    """Match an answer text to a SurveyJS choice value."""
    answer_norm = normalize(answer)
    # Exact match on text
    for c in choices:
        if answer_norm == normalize(c["text"]):
            return c["value"]
    # Containment match
    for c in choices:
        c_norm = normalize(c["text"])
        if answer_norm in c_norm or c_norm in answer_norm:
            return c["value"]
    # Keyword match
    for c in choices:
        c_norm = normalize(c["text"])
        words = [w for w in answer_norm.split() if len(w) > 3]
        if words and all(w in c_norm for w in words):
            return c["value"]
    return None


def match_matrix_row(answer_text, rows):
    """Match an answer text to a matrix row."""
    a_norm = normalize(answer_text)
    for r in rows:
        r_norm = normalize(r["text"])
        if a_norm == r_norm or a_norm in r_norm or r_norm in a_norm:
            return r["value"]
    return None


# ---------------------------------------------------------------------------
# Core: set value via SurveyJS API
# ---------------------------------------------------------------------------

def set_survey_value(page, question_name, value):
    """Set a value on the survey model via JS."""
    page.evaluate(
        "(args) => window.survey.setValue(args.name, args.value)",
        {"name": question_name, "value": value}
    )


def fill_question(page, sq, answers, all_questions, tag=""):
    """Fill a single SurveyJS question. Returns True on success."""
    qname = sq["name"]
    qtype = sq["type"]
    answer = answers[0] if answers else ""

    try:
        if qtype in ("text", "comment"):
            set_survey_value(page, qname, answer)
            return True

        elif qtype == "boolean":
            val = answer.lower() in ("yes", "true", "1")
            set_survey_value(page, qname, val)
            return True

        elif qtype == "radiogroup":
            choices = sq.get("choices", [])
            matched = match_choice(answer, choices)
            if matched is not None:
                set_survey_value(page, qname, matched)
                return True

        elif qtype == "dropdown":
            choices = sq.get("choices", [])
            matched = match_choice(answer, choices)
            if matched is not None:
                set_survey_value(page, qname, matched)
                return True

        elif qtype == "checkbox":
            choices = sq.get("choices", [])
            values = []
            for ans in answers:
                matched = match_choice(ans, choices)
                if matched is not None:
                    values.append(matched)
            if values:
                set_survey_value(page, qname, values)
                return True

        elif qtype == "rating":
            # Star rating — try to parse a number
            num = re.search(r"\d+", answer)
            if num:
                set_survey_value(page, qname, int(num.group()))
                return True
            # If no answer, try setting 5 stars as default
            if not answer:
                set_survey_value(page, qname, 5)
                return True

        elif qtype == "matrix":
            rows = sq.get("rows", [])
            columns = sq.get("columns", [])
            if not rows or not columns:
                return False

            row_val = match_matrix_row(sq["title"], rows)
            if row_val:
                col_val = match_choice(answer, columns)
                if col_val:
                    current = page.evaluate(
                        "(name) => window.survey.getValue(name) || {}",
                        qname
                    )
                    current[row_val] = col_val
                    set_survey_value(page, qname, current)
                    return True

            return False

        elif qtype == "multipletext":
            items = page.evaluate("""(name) => {
                const q = window.survey.getQuestionByName(name);
                if (!q || !q.items) return [];
                return q.items.map(item => ({name: item.name, title: item.title || item.name}));
            }""", qname)

            if items and len(items) == 1:
                set_survey_value(page, qname, {items[0]["name"]: answer})
                return True
            elif items:
                val = {}
                for item in items:
                    for ans in answers:
                        val[item["name"]] = ans
                        break
                if val:
                    set_survey_value(page, qname, val)
                    return True

    except Exception as e:
        thread_print(f"{tag}    Error setting {qname}: {e}")

    return False


def fill_matrix_rows(page, sq, remaining_questions):
    """Special handler: match form-data entries to matrix rows and fill them all at once."""
    rows = sq.get("rows", [])
    columns = sq.get("columns", [])
    qname = sq["name"]
    if not rows or not columns:
        return []

    matrix_data = page.evaluate(
        "(name) => window.survey.getValue(name) || {}",
        qname
    )
    matched_questions = []

    for form_q, form_answers in remaining_questions:
        if not form_answers:
            continue
        q_norm = normalize(form_q)
        answer = form_answers[0]

        for row in rows:
            row_norm = normalize(row["text"])
            if q_norm == row_norm or q_norm in row_norm or row_norm in q_norm:
                col_val = match_choice(answer, columns)
                if col_val:
                    matrix_data[row["value"]] = col_val
                    matched_questions.append(form_q)
                break

    if matrix_data:
        set_survey_value(page, qname, matrix_data)

    return matched_questions

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

def handle_intro_page(page, tag=""):
    """Accept terms and click Start/Next."""
    thread_print(f"{tag}\n  Handling intro page...")
    try:
        page.get_by_text("I have read and accept", exact=False).first.click()
        time.sleep(1)
    except Exception:
        try:
            page.query_selector("input[type='checkbox']").check(force=True)
            time.sleep(0.5)
        except Exception:
            pass

    for sel in [".sd-navigation__next-btn", ".sd-navigation__start-btn",
                "input[value='Next']", "input[value='Start']"]:
        try:
            el = page.query_selector(sel)
            if el and el.is_visible():
                el.click()
                time.sleep(1)
                break
        except Exception:
            pass

    try:
        page.wait_for_load_state("networkidle", timeout=10000)
    except Exception:
        pass
    time.sleep(2)
    thread_print(f"{tag}  Intro page done.")


def next_page(page):
    """Advance to the next survey page via JS API."""
    result = page.evaluate("() => { return window.survey.nextPage(); }")
    time.sleep(1.5)
    return result


def submit_survey(page):
    """Submit the survey on the last page."""
    page.evaluate("() => window.survey.completeLastPage()")
    time.sleep(3)


def capture_qr_and_log(page, email, first_name, last_name, tag=""):
    """Capture the QR code from the completion page and log to CSV."""
    Path(QR_CODES_DIR).mkdir(exist_ok=True)

    # Wait for completion page to render
    time.sleep(3)

    # Try to find and screenshot the QR code element
    qr_path = None
    safe_email = re.sub(r"[^a-zA-Z0-9@+._-]", "_", email)
    timestamp = int(time.time())

    # Look for QR code: could be <img>, <canvas>, or <svg>
    for selector in [
        "img[src*='qr']", "img[alt*='QR']", "img[alt*='qr']",
        "canvas", "svg", ".qr-code", "[class*='qr']", "[class*='QR']",
        "#qr", "img[src*='data:image']",
    ]:
        try:
            el = page.query_selector(selector)
            if el and el.is_visible():
                qr_path = f"{QR_CODES_DIR}/{safe_email}_{timestamp}.png"
                el.screenshot(path=qr_path)
                thread_print(f"{tag}  QR code saved: {qr_path}")
                break
        except Exception:
            continue

    # Fallback: screenshot the whole page if no QR element found
    if qr_path is None:
        qr_path = f"{QR_CODES_DIR}/{safe_email}_{timestamp}_fullpage.png"
        page.screenshot(path=qr_path)
        thread_print(f"{tag}  QR not found as element — full page screenshot saved: {qr_path}")

    # Append to CSV (thread-safe)
    with _csv_lock:
        csv_exists = Path(RESULTS_CSV).exists()
        with open(RESULTS_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            if not csv_exists:
                writer.writerow(["email", "first_name", "last_name", "timestamp", "qr_screenshot"])
            writer.writerow([email, first_name, last_name, time.strftime("%Y-%m-%d %H:%M:%S"), qr_path])
    thread_print(f"{tag}  Logged to {RESULTS_CSV}: {first_name} {last_name} / {email}")

    return qr_path


def get_current_page_info(page):
    """Get current page number and questions on it."""
    return page.evaluate("""() => {
        const pageNo = window.survey.currentPageNo;
        const totalPages = window.survey.visiblePageCount;
        const currentPage = window.survey.currentPage;
        const questions = currentPage ? currentPage.questions.map(q => q.name) : [];
        return {pageNo, totalPages, questions};
    }""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fill_once(system_prompt, run_num):
    """Run a single form fill with its own Playwright instance (thread-safe)."""
    tag = f"[Run {run_num}]"
    qa_pairs, unique_email, first_name, last_name = parse_form_data(FORM_DATA_FILE, tag)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False, slow_mo=50)
        ctx = browser.new_context(viewport={"width": 1280, "height": 900})
        page = ctx.new_page()

        try:
            thread_print(f"{tag}\n  Navigating to {SURVEY_URL}...")
            page.goto(SURVEY_URL, wait_until="networkidle")
            time.sleep(3)

            handle_intro_page(page, tag)

            # Build question map from SurveyJS model
            survey_questions = build_question_map(page)
            thread_print(f"{tag}  Survey has {len(survey_questions)} questions in model")

            # Match form-data to survey questions
            remaining = list(qa_pairs)
            filled_count = 0
            unmatched = []

            # First pass: direct title matches (non-matrix)
            matrix_questions = [sq for sq in survey_questions if sq["type"] == "matrix"]

            for form_q, form_answers in remaining:
                q_short = form_q[:60] + ("..." if len(form_q) > 60 else "")
                sq = match_question(form_q, survey_questions)

                if sq is None:
                    unmatched.append((form_q, form_answers))
                    continue

                if sq["type"] == "matrix":
                    unmatched.append((form_q, form_answers))
                    continue

                if fill_question(page, sq, form_answers, survey_questions, tag):
                    thread_print(f"{tag}  [OK]   {q_short}")
                    filled_count += 1
                else:
                    unmatched.append((form_q, form_answers))
                    thread_print(f"{tag}  [FAIL] {q_short} (type={sq['type']})")

            # Second pass: matrix questions (batch fill rows)
            for msq in matrix_questions:
                matched_qs = fill_matrix_rows(page, msq, unmatched)
                if matched_qs:
                    for mq in matched_qs:
                        q_short = mq[:60] + ("..." if len(mq) > 60 else "")
                        thread_print(f"{tag}  [OK]   {q_short} (matrix: {msq['name']})")
                        filled_count += 1
                    unmatched = [(q, a) for q, a in unmatched if q not in matched_qs]

            # Third pass: LLM for anything still unmatched
            still_unmatched = []
            if unmatched:
                thread_print(f"{tag}\n  {len(unmatched)} questions unmatched, asking LLM...")
                sq_summary = json.dumps(
                    [{"name": sq["name"], "title": sq["title"][:80], "type": sq["type"]}
                     for sq in survey_questions if sq["type"] != "html"],
                    indent=2
                )
                for form_q, form_answers in unmatched:
                    q_short = form_q[:60] + ("..." if len(form_q) > 60 else "")
                    answers_str = "\n".join(form_answers) if form_answers else "(none)"
                    msg = (
                        f"I need to set a value for this form question but couldn't auto-match it.\n\n"
                        f"**Form question:** {form_q}\n"
                        f"**Answer(s):**\n{answers_str}\n\n"
                        f"**Available SurveyJS questions (name | title | type):**\n{sq_summary}\n\n"
                        f"The survey model is accessible via `window.survey`. Use `page.evaluate()` "
                        f"to call `window.survey.setValue(questionName, value)` or any other SurveyJS API.\n"
                        f"Return Python Playwright code using the `page` object."
                    )
                    for attempt in range(MAX_LLM_RETRIES):
                        try:
                            resp = ask_llm(system_prompt, msg)
                            code = extract_code(resp)
                            if code is None:
                                break
                            exec(code, {"page": page, "time": time, "re": re, "json": json})
                            thread_print(f"{tag}  [LLM]  {q_short}")
                            filled_count += 1
                            break
                        except Exception as e:
                            if attempt < MAX_LLM_RETRIES - 1:
                                msg += f"\n\nPrevious code failed: {e}\nPlease fix."
                            else:
                                still_unmatched.append((form_q, form_answers))
                                thread_print(f"{tag}  [MISS] {q_short}")
                    else:
                        still_unmatched.append((form_q, form_answers))

            # Now navigate through pages
            thread_print(f"{tag}\n  All values set. Navigating through survey pages...")
            max_nav = 50
            for i in range(max_nav):
                info = get_current_page_info(page)
                page_str = f"{tag}  Page {info['pageNo']+1}/{info['totalPages']}"

                # Check for validation errors and try to fix
                errors = page.evaluate("""() => {
                    const page = window.survey.currentPage;
                    if (!page) return [];
                    return page.questions
                        .filter(q => q.errors && q.errors.length > 0)
                        .map(q => ({name: q.name, title: q.title, errors: q.errors.map(e => e.text)}));
                }""")
                if errors:
                    thread_print(f"{page_str} — {len(errors)} validation error(s)")
                    for err in errors:
                        thread_print(f"{tag}    {err['name']}: {err['errors']}")
                else:
                    thread_print(page_str)

                advanced = next_page(page)
                if not advanced:
                    # Check if we're on the last page
                    is_last = page.evaluate("() => window.survey.isLastPage")
                    if is_last:
                        thread_print(f"{tag}  Reached last page — submitting...")
                        submit_survey(page)
                        capture_qr_and_log(page, unique_email, first_name, last_name, tag)
                        break

                    # Auto-fix: fill required empty questions with defaults
                    fixed = page.evaluate("""() => {
                        const pg = window.survey.currentPage;
                        if (!pg) return 0;
                        let fixed = 0;
                        for (const q of pg.questions) {
                            if (!q.isVisible || !q.isRequired) continue;
                            const val = window.survey.getValue(q.name);
                            if (val !== null && val !== undefined && val !== '') continue;
                            const type = q.getType();

                            if (type === 'matrix' && Array.isArray(q.rows) && Array.isArray(q.columns)) {
                                const data = {};
                                const defaultCol = q.columns[0].value || q.columns[0];
                                for (const row of q.rows) {
                                    const rv = row.value || row;
                                    data[rv] = defaultCol;
                                }
                                window.survey.setValue(q.name, data);
                                fixed++;
                            } else if (type === 'radiogroup' && q.choices && q.choices.length > 0) {
                                const first = q.choices[0];
                                window.survey.setValue(q.name, first.value || first);
                                fixed++;
                            } else if (type === 'checkbox' && q.choices && q.choices.length > 0) {
                                const first = q.choices[0];
                                window.survey.setValue(q.name, [first.value || first]);
                                fixed++;
                            } else if (type === 'dropdown' && q.choices && q.choices.length > 0) {
                                const first = q.choices[0];
                                window.survey.setValue(q.name, first.value || first);
                                fixed++;
                            } else if (type === 'text' || type === 'comment') {
                                window.survey.setValue(q.name, 'N/A');
                                fixed++;
                            } else if (type === 'boolean') {
                                window.survey.setValue(q.name, true);
                                fixed++;
                            } else if (type === 'rating') {
                                window.survey.setValue(q.name, 3);
                                fixed++;
                            }
                        }
                        return fixed;
                    }""")

                    if fixed > 0:
                        thread_print(f"{tag}  Auto-filled {fixed} required empty question(s) with defaults")
                        if next_page(page):
                            continue

                    # Still stuck — ask LLM
                    thread_print(f"{tag}  Validation errors — asking LLM...")
                    errors = page.evaluate("""() => {
                        const pg = window.survey.currentPage;
                        if (!pg) return [];
                        return pg.questions
                            .filter(q => q.errors && q.errors.length > 0)
                            .map(q => ({
                                name: q.name,
                                title: q.title || q.name,
                                type: q.getType(),
                                errors: q.errors.map(e => e.text),
                                choices: q.choices ? q.choices.slice(0, 10).map(c =>
                                    typeof c === 'object' ? c.text || c.value : c
                                ) : [],
                            }));
                    }""")
                    for err in errors:
                        thread_print(f"{tag}    {err['name']}: {err['errors']}")
                    html = get_page_html_snippet(page, 4000)
                    err_msg = (
                        f"Survey is stuck with validation errors. I need to fix these:\n\n"
                        f"**Errors:** {json.dumps(errors, indent=2)}\n\n"
                        f"**Page HTML:** ```html\n{html}\n```\n\n"
                        f"Use `page.evaluate()` with `window.survey.setValue(name, value)` to fix. "
                        f"Return Python code."
                    )
                    try:
                        resp = ask_llm(system_prompt, err_msg)
                        code = extract_code(resp)
                        if code:
                            exec(code, {"page": page, "time": time, "re": re, "json": json})
                            time.sleep(0.5)
                            if not next_page(page):
                                thread_print(f"{tag}  Still stuck. Force-skipping page...")
                                page.evaluate("() => { window.survey.currentPageNo++; }")
                                time.sleep(1)
                    except Exception as e:
                        thread_print(f"{tag}  LLM fix failed: {e}. Force-skipping...")
                        page.evaluate("() => { window.survey.currentPageNo++; }")
                        time.sleep(1)

            # Final report
            thread_print(f"{tag}\n{'='*60}")
            thread_print(f"{tag} FORM FILLING COMPLETE — {filled_count}/{len(qa_pairs)} questions set")
            thread_print(f"{tag} Name used: {first_name} {last_name}")
            thread_print(f"{tag} Email used: {unique_email}")
            if still_unmatched:
                thread_print(f"{tag}\n{len(still_unmatched)} questions could NOT be filled:")
                for q, a in still_unmatched:
                    thread_print(f"{tag}  - {q[:80]}")
            thread_print(f"{'='*60}")

            time.sleep(3)
            return len(still_unmatched) == 0

        finally:
            browser.close()


def main():
    parser = argparse.ArgumentParser(description="Auto-fill GSL Student Survey")
    parser.add_argument("-n", "--count", type=int, default=1,
                        help="Number of times to fill the form (default: 1)")
    parser.add_argument("-t", "--threads", type=int, default=1,
                        help="Number of parallel browser instances (default: 1)")
    args = parser.parse_args()

    system_prompt = load_system_prompt()
    print(f"Loaded LLM system prompt from {PROMPT_FILE}")
    print(f"Using Groq model: {LLM_MODEL}")
    print(f"Will fill form {args.count} time(s) with {args.threads} thread(s)\n")

    if args.threads <= 1:
        # Sequential mode — same behavior as before
        for i in range(1, args.count + 1):
            print(f"\n{'#'*60}")
            print(f"  RUN {i} / {args.count}")
            print(f"{'#'*60}")

            success = fill_once(system_prompt, i)
            print(f"\n  Run {i}: {'SUCCESS' if success else 'PARTIAL'}")

            if i < args.count:
                print(f"  Next run in 3 seconds...")
                time.sleep(3)
    else:
        # Parallel mode
        results = {}
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = {
                executor.submit(fill_once, system_prompt, i): i
                for i in range(1, args.count + 1)
            }
            for future in as_completed(futures):
                run_num = futures[future]
                try:
                    success = future.result()
                    results[run_num] = "SUCCESS" if success else "PARTIAL"
                except Exception as e:
                    results[run_num] = f"ERROR: {e}"
                    thread_print(f"[Run {run_num}] Failed with exception: {e}")

        # Summary
        print(f"\n{'='*60}")
        print(f"ALL RUNS COMPLETE — {args.count} run(s), {args.threads} thread(s)")
        print(f"{'='*60}")
        for run_num in sorted(results):
            print(f"  Run {run_num}: {results[run_num]}")

    print(f"\n{'='*60}")
    print(f"ALL DONE — completed {args.count} run(s).")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
