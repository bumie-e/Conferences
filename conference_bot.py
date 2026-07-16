#!/usr/bin/env python3
"""
Daily conference/workshop deadline tracker for:
  - Reinforcement Learning
  - Robot Learning
  - Related AI/ML applications
Sends a formatted email digest via Gmail SMTP.

Credentials are read from environment variables (set as GitHub Secrets):
  GMAIL_ADDRESS      — the Gmail account used to send
  GMAIL_APP_PASSWORD — 16-char App Password from Google account settings
  RECIPIENT_EMAIL    — destination address (defaults to GMAIL_ADDRESS)
"""

import os
import re
import smtplib
import logging
from datetime import datetime, timedelta, date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

KEYWORDS = [
    # Core topics
    "reinforcement learning", "robot learning", "robotics",
    "autonomous systems", "multi-agent", "policy optimization",
    "deep reinforcement learning", "imitation learning",
    "sim-to-real", "embodied ai", "motion planning",
    "manipulation", "locomotion", "legged", "humanoid",
    "dexterous", "sim2real",
    # Broader RL / decision-making
    "reward", "offline rl", "online rl", "rlhf",
    "decision making", "world model", "model-based",
    "safe ai", "ai safety", "alignment",
    "foundation model for robot", "robot foundation",
    "language model for robot", "llm for robot",
    "generalist agent", "generalist robot",
    # Application domains the user asked for
    "robot manipulation", "legged robot", "aerial robot",
    "autonomous driving", "autonomous vehicle",
    "navigation", "planning under uncertainty",
    # Also catch common workshop naming patterns
    "world models", "offline dataset", "online adaptation",
    "game-theoretic", "lifelong agent", "agentic system",
]

FLAGSHIP_VENUES = {
    "neurips", "icml", "iclr", "corl", "rss", "icra", "iros",
    "cvpr", "iccv", "eccv", "aamas", "aaai", "ijcai", "aistats",
    "l4dc", "rlc",
}

LOOKAHEAD_DAYS = 90
LOOKBACK_DAYS  = 3


# ── Data sources ──────────────────────────────────────────────────────────────

# Conference prefixes to scan on OpenReview (year updated annually as new ones appear)
OPENREVIEW_PREFIXES = [
    "NeurIPS.cc/2026/Workshop/",
    "ICML.cc/2026/Workshop/",
    "ICLR.cc/2026/Workshop/",
    "ICLR.cc/2027/Workshop/",
    "NeurIPS.cc/2027/Workshop/",
]

def fetch_aideadlines() -> List[Dict]:
    """Pull the ai-deadlines YAML from GitHub — most reliable structured source."""
    url = "https://raw.githubusercontent.com/abhshkdz/ai-deadlines/gh-pages/_data/conferences.yml"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        entries: List[Dict] = []
        current: Dict = {}
        for line in resp.text.splitlines():
            line = line.rstrip()
            if line.startswith("- title:"):
                if current:
                    entries.append(current)
                current = {"title": line.split(":", 1)[1].strip().strip("'\"")}
            elif line.startswith("  ") and ":" in line and current:
                key, _, val = line.strip().partition(":")
                current[key.strip()] = val.strip().strip("'\"")
        if current:
            entries.append(current)
        return entries
    except Exception as exc:
        log.warning("aideadlines fetch failed: %s", exc)
        return []


def _or_parse_deadline_str(date_field: str) -> str:
    """Extract the date part from strings like 'Submission Deadline: Aug 30 2026 12:29PM UTC-0'."""
    # Strip leading label
    for prefix in ("submission deadline:", "deadline:", "due:", "paper deadline:"):
        idx = date_field.lower().find(prefix)
        if idx != -1:
            date_field = date_field[idx + len(prefix):].strip()
            break
    # Keep only the date portion (drop time/timezone)
    parts = date_field.split()
    # Expect: Month Day Year [Time] [TZ]
    if len(parts) >= 3:
        return " ".join(parts[:3])
    return date_field.strip()


def fetch_openreview() -> List[Dict]:
    """Query the OpenReview API for workshop submission deadlines.

    Three-pass strategy (minimises HTTP calls):
      1. For each conference prefix, list Workshop sub-group IDs.
      2. Batch-fetch group details for title / location / website.
      3. Batch-fetch /-/Submission invitations to get precise duedate timestamps.
    """
    import time

    results: List[Dict] = []
    ua   = {"User-Agent": "conference-bot/1.0", "Accept": "application/json"}
    base = "https://api2.openreview.net"
    BATCH = 25

    def _batched_get(endpoint: str, ids: List[str]) -> List[Dict]:
        out = []
        for i in range(0, len(ids), BATCH):
            time.sleep(0.4)
            r = requests.get(
                f"{base}/{endpoint}",
                params={"ids": ",".join(ids[i: i + BATCH])},
                headers=ua,
                timeout=25,
            )
            if r.ok:
                key = "groups" if endpoint == "groups" else "invitations"
                out.extend(r.json().get(key, []))
            else:
                log.warning("OpenReview %s batch failed: %s", endpoint, r.status_code)
        return out

    def _ms_to_date(ms) -> str:
        """Convert millisecond epoch timestamp to 'YYYY-MM-DD' string."""
        try:
            return datetime.utcfromtimestamp(int(ms) / 1000).strftime("%Y-%m-%d")
        except Exception:
            return ""

    all_group_ids: List[str] = []

    # Pass 1 — collect group IDs
    for prefix in OPENREVIEW_PREFIXES:
        try:
            depth = prefix.count("/")
            r = requests.get(
                f"{base}/groups",
                params={"prefix": prefix, "limit": 150, "select": "id"},
                headers=ua,
                timeout=20,
            )
            if not r.ok:
                log.warning("OpenReview prefix listing failed for %s: %s", prefix, r.status_code)
                continue
            ids = [g["id"] for g in r.json().get("groups", [])
                   if g["id"].count("/") == depth]  # direct children only
            log.info("  OpenReview %s → %d workshops", prefix.rstrip("/"), len(ids))
            all_group_ids.extend(ids)
            time.sleep(0.5)
        except Exception as exc:
            log.warning("OpenReview prefix listing error for %s: %s", prefix, exc)

    if not all_group_ids:
        return results

    # Pass 2 — fetch group details (title, location, website, fallback deadline string)
    groups_by_id: Dict[str, Dict] = {}
    for grp in _batched_get("groups", all_group_ids):
        groups_by_id[grp["id"]] = grp

    # Pass 3 — fetch submission invitations for duedate
    inv_ids = [f"{gid}/-/Submission" for gid in all_group_ids]
    duedates: Dict[str, str] = {}
    for inv in _batched_get("invitations", inv_ids):
        gid = inv["id"].replace("/-/Submission", "")
        ms  = inv.get("duedate")
        if ms:
            duedates[gid] = _ms_to_date(ms)

    # Merge
    for gid in all_group_ids:
        grp     = groups_by_id.get(gid, {"id": gid, "content": {}})
        content = grp.get("content", {})

        def val(key: str) -> str:
            v = content.get(key, {})
            return (v.get("value", "") if isinstance(v, dict) else str(v)) or ""

        title    = val("title") or gid.split("/")[-1]
        subtitle = val("subtitle")
        location = val("location")
        website  = val("website")
        start    = val("start_date")
        if start and str(start).isdigit():
            start = _ms_to_date(start)

        # Prefer invitation duedate; fall back to parsing content.date string
        deadline = duedates.get(gid) or _or_parse_deadline_str(val("date"))

        results.append({
            "title":     subtitle or title,
            "full_name": title,
            "deadline":  deadline,
            "date":      start,
            "location":  location,
            "url":       website or f"https://openreview.net/group?id={gid}",
            "source":    "OpenReview",
            "tags":      title + " " + subtitle,
        })

    return results


def fetch_wikicfp() -> List[Dict]:
    """Search WikiCFP for RL and robotics CFPs.

    Results table row layout (two rows per entry):
      Row A (2 cells): acronym-link | full_name
      Row B (3 cells): when | where | deadline
    The header row has 4 cells: Event | When | Where | Deadline
    We identify the right table by that header signature.
    """
    results: List[Dict] = []
    search_terms = [
        "reinforcement+learning",
        "robot+learning",
        "robotics",
        "robot+manipulation",
        "autonomous+systems",
        "multi-agent+systems",
    ]
    ua = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

    for term in search_terms:
        url = f"http://www.wikicfp.com/cfp/servlet/tool.search?q={term}&b=1"
        try:
            resp = requests.get(url, timeout=25, headers=ua)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # Identify the results table by its 4-cell header row
            result_table = None
            for table in soup.find_all("table"):
                header = table.find("tr")
                if header:
                    hcells = [c.get_text(strip=True) for c in header.find_all("td")]
                    if hcells == ["Event", "When", "Where", "Deadline"]:
                        result_table = table
                        break

            if result_table is None:
                log.warning("WikiCFP: could not find results table for '%s'", term)
                continue

            rows = result_table.find_all("tr")
            i = 1  # skip header row
            while i < len(rows):
                row_a = rows[i].find_all("td")
                if len(row_a) == 2 and i + 1 < len(rows):
                    row_b = rows[i + 1].find_all("td")
                    link = row_a[0].find("a")
                    if link and len(row_b) == 3:
                        deadline = row_b[2].get_text(strip=True).split("(")[0].strip()
                        results.append({
                            "title":     link.get_text(strip=True),
                            "full_name": row_a[1].get_text(strip=True),
                            "date":      row_b[0].get_text(strip=True),
                            "location":  row_b[1].get_text(strip=True),
                            "deadline":  deadline,
                            "source":    "WikiCFP",
                            "url":       "http://www.wikicfp.com" + (link.get("href") or ""),
                        })
                    i += 2
                else:
                    i += 1
        except Exception as exc:
            log.warning("WikiCFP fetch failed for '%s': %s", term, exc)

    return results


# ── Filtering & deduplication ─────────────────────────────────────────────────

def parse_deadline(raw: str) -> Optional[date]:
    raw = raw.strip().rstrip("*").strip()
    # Trim trailing time component if present (e.g. "2026-10-10 23:59:59")
    if len(raw) > 10 and raw[10] == " ":
        raw = raw[:10]
    for fmt in ("%Y-%m-%d", "%b %d, %Y", "%B %d, %Y",
                "%d %b %Y", "%d %B %Y", "%b %d %Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            pass
    return None


def is_relevant(conf: Dict) -> bool:
    text = " ".join([
        conf.get("title", ""), conf.get("full_name", ""),
        conf.get("tags", ""), conf.get("sub", ""),
    ]).lower()

    # Standalone "rl" as a whole word (catches "RL from World Feedback", etc.)
    if re.search(r'\brl\b', text):
        return True

    # OpenReview entries are already scoped to major ML conferences — require
    # keyword match so we don't include every NeurIPS/ICML workshop.
    if conf.get("source") == "OpenReview":
        return any(kw in text for kw in KEYWORDS)

    # For WikiCFP / aideadlines: flagship venue alone is enough (main tracks)
    if any(v in text for v in FLAGSHIP_VENUES):
        return True
    return any(kw in text for kw in KEYWORDS)


def in_window(dl: date) -> bool:
    today = date.today()
    return (today - timedelta(days=LOOKBACK_DAYS)) <= dl <= (today + timedelta(days=LOOKAHEAD_DAYS))


def build_deadline_list(ai_entries: List[Dict], wiki_entries: List[Dict],
                        or_entries: List[Dict] = None) -> List[Dict]:
    seen: set = set()
    output: List[Dict] = []

    def add(conf: Dict) -> None:
        key = conf.get("title", "").lower().replace(" ", "")
        if key in seen:
            return
        seen.add(key)
        if not is_relevant(conf):
            return
        raw = conf.get("deadline") or conf.get("abstract_deadline") or ""
        dl = parse_deadline(raw)
        if dl and in_window(dl):
            conf["parsed_deadline"] = dl
            output.append(conf)

    for e in ai_entries:
        add({
            "title":     e.get("title", ""),
            "full_name": e.get("full_name") or e.get("title", ""),
            "deadline":  e.get("deadline", ""),
            "abstract_deadline": e.get("abstract_deadline", ""),
            "date":      e.get("date", ""),
            "location":  e.get("place", ""),
            "url":       e.get("link") or e.get("website", ""),
            "source":    "aideadlines",
            "tags":      e.get("tags", "") + " " + e.get("sub", ""),
        })

    for e in wiki_entries:
        add(e)

    for e in (or_entries or []):
        add(e)

    output.sort(key=lambda x: x["parsed_deadline"])
    return output


# ── Email formatting ──────────────────────────────────────────────────────────

def urgency_badge(dl: date) -> str:
    delta = (dl - date.today()).days
    if delta < 0:
        return f"<span style='color:#e53e3e'>⚠ {abs(delta)}d ago</span>"
    if delta == 0:
        return "<span style='color:#e53e3e;font-weight:bold'>TODAY</span>"
    if delta <= 7:
        return f"<span style='color:#dd6b20'>in {delta}d</span>"
    if delta <= 14:
        return f"<span style='color:#d69e2e'>in {delta}d</span>"
    return f"<span style='color:#38a169'>in {delta}d</span>"


def build_html(deadlines: List[Dict]) -> str:
    today_str = date.today().strftime("%B %d, %Y")
    rows = ""
    for d in deadlines:
        dl: date = d["parsed_deadline"]
        name  = d.get("title", "Unknown")
        full  = d.get("full_name", "") or name
        url   = d.get("url", "#")
        sub   = f"<div style='font-size:12px;color:#718096'>{full}</div>" if full.lower() != name.lower() else ""
        src   = d.get("source", "")
        src_colors = {"OpenReview": "#6b46c1", "WikiCFP": "#2b6cb0", "aideadlines": "#276749"}
        src_badge = (f"<span style='font-size:10px;background:{src_colors.get(src,'#718096')};"
                     f"color:white;padding:1px 5px;border-radius:3px;margin-left:4px'>{src}</span>")
        rows += f"""
        <tr style='border-bottom:1px solid #e2e8f0'>
          <td style='padding:12px 8px;vertical-align:top'>
            <a href='{url}' style='font-weight:600;color:#2b6cb0;text-decoration:none'>{name}</a>{src_badge}{sub}
          </td>
          <td style='padding:12px 8px;white-space:nowrap'>{dl.strftime('%b %d, %Y')}<br>{urgency_badge(dl)}</td>
          <td style='padding:12px 8px;color:#4a5568'>{d.get('date','TBD')}</td>
          <td style='padding:12px 8px;color:#4a5568'>{d.get('location','')}</td>
        </tr>"""

    if not rows:
        rows = "<tr><td colspan='4' style='padding:20px;text-align:center;color:#718096'>No deadlines in the next 60 days.</td></tr>"

    return f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'></head>
<body style='font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;max-width:820px;margin:0 auto;padding:20px;color:#2d3748'>
  <div style='background:linear-gradient(135deg,#667eea,#764ba2);padding:24px;border-radius:8px;margin-bottom:24px'>
    <h1 style='margin:0;color:white;font-size:22px'>RL &amp; Robot Learning — Conference Deadlines</h1>
    <p style='margin:8px 0 0;color:rgba(255,255,255,.85);font-size:14px'>Daily digest &middot; {today_str}</p>
  </div>
  <table style='width:100%;border-collapse:collapse;font-size:14px'>
    <thead>
      <tr style='background:#f7fafc;text-align:left'>
        <th style='padding:10px 8px;color:#4a5568;border-bottom:2px solid #e2e8f0'>Conference</th>
        <th style='padding:10px 8px;color:#4a5568;border-bottom:2px solid #e2e8f0'>Deadline</th>
        <th style='padding:10px 8px;color:#4a5568;border-bottom:2px solid #e2e8f0'>Event Date</th>
        <th style='padding:10px 8px;color:#4a5568;border-bottom:2px solid #e2e8f0'>Location</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
  <p style='margin-top:24px;font-size:12px;color:#a0aec0'>
    Sources: <a href='https://aideadlin.es' style='color:#a0aec0'>aideadlin.es</a> &middot;
    <a href='http://www.wikicfp.com' style='color:#a0aec0'>WikiCFP</a> &middot;
    <a href='https://openreview.net' style='color:#a0aec0'>OpenReview</a> &middot;
    Showing deadlines within {LOOKAHEAD_DAYS} days.
  </p>
</body></html>"""


def build_plain(deadlines: List[Dict]) -> str:
    today_str = date.today().strftime("%B %d, %Y")
    lines = [f"RL & Robot Learning Conference Deadlines — {today_str}", "=" * 60, ""]
    for d in deadlines:
        dl: date = d["parsed_deadline"]
        delta = (dl - date.today()).days
        urgency = f"({abs(delta)}d ago)" if delta < 0 else f"(in {delta}d)"
        lines += [
            f"  {d['title']}  [{d.get('source','')}]",
            f"    Deadline : {dl.strftime('%b %d, %Y')} {urgency}",
            f"    Event    : {d.get('date','TBD')}  {d.get('location','')}",
            f"    Link     : {d.get('url','')}",
            "",
        ]
    if not deadlines:
        lines.append("  No deadlines found in the next 60 days.")
    return "\n".join(lines)


# ── Email ─────────────────────────────────────────────────────────────────────

def send_email(sender: str, app_password: str, recipient: str,
               html: str, plain: str, count: int) -> None:
    subject = (
        f"[Conference Bot] {count} RL/Robot Learning deadline"
        f"{'s' if count != 1 else ''} coming up"
    )
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = sender
    msg["To"]      = recipient
    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, app_password)
        server.sendmail(sender, recipient, msg.as_string())
    log.info("Email sent to %s", recipient)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    sender    = os.environ.get("GMAIL_ADDRESS", "").strip()
    password  = os.environ.get("GMAIL_APP_PASSWORD", "").strip()
    recipient = os.environ.get("RECIPIENT_EMAIL", sender).strip()

    if not sender or not password:
        raise SystemExit(
            "Missing credentials. Set GMAIL_ADDRESS and GMAIL_APP_PASSWORD "
            "as environment variables (or GitHub Secrets)."
        )

    log.info("Fetching from aideadlines...")
    ai_entries = fetch_aideadlines()
    log.info("  %d entries", len(ai_entries))

    log.info("Fetching from WikiCFP...")
    wiki_entries = fetch_wikicfp()
    log.info("  %d entries", len(wiki_entries))

    log.info("Fetching from OpenReview...")
    or_entries = fetch_openreview()
    log.info("  %d entries", len(or_entries))

    deadlines = build_deadline_list(ai_entries, wiki_entries, or_entries)
    log.info("Filtered to %d relevant deadlines in window", len(deadlines))

    html  = build_html(deadlines)
    plain = build_plain(deadlines)
    send_email(sender, password, recipient, html, plain, len(deadlines))


if __name__ == "__main__":
    main()
