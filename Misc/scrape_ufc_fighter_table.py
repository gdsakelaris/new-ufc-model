#!/usr/bin/env python3
"""
Build the master fighter/gym table from the UFC athlete listing.

Scrapes every athlete card from:
  https://www.ufc.com/athletes/all
  https://www.ufc.com/athletes/all?page=1
  ...

and writes a CSV with one row per fighter:
  fighter_name, nickname, weight_class, record, profile_url  (filled)
  gym_name, gym_elevation_ft                                 (empty, for later scripts)

The listing cards already display name, nickname, division and W-L-D record,
so NO individual profile pages are fetched — one pass over the paginated
listing (~290 pages at ~11 cards each) covers all ~3,149 athletes.

Disambiguation: `profile_url` is the strongest key (UFC slugs are unique even
for identical names, e.g. /athlete/bruno-silva vs /athlete/bruno-silva-0);
`record` + `weight_class` + `nickname` are kept as human-friendly helpers for
fuzzy matching against other sites later.

Re-running MERGES into an existing output file: gym_name / gym_elevation_ft
values already filled in (by hand or by later scripts) are preserved, keyed by
profile_url, and fighters no longer on the listing are kept rather than lost.

Install:
  pip install requests beautifulsoup4 lxml

Run:
  python scrape_ufc_fighter_table.py --out ufc_fighter_gym_table.csv

Notes:
  - The script uses polite delays by default.
  - If UFC changes its markup, adjust extract_athlete_cards().
  - If you receive 403 responses, try increasing --delay or a Playwright-based
    scraper.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from dataclasses import dataclass
from html import unescape
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup, Tag


BASE_URL = "https://www.ufc.com"
ATHLETES_ALL_URL = f"{BASE_URL}/athletes/all"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

CSV_COLUMNS = [
    "fighter_name",
    "nickname",
    "weight_class",
    "record",
    "profile_url",
    "gym_name",
    "gym_elevation_ft",
]


@dataclass(frozen=True)
class AthleteRow:
    fighter_name: str
    nickname: str
    weight_class: str
    record: str
    profile_url: str


def clean_text(value: str | None) -> str:
    """Normalize whitespace and HTML entities."""
    if not value:
        return ""
    return re.sub(r"\s+", " ", unescape(value)).strip()


def clean_record(value: str) -> str:
    """'24-5-0 (W-L-D)' -> '24-5-0'."""
    return clean_text(re.sub(r"\(\s*W\s*-\s*L\s*-\s*D\s*\)", "", value, flags=re.IGNORECASE))


def clean_nickname(value: str) -> str:
    """Strip the decorative quotes UFC wraps nicknames in."""
    return clean_text(value).strip('"“”‘’ ')


def normalize_url(url: str) -> str:
    """
    Normalize a UFC athlete URL:
      - absolute URL
      - no query string
      - no fragment
      - no trailing slash
    """
    absolute = urljoin(BASE_URL, url)
    parsed = urlparse(absolute)
    parsed = parsed._replace(query="", fragment="")
    normalized = urlunparse(parsed).rstrip("/")
    return normalized


def get_soup(session: requests.Session, url: str, timeout: int = 30) -> BeautifulSoup:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return BeautifulSoup(response.text, "lxml")


def athlete_listing_url(page_index: int) -> str:
    if page_index == 0:
        return ATHLETES_ALL_URL
    return f"{ATHLETES_ALL_URL}?page={page_index}"


def select_text(node: Tag, selector: str) -> str:
    found = node.select_one(selector)
    return clean_text(found.get_text(" ", strip=True)) if found else ""


def _valid_profile_url(href: str) -> str:
    profile_url = normalize_url(href)
    parsed = urlparse(profile_url)
    if parsed.netloc.endswith("ufc.com") and parsed.path.startswith("/athlete/"):
        return profile_url
    return ""


def _slug_name(profile_url: str) -> str:
    slug = urlparse(profile_url).path.rstrip("/").split("/")[-1]
    return slug.replace("-", " ").title()


def extract_athlete_cards(soup: BeautifulSoup) -> list[AthleteRow]:
    """
    Parse athlete listing cards from a /athletes/all page.

    Each card is a flipcard whose FRONT face carries the bio fields and whose
    BACK face carries the profile link, so fields must be read from the whole
    flipcard wrapper (the name appears on both faces; record/division only on
    the front):
      .c-listing-athlete__name      Fighter name
      .c-listing-athlete__nickname  "Nickname"
      .c-listing-athlete__title     Division (e.g. "Lightweight")
      .c-listing-athlete__record    "24-5-0 (W-L-D)"
    The /athlete/<slug> link is the unique key.
    """
    rows: list[AthleteRow] = []
    seen: set[str] = set()

    cards = soup.select(".c-listing-athlete-flipcard__inner")
    if not cards:
        cards = soup.select(".c-listing-athlete-flipcard")

    for card in cards:
        anchor = card.select_one('a[href*="/athlete/"]')
        if anchor is None or not anchor.get("href"):
            continue
        profile_url = _valid_profile_url(anchor["href"])
        if not profile_url or profile_url in seen:
            continue

        name = select_text(card, ".c-listing-athlete__name")
        if not name:
            name = _slug_name(profile_url)

        seen.add(profile_url)
        rows.append(AthleteRow(
            fighter_name=name,
            nickname=clean_nickname(select_text(card, ".c-listing-athlete__nickname")),
            weight_class=select_text(card, ".c-listing-athlete__title"),
            record=clean_record(select_text(card, ".c-listing-athlete__record")),
            profile_url=profile_url,
        ))

    # Safety net for markup changes: any profile link not inside a recognized
    # card still yields a slug-named row (bio fields empty) so no fighter is
    # silently dropped.
    for anchor in soup.select('a[href*="/athlete/"]'):
        href = anchor.get("href")
        if not href:
            continue
        profile_url = _valid_profile_url(href)
        if not profile_url or profile_url in seen:
            continue
        seen.add(profile_url)
        rows.append(AthleteRow(_slug_name(profile_url), "", "", "", profile_url))

    return rows


def collect_all_athletes(
    session: requests.Session,
    max_pages: int,
    delay: float,
    stop_after_empty_pages: int = 3,
) -> dict[str, AthleteRow]:
    """
    Walk the paginated /athletes/all pages until no new athletes are found.

    Uses a "consecutive no-new-pages" stop condition because UFC may change the
    final page behavior over time.
    """
    athletes: dict[str, AthleteRow] = {}
    no_new_pages = 0

    for page_index in range(max_pages):
        url = athlete_listing_url(page_index)
        print(f"[listing] Fetching page {page_index}: {url}", file=sys.stderr)

        try:
            soup = get_soup(session, url)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            print(f"[listing] HTTP {status} at {url}; stopping listing crawl.", file=sys.stderr)
            break

        page_rows = extract_athlete_cards(soup)
        new_rows = [row for row in page_rows if row.profile_url not in athletes]

        # Upgrade sparse safety-net rows (e.g. from a promo link on an earlier
        # page) when the fighter's real card shows up with bio fields filled.
        for row in page_rows:
            prior = athletes.get(row.profile_url)
            if prior is not None and not prior.record and row.record:
                athletes[row.profile_url] = row

        if new_rows:
            for row in new_rows:
                athletes[row.profile_url] = row
            no_new_pages = 0
            print(
                f"[listing] Found {len(new_rows)} new athletes "
                f"({len(athletes)} total).",
                file=sys.stderr,
            )
        else:
            no_new_pages += 1
            print(
                f"[listing] No new athletes on this page "
                f"({no_new_pages}/{stop_after_empty_pages}).",
                file=sys.stderr,
            )
            if no_new_pages >= stop_after_empty_pages:
                break

        time.sleep(delay)

    return athletes


def load_existing(out_path: str) -> dict[str, dict[str, str]]:
    """Existing output rows keyed by profile_url (for merge-on-rerun)."""
    if not os.path.exists(out_path):
        return {}
    with open(out_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return {
            row["profile_url"]: row
            for row in reader
            if row.get("profile_url")
        }


def write_csv(rows: list[dict[str, str]], out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    return session


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scrape all UFC fighters into a fighter/gym table "
        "(gym columns left empty for later scripts)."
    )
    parser.add_argument(
        "--out",
        default="ufc_fighter_gym_table.csv",
        help="Output CSV path. Default: ufc_fighter_gym_table.csv",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=500,
        help="Maximum listing pages to crawl. Default: 500",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.75,
        help="Delay between requests in seconds. Default: 0.75",
    )
    parser.add_argument(
        "--expected-total",
        type=int,
        default=3149,
        help="Expected athlete count for a sanity note at the end (0 disables). "
        "Default: 3149 (ufc.com/athletes/all as of 2026-07).",
    )
    args = parser.parse_args()

    session = build_session()
    athletes = collect_all_athletes(
        session=session,
        max_pages=args.max_pages,
        delay=args.delay,
    )

    existing = load_existing(args.out)

    # Merge: scraped fields refresh every run; gym_name / gym_elevation_ft are
    # never clobbered once filled. Fighters that vanished from the listing are
    # kept so previously filled gym data is not lost.
    merged: list[dict[str, str]] = []
    preserved_gym_values = 0
    for row in athletes.values():
        out_row = {
            "fighter_name": row.fighter_name,
            "nickname": row.nickname,
            "weight_class": row.weight_class,
            "record": row.record,
            "profile_url": row.profile_url,
            "gym_name": "",
            "gym_elevation_ft": "",
        }
        prior = existing.get(row.profile_url)
        if prior:
            out_row["gym_name"] = prior.get("gym_name", "")
            out_row["gym_elevation_ft"] = prior.get("gym_elevation_ft", "")
            if out_row["gym_name"] or out_row["gym_elevation_ft"]:
                preserved_gym_values += 1
        merged.append(out_row)

    stale = [
        prior for url, prior in existing.items() if url not in athletes
    ]
    merged.extend(stale)

    merged.sort(key=lambda r: (r.get("fighter_name", "").lower(), r.get("profile_url", "")))
    write_csv(merged, args.out)

    with_record = sum(1 for r in merged if r.get("record"))
    print("\nDone.", file=sys.stderr)
    print(f"Athletes scraped from listing: {len(athletes)}", file=sys.stderr)
    if stale:
        print(f"Kept from previous file (no longer listed): {len(stale)}", file=sys.stderr)
    if preserved_gym_values:
        print(f"Rows with preserved gym data: {preserved_gym_values}", file=sys.stderr)
    print(f"Rows with a record: {with_record}/{len(merged)}", file=sys.stderr)
    if args.expected_total:
        delta = len(athletes) - args.expected_total
        print(
            f"Expected ~{args.expected_total} athletes; scraped {len(athletes)} "
            f"({delta:+d}).",
            file=sys.stderr,
        )
    print(f"CSV written to: {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
