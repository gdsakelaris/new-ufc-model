#!/usr/bin/env python3
"""
Scrape UFC athlete profile pages for the "Trains at" field.

Target:
  https://www.ufc.com/athletes/all
  https://www.ufc.com/athletes/all?page=1
  https://www.ufc.com/athlete/<fighter-slug>

Output:
  CSV containing fighter_name, profile_url, trains_at

Install:
  pip install requests beautifulsoup4 lxml

Run:
  python scrape_ufc_trains_at.py --out ufc_trains_at.csv

Notes:
  - The script uses polite delays by default.
  - If UFC changes its markup, adjust parse_trains_at().
  - If you receive 403 responses, try running locally with a normal browser-like User-Agent,
    increasing --delay, or switching to a Playwright-based scraper.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass
from html import unescape
from typing import Iterable
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


@dataclass(frozen=True)
class TrainAtResult:
    fighter_name: str
    profile_url: str
    trains_at: str


def clean_text(value: str | None) -> str:
    """Normalize whitespace and HTML entities."""
    if not value:
        return ""
    return re.sub(r"\s+", " ", unescape(value)).strip()


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


def extract_athlete_links(soup: BeautifulSoup) -> list[str]:
    """
    Pull athlete profile URLs from a UFC listing page.

    UFC profile links are usually of the form:
      /athlete/amir-albazi
      https://www.ufc.com/athlete/amir-albazi
    """
    links: set[str] = set()

    for anchor in soup.select('a[href*="/athlete/"]'):
        href = anchor.get("href")
        if not href:
            continue

        normalized = normalize_url(href)
        parsed = urlparse(normalized)

        # Keep only UFC athlete profile pages, not the /athletes listing pages.
        if parsed.netloc.endswith("ufc.com") and parsed.path.startswith("/athlete/"):
            links.add(normalized)

    return sorted(links)


def collect_all_athlete_links(
    session: requests.Session,
    max_pages: int,
    delay: float,
    stop_after_empty_pages: int = 3,
) -> list[str]:
    """
    Walk the paginated /athletes/all pages until no new athlete links are found.

    Uses a "consecutive no-new-pages" stop condition because UFC may change the
    final page behavior over time.
    """
    seen: set[str] = set()
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

        page_links = extract_athlete_links(soup)
        new_links = [link for link in page_links if link not in seen]

        if new_links:
            seen.update(new_links)
            no_new_pages = 0
            print(
                f"[listing] Found {len(new_links)} new links "
                f"({len(seen)} total).",
                file=sys.stderr,
            )
        else:
            no_new_pages += 1
            print(
                f"[listing] No new links on this page "
                f"({no_new_pages}/{stop_after_empty_pages}).",
                file=sys.stderr,
            )
            if no_new_pages >= stop_after_empty_pages:
                break

        time.sleep(delay)

    return sorted(seen)


def parse_fighter_name(soup: BeautifulSoup, fallback_url: str) -> str:
    """
    Try a few likely UFC page selectors, then fall back to og:title/title,
    then finally the URL slug.
    """
    selectors = [
        "h1",
        ".hero-profile__name",
        ".c-hero__headline",
        ".c-profile-header__name",
    ]

    for selector in selectors:
        node = soup.select_one(selector)
        text = clean_text(node.get_text(" ", strip=True) if node else "")
        if text:
            return text

    og_title = soup.select_one('meta[property="og:title"]')
    if og_title and og_title.get("content"):
        text = clean_text(og_title["content"])
        if text:
            return text.replace(" | UFC", "").strip()

    if soup.title and soup.title.string:
        text = clean_text(soup.title.string)
        if text:
            return text.replace(" | UFC", "").strip()

    slug = urlparse(fallback_url).path.rstrip("/").split("/")[-1]
    return slug.replace("-", " ").title()


def find_next_bio_text(label_node: Tag) -> str:
    """
    Given a label node like:
      <div class="c-bio__label">Trains at</div>
      <div class="c-bio__text">Atreus MMA</div>

    Return the adjacent c-bio__text value.
    """
    next_text = label_node.find_next_sibling(
        lambda tag: isinstance(tag, Tag)
        and "c-bio__text" in tag.get("class", [])
    )
    if next_text:
        return clean_text(next_text.get_text(" ", strip=True))

    # Sometimes the text might be nested under the same parent wrapper.
    parent = label_node.parent
    if isinstance(parent, Tag):
        text_node = parent.select_one(".c-bio__text")
        if text_node:
            return clean_text(text_node.get_text(" ", strip=True))

    return ""


def parse_trains_at(soup: BeautifulSoup) -> str:
    """
    Extract the UFC bio value whose label is exactly "Trains at".

    Primary method:
      .c-bio__label text == "Trains at" -> nearest .c-bio__text

    Fallback:
      text-pattern search if the markup changes slightly.
    """
    for label_node in soup.select(".c-bio__label"):
        label = clean_text(label_node.get_text(" ", strip=True)).lower()
        label = re.sub(r"[^a-z ]+", "", label)

        if label == "trains at":
            trains_at = find_next_bio_text(label_node)
            if trains_at:
                return trains_at

    # Fallback for flattened text. This is intentionally conservative.
    page_text = clean_text(soup.get_text(" ", strip=True))
    match = re.search(
        r"\bTrains at\b\s+(.+?)(?:\s+(?:Fighting style|Age|Height|Weight|"
        r"Octagon Debut|Reach|Leg reach|Place of Birth)\b|$)",
        page_text,
        flags=re.IGNORECASE,
    )
    if match:
        return clean_text(match.group(1))

    return ""


def scrape_profile(
    session: requests.Session,
    profile_url: str,
) -> TrainAtResult | None:
    soup = get_soup(session, profile_url)
    fighter_name = parse_fighter_name(soup, profile_url)
    trains_at = parse_trains_at(soup)

    if not trains_at:
        return None

    return TrainAtResult(
        fighter_name=fighter_name,
        profile_url=profile_url,
        trains_at=trains_at,
    )


def write_csv(results: Iterable[TrainAtResult], out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["fighter_name", "profile_url", "trains_at"],
        )
        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "fighter_name": result.fighter_name,
                    "profile_url": result.profile_url,
                    "trains_at": result.trains_at,
                }
            )


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    return session


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Scrape UFC fighter profile pages for the "Trains at" field.'
    )
    parser.add_argument(
        "--out",
        default="ufc_trains_at.csv",
        help="Output CSV path. Default: ufc_trains_at.csv",
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
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of athlete profiles to scrape, useful for testing.",
    )
    args = parser.parse_args()

    session = build_session()

    athlete_links = collect_all_athlete_links(
        session=session,
        max_pages=args.max_pages,
        delay=args.delay,
    )

    if args.limit is not None:
        athlete_links = athlete_links[: args.limit]

    print(f"[profiles] Scraping {len(athlete_links)} athlete profiles.", file=sys.stderr)

    results: list[TrainAtResult] = []
    failures: list[tuple[str, str]] = []

    for index, profile_url in enumerate(athlete_links, start=1):
        print(f"[profiles] {index}/{len(athlete_links)} {profile_url}", file=sys.stderr)

        try:
            result = scrape_profile(session, profile_url)
        except Exception as exc:
            failures.append((profile_url, str(exc)))
            print(f"[profiles] Failed: {profile_url} -> {exc}", file=sys.stderr)
            time.sleep(args.delay)
            continue

        if result:
            results.append(result)
            print(
                f"[profiles] Found trains_at: {result.fighter_name} -> {result.trains_at}",
                file=sys.stderr,
            )

        time.sleep(args.delay)

    write_csv(results, args.out)

    print(f"\nDone.", file=sys.stderr)
    print(f"Profiles checked: {len(athlete_links)}", file=sys.stderr)
    print(f"Profiles with Trains at: {len(results)}", file=sys.stderr)
    print(f"Failures: {len(failures)}", file=sys.stderr)
    print(f"CSV written to: {args.out}", file=sys.stderr)

    if failures:
        failures_path = args.out.rsplit(".", 1)[0] + "_failures.csv"
        with open(failures_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["profile_url", "error"])
            writer.writeheader()
            for profile_url, error in failures:
                writer.writerow({"profile_url": profile_url, "error": error})
        print(f"Failures written to: {failures_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
