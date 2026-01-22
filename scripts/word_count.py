#!/usr/bin/env python3
"""
Compute essay metadata: word count, reading time, extract tags.

Usage:
    python scripts/word_count.py docs/essays/001-introduction.html
    python scripts/word_count.py docs/essays/*.html --json
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    from html.parser import HTMLParser


if not HAS_BS4:
    class TextExtractor(HTMLParser):
        """Extract text content from HTML, ignoring tags and scripts."""

        def __init__(self):
            super().__init__()
            self.text_parts = []
            self.skip_tags = {'script', 'style', 'head'}
            self.skip_depth = 0

        def handle_starttag(self, tag, attrs):
            if tag.lower() in self.skip_tags or self.skip_depth > 0:
                self.skip_depth += 1

        def handle_endtag(self, tag):
            if self.skip_depth > 0:
                self.skip_depth -= 1

        def handle_data(self, data):
            if self.skip_depth == 0:
                self.text_parts.append(data)

        def get_text(self):
            return ' '.join(self.text_parts)


def extract_text_from_html(html_content: str) -> str:
    """Extract plain text from HTML content."""
    if HAS_BS4:
        soup = BeautifulSoup(html_content, 'html.parser')
        # Remove script, style, and head tags
        for tag in soup(['script', 'style', 'head']):
            tag.decompose()
        return soup.get_text(separator=' ')
    else:
        parser = TextExtractor()
        parser.feed(html_content)
        return parser.get_text()


def count_words(text: str) -> int:
    """Count words in text, handling various whitespace and punctuation."""
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Split on whitespace and filter empty strings
    words = [w for w in text.split() if w and not w.isspace()]
    # Filter out standalone punctuation
    words = [w for w in words if re.search(r'[a-zA-Z0-9]', w)]
    return len(words)


def calculate_reading_time(word_count: int, words_per_minute: int = 200) -> int:
    """Calculate reading time in minutes at given reading speed."""
    return max(1, round(word_count / words_per_minute))


def extract_tags_from_html(html_content: str) -> list[str]:
    """Extract tags from HTML meta tags or data attributes."""
    tags = []

    # Look for meta tags with name="keywords" or name="tags"
    meta_pattern = r'<meta\s+[^>]*name=["\'](?:keywords|tags)["\']\s+[^>]*content=["\']([^"\']+)["\']'
    matches = re.findall(meta_pattern, html_content, re.IGNORECASE)
    for match in matches:
        tags.extend([t.strip() for t in match.split(',')])

    # Also look for data-tags attribute
    data_tags_pattern = r'data-tags=["\']([^"\']+)["\']'
    matches = re.findall(data_tags_pattern, html_content, re.IGNORECASE)
    for match in matches:
        tags.extend([t.strip() for t in match.split(',')])

    # Remove duplicates while preserving order
    seen = set()
    unique_tags = []
    for tag in tags:
        if tag.lower() not in seen:
            seen.add(tag.lower())
            unique_tags.append(tag)

    return unique_tags


def extract_title_from_html(html_content: str) -> str:
    """Extract title from HTML."""
    # Try <title> tag first
    title_match = re.search(r'<title[^>]*>([^<]+)</title>', html_content, re.IGNORECASE)
    if title_match:
        return title_match.group(1).strip()

    # Try <h1> tag
    h1_match = re.search(r'<h1[^>]*>([^<]+)</h1>', html_content, re.IGNORECASE)
    if h1_match:
        return h1_match.group(1).strip()

    return "Untitled"


def extract_date_from_html(html_content: str) -> str:
    """Extract publication date from HTML meta tags."""
    # Look for meta date tags
    date_pattern = r'<meta\s+[^>]*name=["\'](?:date|publication-date|published)["\']\s+[^>]*content=["\']([^"\']+)["\']'
    match = re.search(date_pattern, html_content, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Look for time element with datetime attribute
    time_pattern = r'<time[^>]*datetime=["\']([^"\']+)["\']'
    match = re.search(time_pattern, html_content, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Return today's date as fallback
    return datetime.now().strftime('%Y-%m-%d')


def analyze_essay(filepath: Path) -> dict:
    """Analyze an essay HTML file and return metadata."""
    html_content = filepath.read_text(encoding='utf-8')
    text = extract_text_from_html(html_content)
    word_count = count_words(text)

    return {
        'file': str(filepath),
        'title': extract_title_from_html(html_content),
        'word_count': word_count,
        'reading_time_minutes': calculate_reading_time(word_count),
        'tags': extract_tags_from_html(html_content),
        'date': extract_date_from_html(html_content),
        'meets_minimum': word_count >= 12000
    }


def format_reading_time(minutes: int) -> str:
    """Format reading time for display."""
    if minutes < 60:
        return f"{minutes} min read"
    hours = minutes // 60
    remaining_minutes = minutes % 60
    if remaining_minutes == 0:
        return f"{hours} hr read"
    return f"{hours} hr {remaining_minutes} min read"


def main():
    parser = argparse.ArgumentParser(
        description='Compute essay metadata: word count, reading time, tags.'
    )
    parser.add_argument(
        'files',
        nargs='+',
        type=Path,
        help='HTML essay files to analyze'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )
    parser.add_argument(
        '--min-words',
        type=int,
        default=12000,
        help='Minimum word count requirement (default: 12000)'
    )

    args = parser.parse_args()

    results = []
    all_pass = True

    for filepath in args.files:
        if not filepath.exists():
            print(f"Error: File not found: {filepath}", file=sys.stderr)
            continue

        metadata = analyze_essay(filepath)
        metadata['meets_minimum'] = metadata['word_count'] >= args.min_words
        results.append(metadata)

        if not metadata['meets_minimum']:
            all_pass = False

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for meta in results:
            print(f"\n{'='*60}")
            print(f"File: {meta['file']}")
            print(f"Title: {meta['title']}")
            print(f"Word Count: {meta['word_count']:,}")
            print(f"Reading Time: {format_reading_time(meta['reading_time_minutes'])}")
            print(f"Date: {meta['date']}")
            print(f"Tags: {', '.join(meta['tags']) if meta['tags'] else 'None'}")

            if meta['meets_minimum']:
                print(f"Status: PASS (>= {args.min_words:,} words)")
            else:
                shortfall = args.min_words - meta['word_count']
                print(f"Status: FAIL ({shortfall:,} words short of {args.min_words:,})")
            print(f"{'='*60}")

    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
