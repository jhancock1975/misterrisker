#!/usr/bin/env python3
"""Download Berkshire Hathaway quarterly reports and import into ChromaDB.

This script:
1. Downloads quarterly 10-Q reports from berkshirehathaway.com
2. Extracts text from PDFs using pdfplumber
3. Chunks and indexes them into ChromaDB

Reports available: 1996-2025 (quarterly reports)
"""

import os
import sys
import logging
import requests
from pathlib import Path
from typing import Optional
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Base URL for Berkshire Hathaway reports
BASE_URL = "https://www.berkshirehathaway.com"

# Output directory for downloaded PDFs
REPORTS_DIR = Path(__file__).parent.parent / "resources" / "berkshire_reports"


def get_quarterly_report_urls() -> list[dict]:
    """Generate list of quarterly report URLs based on known patterns."""
    reports = []
    
    # Quarterly reports (1st, 2nd, 3rd quarter - 4th quarter is in annual report)
    quarters = ["1stqtr", "2ndqtr", "3rdqtr"]
    
    # Years with quarterly reports (1996-2025)
    # Note: Earlier years may not have all quarters available
    for year in range(2025, 1995, -1):  # 2025 down to 1996
        year_suffix = str(year)[2:]  # "25" for 2025
        
        for qtr in quarters:
            # Skip future quarters
            if year == 2025 and qtr == "3rdqtr":
                # As of Dec 2025, Q3 2025 should be available
                pass
            
            pdf_name = f"{qtr}{year_suffix}.pdf"
            pdf_url = f"{BASE_URL}/qtrly/{pdf_name}"
            
            reports.append({
                "year": year,
                "quarter": qtr.replace("qtr", ""),  # "1st", "2nd", "3rd"
                "url": pdf_url,
                "filename": pdf_name
            })
    
    return reports


def download_report(url: str, output_path: Path) -> bool:
    """Download a single report PDF.
    
    Args:
        url: URL to download from
        output_path: Where to save the file
        
    Returns:
        True if successful, False otherwise
    """
    if output_path.exists():
        logger.info(f"  Already exists: {output_path.name}")
        return True
    
    try:
        logger.info(f"  Downloading: {url}")
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            # Check if it's actually a PDF
            if response.content[:4] == b'%PDF':
                output_path.write_bytes(response.content)
                logger.info(f"  Saved: {output_path.name} ({len(response.content):,} bytes)")
                return True
            else:
                logger.warning(f"  Not a PDF: {url}")
                return False
        elif response.status_code == 404:
            logger.debug(f"  Not found (404): {url}")
            return False
        else:
            logger.warning(f"  HTTP {response.status_code}: {url}")
            return False
            
    except Exception as e:
        logger.error(f"  Error downloading {url}: {e}")
        return False


def download_all_reports() -> list[Path]:
    """Download all available quarterly reports.
    
    Returns:
        List of paths to downloaded PDFs
    """
    # Create output directory
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    reports = get_quarterly_report_urls()
    logger.info(f"Checking {len(reports)} potential quarterly reports...")
    
    downloaded = []
    
    for report in reports:
        output_path = REPORTS_DIR / report["filename"]
        
        if download_report(report["url"], output_path):
            downloaded.append(output_path)
        
        # Be nice to the server
        time.sleep(0.5)
    
    logger.info(f"\nDownloaded {len(downloaded)} reports to {REPORTS_DIR}")
    return downloaded


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF file.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text content
    """
    try:
        import pdfplumber
    except ImportError:
        logger.error("pdfplumber not installed. Run: uv add pdfplumber")
        return ""
    
    text_parts = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text()
                    if text:
                        text_parts.append(f"\n--- Page {page_num + 1} ---\n{text}")
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num + 1} from {pdf_path.name}: {e}")
                    
    except Exception as e:
        logger.error(f"Error opening PDF {pdf_path}: {e}")
    
    return "\n".join(text_parts)


def create_markdown_from_pdfs(pdf_dir: Path, output_file: Path) -> int:
    """Convert all PDFs to a single markdown file for ChromaDB indexing.
    
    Args:
        pdf_dir: Directory containing PDFs
        output_file: Output markdown file path
        
    Returns:
        Number of reports processed
    """
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_dir}")
        return 0
    
    logger.info(f"Extracting text from {len(pdf_files)} PDFs...")
    
    content_parts = [
        "# Berkshire Hathaway Quarterly Reports\n\n",
        "This document contains quarterly 10-Q reports from Berkshire Hathaway Inc.\n",
        "These reports provide valuable insights into Warren Buffett's investment philosophy,\n",
        "company operations, insurance business, and market commentary.\n\n",
        "---\n\n"
    ]
    
    processed = 0
    
    for pdf_path in pdf_files:
        logger.info(f"Processing: {pdf_path.name}")
        
        # Parse filename to get quarter/year
        # Format: 1stqtr25.pdf, 2ndqtr24.pdf, etc.
        name = pdf_path.stem
        quarter = name[:3]  # "1st", "2nd", "3rd"
        year_suffix = name.replace("qtr", "")[3:]  # "25", "24", etc.
        
        try:
            year = int("20" + year_suffix) if int(year_suffix) < 96 else int("19" + year_suffix)
        except:
            year = year_suffix
        
        # Extract text
        text = extract_text_from_pdf(pdf_path)
        
        if text:
            content_parts.append(f"## Berkshire Hathaway {quarter.upper()} Quarter {year} Report\n\n")
            content_parts.append(f"Source: {pdf_path.name}\n\n")
            content_parts.append(text)
            content_parts.append("\n\n---\n\n")
            processed += 1
        else:
            logger.warning(f"  No text extracted from {pdf_path.name}")
    
    # Write combined markdown
    output_file.write_text("\n".join(content_parts))
    logger.info(f"\nCreated {output_file} with {processed} reports")
    
    return processed


def index_to_chromadb() -> int:
    """Re-index all resources including Berkshire reports into ChromaDB.
    
    Returns:
        Number of chunks indexed
    """
    from src.services.rag_service import RAGService
    
    logger.info("Re-indexing resources into ChromaDB...")
    
    # Force reindex to include new content
    rag = RAGService(resources_dir="resources", auto_index=False)
    total_chunks = rag.index_resources(force_reindex=True)
    
    logger.info(f"Indexed {total_chunks} total chunks")
    
    # Show stats
    stats = rag.get_stats()
    logger.info(f"ChromaDB stats: {stats}")
    
    return total_chunks


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Berkshire Hathaway reports")
    parser.add_argument("--download", action="store_true", help="Download PDFs")
    parser.add_argument("--extract", action="store_true", help="Extract text from PDFs")
    parser.add_argument("--index", action="store_true", help="Index into ChromaDB")
    parser.add_argument("--all", action="store_true", help="Do all steps")
    
    args = parser.parse_args()
    
    if args.all or (not args.download and not args.extract and not args.index):
        args.download = args.extract = args.index = True
    
    if args.download:
        logger.info("=" * 60)
        logger.info("STEP 1: Downloading Berkshire Hathaway Reports")
        logger.info("=" * 60)
        downloaded = download_all_reports()
        logger.info(f"Downloaded {len(downloaded)} reports")
    
    if args.extract:
        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 2: Extracting Text from PDFs")
        logger.info("=" * 60)
        
        output_md = Path(__file__).parent.parent / "resources" / "berkshire-quarterly-reports.md"
        processed = create_markdown_from_pdfs(REPORTS_DIR, output_md)
        logger.info(f"Processed {processed} reports into {output_md}")
    
    if args.index:
        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 3: Indexing into ChromaDB")
        logger.info("=" * 60)
        chunks = index_to_chromadb()
        logger.info(f"Indexed {chunks} chunks")
    
    logger.info("")
    logger.info("Done!")


if __name__ == "__main__":
    main()
