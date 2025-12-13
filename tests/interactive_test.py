"""Interactive Playwright test for manual testing with Mister Risker.

This script opens a visible browser and pauses for manual interaction.
Run with: python tests/interactive_test.py
"""

import asyncio
from playwright.async_api import async_playwright


async def main():
    """Open browser and wait for manual interaction."""
    print("🚀 Starting interactive Playwright session...")
    print("📍 Opening Mister Risker at http://localhost:8000")
    print("⏸️  Browser will stay open for manual testing")
    print("   Press Ctrl+C in terminal to close when done\n")
    
    async with async_playwright() as p:
        # Launch browser in headed mode (visible)
        browser = await p.chromium.launch(
            headless=False,
            slow_mo=100,  # Slow down actions so we can see them
        )
        
        # Create a new context and page
        context = await browser.new_context(
            viewport={"width": 1400, "height": 900}
        )
        page = await context.new_page()
        
        # Navigate to Mister Risker
        await page.goto("http://localhost:8000")
        
        # Wait for the page to load
        await page.wait_for_load_state("networkidle")
        
        print("✅ Browser is open and ready!")
        print("💬 You can now interact with Mister Risker")
        print("📝 I can see screenshots if you describe what's happening\n")
        
        # Use page.pause() for interactive debugging
        # This opens Playwright Inspector and waits
        await page.pause()
        
        # Cleanup
        await browser.close()
        print("\n👋 Browser closed. Session ended.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Session interrupted. Goodbye!")
