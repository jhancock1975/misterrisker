"""
Tests for the chat export button functionality.

The export button should:
1. Be present in the HTML template
2. Have proper CSS styling
3. Include the exportChatToClipboard JavaScript function
4. Use html2canvas library for capturing
5. Attempt clipboard copy first, fall back to download
"""

import pytest
from unittest.mock import patch
import re

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# =============================================================================
# Test: Export Button HTML Structure
# =============================================================================

class TestExportButtonHTML:
    """Tests for export button presence in HTML template."""
    
    def test_export_button_exists_in_template(self, log):
        """Should have an export button in the HTML template."""
        from web.app import HTML_TEMPLATE
        
        log.info("Testing export button exists in template")
        
        assert 'export-btn' in HTML_TEMPLATE
        assert 'exportChatToClipboard()' in HTML_TEMPLATE
    
    def test_export_button_has_onclick_handler(self, log):
        """Should have onclick handler calling exportChatToClipboard."""
        from web.app import HTML_TEMPLATE
        
        log.info("Testing export button onclick handler")
        
        # Check for the button with onclick
        assert 'onclick="exportChatToClipboard()"' in HTML_TEMPLATE
    
    def test_export_button_has_title_attribute(self, log):
        """Should have a title attribute for accessibility."""
        from web.app import HTML_TEMPLATE
        
        log.info("Testing export button title attribute")
        
        assert 'title="Copy conversation as image"' in HTML_TEMPLATE
    
    def test_export_button_has_svg_icon(self, log):
        """Should have an SVG icon inside the button."""
        from web.app import HTML_TEMPLATE
        
        log.info("Testing export button SVG icon")
        
        # The button should contain an SVG element
        # Find the export button section and check for SVG
        button_match = re.search(
            r'<button class="export-btn"[^>]*>.*?</button>',
            HTML_TEMPLATE,
            re.DOTALL
        )
        assert button_match is not None
        button_html = button_match.group(0)
        assert '<svg' in button_html
        assert '</svg>' in button_html
    
    def test_export_button_has_text_label(self, log):
        """Should have 'Export' text label."""
        from web.app import HTML_TEMPLATE
        
        log.info("Testing export button text label")
        
        # Find the button and check for Export text
        button_match = re.search(
            r'<button class="export-btn"[^>]*>.*?</button>',
            HTML_TEMPLATE,
            re.DOTALL
        )
        assert button_match is not None
        button_html = button_match.group(0)
        assert 'Export' in button_html


# =============================================================================
# Test: Export Button CSS Styling
# =============================================================================

class TestExportButtonCSS:
    """Tests for export button CSS styles."""
    
    def test_export_button_base_styles(self, log):
        """Should have base CSS styles for the export button."""
        from web.app import HTML_TEMPLATE
        
        log.info("Testing export button base CSS")
        
        assert '.export-btn {' in HTML_TEMPLATE
        assert 'cursor: pointer' in HTML_TEMPLATE
    
    def test_export_button_hover_styles(self, log):
        """Should have hover styles."""
        from web.app import HTML_TEMPLATE
        
        log.info("Testing export button hover CSS")
        
        assert '.export-btn:hover {' in HTML_TEMPLATE
    
    def test_export_button_copying_state_styles(self, log):
        """Should have styles for the copying/active state."""
        from web.app import HTML_TEMPLATE
        
        log.info("Testing export button copying state CSS")
        
        assert '.export-btn.copying {' in HTML_TEMPLATE


# =============================================================================
# Test: Export JavaScript Function
# =============================================================================

class TestExportJavaScriptFunction:
    """Tests for the exportChatToClipboard JavaScript function."""
    
    def test_export_function_defined(self, log):
        """Should define the exportChatToClipboard function."""
        from web.app import HTML_TEMPLATE
        
        log.info("Testing exportChatToClipboard function definition")
        
        assert 'async function exportChatToClipboard()' in HTML_TEMPLATE
    
    def test_uses_html2canvas_library(self, log):
        """Should use html2canvas library for screenshot capture."""
        from web.app import HTML_TEMPLATE
        
        log.info("Testing html2canvas usage")
        
        # Check library is included
        assert 'html2canvas' in HTML_TEMPLATE
        # Check it's called in the function
        assert 'await html2canvas(' in HTML_TEMPLATE
    
    def test_captures_chat_container(self, log):
        """Should capture the chatContainer element."""
        from web.app import HTML_TEMPLATE
        
        log.info("Testing chat container capture")
        
        assert 'html2canvas(chatContainer' in HTML_TEMPLATE
    
    def test_converts_to_png_blob(self, log):
        """Should convert canvas to PNG blob."""
        from web.app import HTML_TEMPLATE
        
        log.info("Testing PNG blob conversion")
        
        assert "canvas.toBlob(resolve, 'image/png')" in HTML_TEMPLATE
    
    def test_attempts_clipboard_api_first(self, log):
        """Should try ClipboardItem API for copying image."""
        from web.app import HTML_TEMPLATE
        
        log.info("Testing Clipboard API attempt")
        
        assert 'navigator.clipboard.write' in HTML_TEMPLATE
        assert 'ClipboardItem' in HTML_TEMPLATE
        assert "'image/png'" in HTML_TEMPLATE
    
    def test_falls_back_to_download(self, log):
        """Should fall back to downloading the file on clipboard error."""
        from web.app import HTML_TEMPLATE
        
        log.info("Testing download fallback")
        
        # Should create a download link
        assert "link.download = 'mister-risker-chat-'" in HTML_TEMPLATE
        assert '.png' in HTML_TEMPLATE
        assert 'link.click()' in HTML_TEMPLATE
    
    def test_shows_loading_state(self, log):
        """Should show loading state while capturing."""
        from web.app import HTML_TEMPLATE
        
        log.info("Testing loading state")
        
        assert '⏳ Capturing...' in HTML_TEMPLATE
    
    def test_shows_success_state(self, log):
        """Should show success state after copying."""
        from web.app import HTML_TEMPLATE
        
        log.info("Testing success state")
        
        assert '✓ Copied!' in HTML_TEMPLATE
    
    def test_shows_download_state(self, log):
        """Should show download state when falling back."""
        from web.app import HTML_TEMPLATE
        
        log.info("Testing download state")
        
        assert '⬇️ Downloaded!' in HTML_TEMPLATE
    
    def test_shows_error_state(self, log):
        """Should show error state on failure."""
        from web.app import HTML_TEMPLATE
        
        log.info("Testing error state")
        
        assert '❌ Failed' in HTML_TEMPLATE
    
    def test_restores_button_after_action(self, log):
        """Should restore original button HTML after action completes."""
        from web.app import HTML_TEMPLATE
        
        log.info("Testing button restoration")
        
        # Should save and restore original HTML
        assert 'originalHTML = exportBtn.innerHTML' in HTML_TEMPLATE
        assert 'exportBtn.innerHTML = originalHTML' in HTML_TEMPLATE
    
    def test_temporarily_expands_chat_container(self, log):
        """Should temporarily expand chat container to capture all content."""
        from web.app import HTML_TEMPLATE
        
        log.info("Testing container expansion for full capture")
        
        # Should set height to auto to show all content
        assert "chatContainer.style.height = 'auto'" in HTML_TEMPLATE
        assert "chatContainer.style.maxHeight = 'none'" in HTML_TEMPLATE
        assert "chatContainer.style.overflow = 'visible'" in HTML_TEMPLATE
    
    def test_restores_chat_container_dimensions(self, log):
        """Should restore chat container dimensions after capture."""
        from web.app import HTML_TEMPLATE
        
        log.info("Testing container dimension restoration")
        
        # Should store and restore original values
        assert 'originalHeight = chatContainer.style.height' in HTML_TEMPLATE
        assert 'originalOverflow = chatContainer.style.overflow' in HTML_TEMPLATE
        assert 'chatContainer.style.height = originalHeight' in HTML_TEMPLATE


# =============================================================================
# Test: html2canvas Library Inclusion
# =============================================================================

class TestHtml2CanvasLibrary:
    """Tests for html2canvas library inclusion."""
    
    def test_html2canvas_cdn_included(self, log):
        """Should include html2canvas from CDN."""
        from web.app import HTML_TEMPLATE
        
        log.info("Testing html2canvas CDN inclusion")
        
        assert 'html2canvas' in HTML_TEMPLATE
        # Should be a script tag with CDN URL
        assert '<script src="https://' in HTML_TEMPLATE or 'html2canvas' in HTML_TEMPLATE
    
    def test_html2canvas_options(self, log):
        """Should use appropriate html2canvas options."""
        from web.app import HTML_TEMPLATE
        
        log.info("Testing html2canvas options")
        
        # Should set background color
        assert "backgroundColor: '#f8f9fa'" in HTML_TEMPLATE
        # Should use high quality scale
        assert 'scale: 2' in HTML_TEMPLATE


# =============================================================================
# Test: Export Button Position in Header
# =============================================================================

class TestExportButtonPosition:
    """Tests for export button position in the header."""
    
    def test_export_button_in_header_right(self, log):
        """Should be positioned in header-right container."""
        from web.app import HTML_TEMPLATE
        
        log.info("Testing export button in header-right")
        
        # Find header-right div and check it contains export button
        header_right_match = re.search(
            r'<div class="header-right">.*?</div>',
            HTML_TEMPLATE,
            re.DOTALL
        )
        assert header_right_match is not None
        header_right_html = header_right_match.group(0)
        assert 'export-btn' in header_right_html
    
    def test_export_button_before_clear_button(self, log):
        """Should appear before the clear button in the header."""
        from web.app import HTML_TEMPLATE
        
        log.info("Testing export button order")
        
        # Find the actual button elements in the HTML (not CSS class definitions)
        export_button_pos = HTML_TEMPLATE.find('<button class="export-btn"')
        clear_button_pos = HTML_TEMPLATE.find('<button class="clear-btn"')
        
        # Export button should come before clear button
        assert export_button_pos > 0, "Export button not found"
        assert clear_button_pos > 0, "Clear button not found"
        assert export_button_pos < clear_button_pos
