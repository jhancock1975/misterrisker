"""
Tests for image generation functionality in the chat application.

The chatbot should be able to:
1. Detect when a user is requesting image/visual content generation
2. Generate SVG images, JavaScript animations, or HTML-based visuals
3. Return the generated content in a format the frontend can render
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client for image generation tests."""
    client = MagicMock()
    return client


# =============================================================================
# Test: Image Generation Request Detection
# =============================================================================

class TestImageGenerationDetection:
    """Tests for detecting image generation requests."""
    
    def test_detects_draw_request(self, log):
        """Should detect 'draw' as an image generation request."""
        from web.app import TradingChatBot
        
        log.info("Testing draw request detection")
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch('web.app.CoinbaseMCPServer'):
                with patch('web.app.SchwabMCPServer'):
                    bot = TradingChatBot(use_agents=False)
        
        assert bot._is_image_generation_request("draw a picture of a duck")
        assert bot._is_image_generation_request("Draw me a chart")
        assert bot._is_image_generation_request("Can you draw something?")
        
        log.info("RESULT: Draw requests detected correctly")
    
    def test_detects_create_image_request(self, log):
        """Should detect 'create an image' as an image generation request."""
        from web.app import TradingChatBot
        
        log.info("Testing create image request detection")
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch('web.app.CoinbaseMCPServer'):
                with patch('web.app.SchwabMCPServer'):
                    bot = TradingChatBot(use_agents=False)
        
        assert bot._is_image_generation_request("create an image of a stock chart")
        assert bot._is_image_generation_request("Create a picture of Bitcoin")
        
        log.info("RESULT: Create image requests detected correctly")
    
    def test_detects_generate_visual_request(self, log):
        """Should detect requests for visualizations and animations."""
        from web.app import TradingChatBot
        
        log.info("Testing visual/animation request detection")
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch('web.app.CoinbaseMCPServer'):
                with patch('web.app.SchwabMCPServer'):
                    bot = TradingChatBot(use_agents=False)
        
        assert bot._is_image_generation_request("generate an SVG of a logo")
        assert bot._is_image_generation_request("make an animation of a bouncing ball")
        assert bot._is_image_generation_request("create a visualization of my portfolio")
        assert bot._is_image_generation_request("show me a graphic of Bitcoin price")
        
        log.info("RESULT: Visual/animation requests detected correctly")
    
    def test_does_not_detect_normal_requests(self, log):
        """Should not detect normal trading requests as image generation."""
        from web.app import TradingChatBot
        
        log.info("Testing that normal requests are not detected as image generation")
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch('web.app.CoinbaseMCPServer'):
                with patch('web.app.SchwabMCPServer'):
                    bot = TradingChatBot(use_agents=False)
        
        assert not bot._is_image_generation_request("what is my balance")
        assert not bot._is_image_generation_request("buy 100 shares of AAPL")
        assert not bot._is_image_generation_request("tell me the latest news")
        assert not bot._is_image_generation_request("describe the market conditions")
        
        log.info("RESULT: Normal requests not detected as image generation")


# =============================================================================
# Test: SVG Generation
# =============================================================================

class TestSVGGeneration:
    """Tests for SVG image generation."""
    
    @pytest.mark.asyncio
    async def test_generates_svg_response(self, log):
        """Should generate SVG content for draw requests."""
        from web.app import TradingChatBot
        
        log.info("Testing SVG generation")
        
        mock_svg = '<svg width="200" height="200"><circle cx="100" cy="100" r="50" fill="yellow"/></svg>'
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch('web.app.CoinbaseMCPServer'):
                with patch('web.app.SchwabMCPServer'):
                    bot = TradingChatBot(use_agents=False)
        
        # Mock the LLM to return SVG content
        mock_response = MagicMock()
        mock_response.content = f"Here's a yellow circle:\n\n```svg\n{mock_svg}\n```"
        bot.llm = MagicMock()
        bot.llm.ainvoke = AsyncMock(return_value=mock_response)
        
        response = await bot.process_message("draw a yellow circle")
        
        # Response should contain SVG marker for frontend to detect
        assert "```svg" in response or "<svg" in response
        
        log.info("RESULT: SVG content generated successfully")
    
    @pytest.mark.asyncio
    async def test_response_format_includes_image_type(self, log):
        """Response should include type info when containing generated image."""
        from web.app import TradingChatBot
        
        log.info("Testing response format for images")
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch('web.app.CoinbaseMCPServer'):
                with patch('web.app.SchwabMCPServer'):
                    bot = TradingChatBot(use_agents=False)
        
        mock_response = MagicMock()
        mock_response.content = '<svg width="100" height="100"><rect fill="blue"/></svg>'
        bot.llm = MagicMock()
        bot.llm.ainvoke = AsyncMock(return_value=mock_response)
        
        response = await bot.generate_image("draw a blue square")
        
        # Should return dict with type and content
        assert isinstance(response, dict)
        assert "type" in response
        assert "content" in response
        assert response["type"] in ["svg", "html", "animation"]
        
        log.info("RESULT: Response format correct for generated images")


# =============================================================================
# Test: JavaScript Animation Generation
# =============================================================================

class TestAnimationGeneration:
    """Tests for JavaScript animation generation."""
    
    @pytest.mark.asyncio
    async def test_generates_animation_response(self, log):
        """Should generate JavaScript animation for animation requests."""
        from web.app import TradingChatBot
        
        log.info("Testing animation generation")
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch('web.app.CoinbaseMCPServer'):
                with patch('web.app.SchwabMCPServer'):
                    bot = TradingChatBot(use_agents=False)
        
        mock_animation = """
        <div id="animation-container">
            <canvas id="canvas" width="300" height="200"></canvas>
            <script>
                const canvas = document.getElementById('canvas');
                const ctx = canvas.getContext('2d');
                let x = 0;
                function animate() {
                    ctx.clearRect(0, 0, 300, 200);
                    ctx.beginPath();
                    ctx.arc(x, 100, 20, 0, Math.PI * 2);
                    ctx.fill();
                    x = (x + 2) % 300;
                    requestAnimationFrame(animate);
                }
                animate();
            </script>
        </div>
        """
        
        mock_response = MagicMock()
        mock_response.content = mock_animation
        bot.llm = MagicMock()
        bot.llm.ainvoke = AsyncMock(return_value=mock_response)
        
        response = await bot.generate_image("make an animation of a bouncing ball")
        
        assert isinstance(response, dict)
        assert response["type"] == "animation"
        assert "<script>" in response["content"] or "canvas" in response["content"]
        
        log.info("RESULT: Animation content generated successfully")


# =============================================================================
# Test: API Endpoint for Image Generation
# =============================================================================

class TestImageGenerationEndpoint:
    """Tests for the /chat endpoint handling image generation."""
    
    @pytest.mark.asyncio
    async def test_chat_endpoint_returns_image_response(self, log):
        """Chat endpoint should return image content in response."""
        from web.app import TradingChatBot
        
        log.info("Testing chat endpoint with image generation request")
        
        # Create a bot with mocked LLM that returns SVG
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch('web.app.CoinbaseMCPServer'):
                with patch('web.app.SchwabMCPServer'):
                    bot = TradingChatBot(use_agents=False)
        
        # Mock the LLM to return SVG content
        mock_response = MagicMock()
        mock_response.content = '<svg width="100" height="100"><circle cx="50" cy="50" r="40" fill="red"/></svg>'
        bot.llm = MagicMock()
        bot.llm.ainvoke = AsyncMock(return_value=mock_response)
        
        # Test through process_message
        response = await bot.process_message("draw a red circle")
        
        # Response should include the SVG content
        assert "svg" in response.lower() or "<svg" in response
        assert "circle" in response.lower()
        
        log.info("RESULT: Chat endpoint handles image generation correctly")


# =============================================================================
# Test: Frontend Rendering Detection
# =============================================================================

class TestFrontendRenderingSupport:
    """Tests for ensuring responses are frontend-renderable."""
    
    def test_svg_wrapped_for_frontend(self, log):
        """SVG content should be wrapped in a way frontend can detect and render."""
        from web.app import TradingChatBot
        
        log.info("Testing SVG wrapping for frontend")
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch('web.app.CoinbaseMCPServer'):
                with patch('web.app.SchwabMCPServer'):
                    bot = TradingChatBot(use_agents=False)
        
        raw_svg = '<svg width="100" height="100"><circle/></svg>'
        wrapped = bot._wrap_generated_content("svg", raw_svg, "A circle")
        
        # Should contain markers that frontend can detect
        assert "<!--GENERATED_IMAGE:svg-->" in wrapped or "```svg" in wrapped
        assert raw_svg in wrapped
        
        log.info("RESULT: SVG wrapped correctly for frontend")
    
    def test_animation_wrapped_for_frontend(self, log):
        """Animation content should be wrapped for frontend iframe rendering."""
        from web.app import TradingChatBot
        
        log.info("Testing animation wrapping for frontend")
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch('web.app.CoinbaseMCPServer'):
                with patch('web.app.SchwabMCPServer'):
                    bot = TradingChatBot(use_agents=False)
        
        raw_animation = '<canvas id="c"></canvas><script>/*animation*/</script>'
        wrapped = bot._wrap_generated_content("animation", raw_animation, "An animation")
        
        # Should contain markers for frontend
        assert "<!--GENERATED_IMAGE:animation-->" in wrapped or "```html" in wrapped
        
        log.info("RESULT: Animation wrapped correctly for frontend")


# =============================================================================
# Test: Error Handling
# =============================================================================

class TestImageGenerationErrors:
    """Tests for error handling in image generation."""
    
    @pytest.mark.asyncio
    async def test_handles_generation_failure(self, log):
        """Should handle LLM failures gracefully."""
        from web.app import TradingChatBot
        
        log.info("Testing generation failure handling")
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch('web.app.CoinbaseMCPServer'):
                with patch('web.app.SchwabMCPServer'):
                    bot = TradingChatBot(use_agents=False)
        
        bot.llm = MagicMock()
        bot.llm.ainvoke = AsyncMock(side_effect=Exception("API Error"))
        
        response = await bot.generate_image("draw something")
        
        assert isinstance(response, dict)
        assert "error" in response or response.get("type") == "error"
        
        log.info("RESULT: Generation failure handled gracefully")
    
    @pytest.mark.asyncio
    async def test_handles_invalid_svg_output(self, log):
        """Should handle when LLM doesn't return valid SVG."""
        from web.app import TradingChatBot
        
        log.info("Testing invalid SVG handling")
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch('web.app.CoinbaseMCPServer'):
                with patch('web.app.SchwabMCPServer'):
                    bot = TradingChatBot(use_agents=False)
        
        mock_response = MagicMock()
        mock_response.content = "I'm sorry, I can't draw images directly."
        bot.llm = MagicMock()
        bot.llm.ainvoke = AsyncMock(return_value=mock_response)
        
        response = await bot.generate_image("draw a cat")
        
        # Should return the text response with type indicating no image
        assert isinstance(response, dict)
        assert response.get("type") in ["text", "error"] or "content" in response
        
        log.info("RESULT: Invalid SVG output handled gracefully")
