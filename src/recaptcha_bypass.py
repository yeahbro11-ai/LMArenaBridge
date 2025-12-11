"""
Enterprise reCAPTCHA v3 Bypass Module

This module implements Enterprise reCAPTCHA v3 token extraction for LM Arena authentication.
It handles the invisible captcha challenge by processing the anchor URL and extracting
the X-RECAPTCHA-TOKEN token required for successful authentication.

Features:
- Enterprise reCAPTCHA v3 token extraction from anchor URLs
- Headless browser context for handling third-party cookie/storage access
- Retry logic and proper error handling
- Token caching to avoid redundant extractions
- Integration with existing AsyncCamoufox setup

Usage:
    bypass = RecaptchaBypass()
    token = await bypass.extract_token(anchor_url)
    # Use token in X-RECAPTCHA-TOKEN header
"""

import asyncio
import re
import json
import time
from typing import Optional, Dict, Any
from urllib.parse import urlparse, parse_qs

import httpx
from camoufox.async_api import AsyncCamoufox

# Import functions from main module (will be available when used as part of the application)
try:
    from main import debug_print, get_config
except ImportError:
    # Fallback debug function for standalone testing
    def debug_print(*args, **kwargs):
        print("DEBUG:", *args, **kwargs)
    
    def get_config():
        return {"auth_tokens": [], "cf_clearance": ""}


class RecaptchaBypass:
    """
    Enterprise reCAPTCHA v3 bypass using headless browser automation.
    
    Handles the extraction of X-RECAPTCHA-TOKEN tokens from invisible captcha challenges
    by navigating to the anchor URL in a browser context and extracting the token
    from the challenge response.
    """
    
    def __init__(self, cache_ttl: int = 120):
        """
        Initialize the reCAPTCHA bypass.
        
        Args:
            cache_ttl: Token cache time-to-live in seconds (default: 2 minutes)
        """
        self.cache_ttl = cache_ttl
        self._token_cache: Dict[str, tuple[str, float]] = {}
        self._browser = None
        
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached token is still valid"""
        if key not in self._token_cache:
            return False
        
        timestamp, _ = self._token_cache[key]
        return time.time() - timestamp < self.cache_ttl
    
    def _get_cached_token(self, key: str) -> Optional[str]:
        """Get cached token if valid"""
        if self._is_cache_valid(key):
            return self._token_cache[key][1]
        return None
    
    def _cache_token(self, key: str, token: str):
        """Cache token with timestamp"""
        self._token_cache[key] = (time.time(), token)
    
    def _extract_site_key(self, anchor_url: str) -> Optional[str]:
        """
        Extract site key (k parameter) from anchor URL.
        
        Args:
            anchor_url: The reCAPTCHA enterprise anchor URL
            
        Returns:
            Site key if found, None otherwise
        """
        try:
            parsed = urlparse(anchor_url)
            params = parse_qs(parsed.query)
            site_key = params.get('k', [None])[0]
            
            if site_key:
                debug_print(f"🔑 Extracted site key: {site_key[:20]}...")
                return site_key
            
            debug_print("❌ No site key found in anchor URL")
            return None
            
        except Exception as e:
            debug_print(f"❌ Error extracting site key: {e}")
            return None
    
    def _validate_anchor_url(self, anchor_url: str) -> bool:
        """
        Validate that the URL is a proper reCAPTCHA enterprise anchor URL.
        
        Args:
            anchor_url: URL to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            parsed = urlparse(anchor_url)
            
            # Check domain
            if not parsed.netloc.endswith('google.com'):
                debug_print(f"❌ Invalid domain: {parsed.netloc}")
                return False
            
            # Check path
            if not parsed.path.startswith('/recaptcha/enterprise/anchor'):
                debug_print(f"❌ Invalid path: {parsed.path}")
                return False
            
            # Check required parameters
            params = parse_qs(parsed.query)
            required_params = ['k', 'co', 'v']
            
            for param in required_params:
                if param not in params:
                    debug_print(f"❌ Missing required parameter: {param}")
                    return False
            
            # Check that it's invisible
            size = params.get('size', [''])[0]
            if size != 'invisible':
                debug_print(f"⚠️  Expected invisible captcha, got size: {size}")
            
            debug_print(f"✅ Anchor URL validation passed")
            return True
            
        except Exception as e:
            debug_print(f"❌ Error validating anchor URL: {e}")
            return False
    
    async def _get_browser(self) -> AsyncCamoufox:
        """Get or create AsyncCamoufox browser instance"""
        if self._browser is None or not self._browser.is_connected():
            debug_print("🆕 Creating new AsyncCamoufox browser instance...")
            self._browser = AsyncCamoufox(headless=True)
        return self._browser
    
    async def _close_browser(self):
        """Close browser instance if open"""
        if self._browser and self._browser.is_connected():
            try:
                await self._browser.close()
                debug_print("🔒 Closed AsyncCamoufox browser instance")
            except Exception as e:
                debug_print(f"⚠️  Error closing browser: {e}")
            finally:
                self._browser = None
    
    async def _extract_token_from_browser(self, anchor_url: str) -> Optional[str]:
        """
        Extract X-RECAPTCHA-TOKEN using browser automation.
        
        This method navigates to the anchor URL, waits for the invisible captcha
        challenge to complete, and extracts the token from the response.
        
        Args:
            anchor_url: reCAPTCHA enterprise anchor URL
            
        Returns:
            X-RECAPTCHA-TOKEN if successful, None otherwise
        """
        browser = None
        try:
            debug_print(f"🌐 Starting browser-based token extraction...")
            debug_print(f"📍 Target URL: {anchor_url[:100]}...")
            
            # Get browser instance
            browser = await self._get_browser()
            
            # Create new page
            page = await browser.new_page()
            
            # Set up response interceptor to capture token responses
            token_response = None
            
            async def capture_token_response(response):
                nonlocal token_response
                try:
                    url = response.url
                    if 'recaptcha/enterprise' in url and response.status == 200:
                        # Look for token in response text
                        text = await response.text()
                        
                        # Try to find token in various patterns
                        patterns = [
                            r'"response":"([A-Za-z0-9_-]{100,})"',
                            r'"recaptcha_token":"([A-Za-z0-9_-]{100,})"',
                            r'token["\']\s*:\s*["\']([A-Za-z0-9_-]{100,})["\']',
                            r'"token":\s*"([A-Za-z0-9_-]{100,})"',
                            r'([A-Za-z0-9_-]{100,})'
                        ]
                        
                        for pattern in patterns:
                            match = re.search(pattern, text)
                            if match:
                                potential_token = match.group(1)
                                if len(potential_token) > 50:  # Likely a real token
                                    token_response = potential_token
                                    debug_print(f"🎯 Token captured from response: {potential_token[:20]}...")
                                    break
                        
                except Exception as e:
                    debug_print(f"⚠️  Error capturing response: {e}")
            
            # Register response interceptor
            page.on('response', capture_token_response)
            
            # Set viewport and user agent
            await page.set_viewport_size({"width": 1920, "height": 1080})
            await page.set_extra_http_headers({
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            })
            
            # Navigate to anchor URL
            debug_print("📤 Navigating to anchor URL...")
            await page.goto(anchor_url, wait_until="networkidle", timeout=30000)
            
            # Wait for captcha challenge to potentially complete
            debug_print("⏳ Waiting for captcha challenge...")
            await asyncio.sleep(5)
            
            # Try to find captcha iframe and trigger if needed
            try:
                # Look for invisible reCAPTCHA iframe
                iframe = await page.wait_for_selector('iframe[src*="recaptcha/enterprise"]', timeout=10000)
                if iframe:
                    debug_print("✅ Found reCAPTCHA iframe")
                    # Click on iframe to trigger challenge
                    await iframe.click()
                    await asyncio.sleep(3)
            except Exception as e:
                debug_print(f"⚠️  No iframe found or click failed: {e}")
            
            # Additional wait for challenge completion
            debug_print("⏳ Waiting for challenge completion...")
            await asyncio.sleep(5)
            
            # Try multiple extraction methods
            extraction_methods = [
                self._extract_from_page_content,
                self._extract_from_network_tab,
                self._extract_from_browser_storage
            ]
            
            for method in extraction_methods:
                try:
                    token = await method(page)
                    if token:
                        debug_print(f"🎉 Token extracted successfully using {method.__name__}")
                        return token
                except Exception as e:
                    debug_print(f"⚠️  {method.__name__} failed: {e}")
                    continue
            
            # If still no token, try to execute JavaScript to get it
            try:
                js_token = await page.evaluate("""
                    () => {
                        // Try to find token in various ways
                        const methods = [
                            () => window.___grecaptcha_cfg?.callback,
                            () => window.grecaptcha?.enterprise?.render?.token,
                            () => document.querySelector('textarea[name="g-recaptcha-response"]')?.value,
                            () => window.grecaptcha?.enterprise?.getResponse?.(),
                            () => {
                                // Look for token in script tags
                                const scripts = document.querySelectorAll('script');
                                for (let script of scripts) {
                                    if (script.textContent) {
                                        const match = script.textContent.match(/"response":"([A-Za-z0-9_-]{100,})"/);
                                        if (match) return match[1];
                                    }
                                }
                                return null;
                            }
                        ];
                        
                        for (let method of methods) {
                            try {
                                const result = method();
                                if (result && typeof result === 'string' && result.length > 50) {
                                    return result;
                                }
                            } catch (e) {
                                // Continue to next method
                            }
                        }
                        return null;
                    }
                """)
                
                if js_token:
                    debug_print(f"🎯 Token extracted via JavaScript: {js_token[:20]}...")
                    return js_token
                    
            except Exception as e:
                debug_print(f"⚠️  JavaScript extraction failed: {e}")
            
            debug_print("❌ All token extraction methods failed")
            return None
            
        except Exception as e:
            debug_print(f"❌ Browser extraction failed: {e}")
            return None
            
        finally:
            if page:
                try:
                    await page.close()
                except Exception:
                    pass
    
    async def _extract_from_page_content(self, page) -> Optional[str]:
        """Extract token from page content"""
        try:
            content = await page.content()
            
            # Look for token in various patterns
            patterns = [
                r'"response":"([A-Za-z0-9_-]{100,})"',
                r'"recaptcha_token":"([A-Za-z0-9_-]{100,})"',
                r'token["\']\s*:\s*["\']([A-Za-z0-9_-]{100,})["\']',
                r'"token":\s*"([A-Za-z0-9_-]{100,})"'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    if len(match) > 50:  # Likely a real token
                        return match
            
            return None
            
        except Exception as e:
            debug_print(f"⚠️  Page content extraction failed: {e}")
            return None
    
    async def _extract_from_network_tab(self, page) -> Optional[str]:
        """Extract token from network requests"""
        try:
            # Check if we can access network tab
            responses = await page.evaluate("""
                () => {
                    // This is a placeholder - actual network access would require
                    // setting up a response interceptor earlier
                    return null;
                }
            """)
            
            return responses
            
        except Exception as e:
            debug_print(f"⚠️  Network tab extraction failed: {e}")
            return None
    
    async def _extract_from_browser_storage(self, page) -> Optional[str]:
        """Extract token from browser storage (localStorage, sessionStorage, cookies)"""
        try:
            storage_token = await page.evaluate("""
                () => {
                    // Try localStorage
                    for (let key in localStorage) {
                        if (key.toLowerCase().includes('recaptcha') || key.toLowerCase().includes('captcha')) {
                            const value = localStorage.getItem(key);
                            if (value && value.length > 50) return value;
                        }
                    }
                    
                    // Try sessionStorage
                    for (let key in sessionStorage) {
                        if (key.toLowerCase().includes('recaptcha') || key.toLowerCase().includes('captcha')) {
                            const value = sessionStorage.getItem(key);
                            if (value && value.length > 50) return value;
                        }
                    }
                    
                    // Try cookies
                    const cookies = document.cookie.split(';');
                    for (let cookie of cookies) {
                        const [name, value] = cookie.trim().split('=');
                        if (name.toLowerCase().includes('recaptcha') || name.toLowerCase().includes('captcha')) {
                            if (value && value.length > 50) return value;
                        }
                    }
                    
                    return null;
                }
            """)
            
            if storage_token:
                debug_print(f"💾 Token found in browser storage")
                return storage_token
            
            return None
            
        except Exception as e:
            debug_print(f"⚠️  Browser storage extraction failed: {e}")
            return None
    
    async def extract_token(self, anchor_url: str, max_retries: int = 3) -> Optional[str]:
        """
        Extract Enterprise reCAPTCHA v3 token from anchor URL.
        
        This is the main entry point for token extraction. It handles:
        - URL validation
        - Token caching
        - Browser automation
        - Retry logic
        - Error handling
        
        Args:
            anchor_url: The reCAPTCHA enterprise anchor URL
            max_retries: Maximum number of retry attempts
            
        Returns:
            X-RECAPTCHA-TOKEN if successful, None otherwise
        """
        # Validate anchor URL
        if not self._validate_anchor_url(anchor_url):
            debug_print("❌ Invalid anchor URL provided")
            return None
        
        # Create cache key based on site key
        site_key = self._extract_site_key(anchor_url)
        if not site_key:
            debug_print("❌ Could not extract site key from anchor URL")
            return None
        
        cache_key = f"{site_key}_{hash(anchor_url) % 10000}"  # Hash for variety
        
        # Check cache first
        cached_token = self._get_cached_token(cache_key)
        if cached_token:
            debug_print(f"✅ Using cached reCAPTCHA token")
            return cached_token
        
        # Try extraction with retries
        for attempt in range(max_retries):
            try:
                debug_print(f"🔄 Attempting token extraction (attempt {attempt + 1}/{max_retries})")
                
                # Extract token using browser automation
                token = await self._extract_token_from_browser(anchor_url)
                
                if token:
                    # Cache successful token
                    self._cache_token(cache_key, token)
                    debug_print(f"🎉 Successfully extracted reCAPTCHA token: {token[:20]}...")
                    return token
                
                if attempt < max_retries - 1:
                    debug_print(f"⏳ Retrying in 2 seconds...")
                    await asyncio.sleep(2)
                    
            except Exception as e:
                debug_print(f"❌ Extraction attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
        
        debug_print(f"❌ All {max_retries} extraction attempts failed")
        return None
    
    async def refresh_token(self, anchor_url: str) -> Optional[str]:
        """
        Force refresh of reCAPTCHA token (bypass cache).
        
        Args:
            anchor_url: The reCAPTCHA enterprise anchor URL
            
        Returns:
            Fresh X-RECAPTCHA-TOKEN if successful, None otherwise
        """
        # Clear cache for this key
        site_key = self._extract_site_key(anchor_url)
        if site_key:
            cache_key = f"{site_key}_{hash(anchor_url) % 10000}"
            if cache_key in self._token_cache:
                del self._token_cache[cache_key]
                debug_print(f"🗑️  Cleared cached token for cache key: {cache_key}")
        
        # Extract fresh token
        return await self.extract_token(anchor_url, max_retries=2)
    
    async def get_anchor_url_from_page(self, page) -> Optional[str]:
        """
        Extract anchor URL from a loaded page.
        
        This method searches for reCAPTCHA anchor URLs in page content,
        which can be useful when the anchor URL is not directly available.
        
        Args:
            page: Playwright page object
            
        Returns:
            Anchor URL if found, None otherwise
        """
        try:
            # Look for anchor URLs in page content
            content = await page.content()
            
            # Pattern to find anchor URLs
            anchor_pattern = r'https://www\.google\.com/recaptcha/enterprise/anchor[^\s"\'<>]*'
            matches = re.findall(anchor_pattern, content)
            
            if matches:
                anchor_url = matches[0]
                debug_print(f"🔗 Found anchor URL in page: {anchor_url[:100]}...")
                return anchor_url
            
            debug_print("❌ No anchor URL found in page")
            return None
            
        except Exception as e:
            debug_print(f"❌ Error extracting anchor URL from page: {e}")
            return None
    
    async def cleanup(self):
        """Cleanup resources"""
        await self._close_browser()
        self._token_cache.clear()
        debug_print("🧹 RecaptchaBypass cleanup completed")


# Global instance for reuse across the application
_bypass_instance: Optional[RecaptchaBypass] = None


def get_recaptcha_bypass() -> RecaptchaBypass:
    """Get global RecaptchaBypass instance"""
    global _bypass_instance
    if _bypass_instance is None:
        _bypass_instance = RecaptchaBypass()
    return _bypass_instance


async def extract_recaptcha_token(anchor_url: str) -> Optional[str]:
    """
    Convenience function to extract reCAPTCHA token.
    
    Args:
        anchor_url: The reCAPTCHA enterprise anchor URL
        
    Returns:
        X-RECAPTCHA-TOKEN if successful, None otherwise
    """
    bypass = get_recaptcha_bypass()
    return await bypass.extract_token(anchor_url)


async def cleanup_recaptcha_bypass():
    """Cleanup global bypass instance"""
    global _bypass_instance
    if _bypass_instance:
        await _bypass_instance.cleanup()
        _bypass_instance = None