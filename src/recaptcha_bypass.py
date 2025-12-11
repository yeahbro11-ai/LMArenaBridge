"""
Enterprise reCAPTCHA v3 Bypass Module (HTTP-based)

This module implements Enterprise reCAPTCHA v3 token extraction for LM Arena
authentication using direct HTTP requests. The legacy browser automation
approach has been replaced with a pure HTTP flow that mirrors what the browser
normally does:

1. Load the reCAPTCHA anchor iframe
2. Parse the challenge payload and extract the challenge token (`c`)
3. Call the enterprise reload endpoint to mint an `X-RECAPTCHA-TOKEN`
4. Return the token so it can be forwarded to LM Arena endpoints

Features:
- Enterprise reCAPTCHA v3 token extraction via official HTTP endpoints
- Proper header spoofing (Origin, Referer, User-Agent, Sec-Fetch-*)
- Automatic normalization of anchor URLs (domain, `co` parameter, etc.)
- Retry logic with exponential backoff and token caching
- Graceful fallback to anchor-embedded tokens when reload responses are empty

Usage:
    bypass = RecaptchaBypass()
    token = await bypass.extract_token(anchor_url)
    # Use token in X-RECAPTCHA-TOKEN header
"""

import asyncio
import base64
import json
import re
import time
from typing import Optional, Dict, Any, List
from urllib.parse import parse_qs, urlencode, urlparse, unquote_plus

import httpx

# Import functions from main module (will be available when used as part of the application)
try:
    from main import debug_print
except ImportError:
    # Fallback debug function for standalone testing
    def debug_print(*args, **kwargs):
        print("DEBUG:", *args, **kwargs)


ANCHOR_PAYLOAD_REGEX = re.compile(r'recaptcha\\.anchor\\.Main\\.init\("(.+?)"\);', re.S)
TOKEN_PATTERN = re.compile(r'^[A-Za-z0-9_-]{120,}$')
CHALLENGE_PATTERN = re.compile(r'^RC-[A-Za-z0-9_-]{5,}$')
BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


class RecaptchaBypass:
    """Enterprise reCAPTCHA v3 bypass using direct HTTP calls."""

    def __init__(
        self,
        cache_ttl: int = 120,
        request_timeout: float = 30.0,
        retry_backoff_base: float = 1.5,
        max_backoff: float = 6.0,
    ):
        """Initialize the reCAPTCHA bypass."""

        self.cache_ttl = cache_ttl
        self.request_timeout = request_timeout
        self.retry_backoff_base = retry_backoff_base
        self.max_backoff = max_backoff
        self._token_cache: Dict[str, tuple[str, float]] = {}

    # ---------------------------------------------------------------------
    # Cache helpers
    # ---------------------------------------------------------------------
    def _is_cache_valid(self, key: str) -> bool:
        if key not in self._token_cache:
            return False
        token, timestamp = self._token_cache[key]
        return time.time() - timestamp < self.cache_ttl

    def _get_cached_token(self, key: str) -> Optional[str]:
        if self._is_cache_valid(key):
            return self._token_cache[key][0]
        if key in self._token_cache:
            del self._token_cache[key]
        return None

    def _cache_token(self, key: str, token: str) -> None:
        self._token_cache[key] = (token, time.time())
        debug_print(f"💾 Cached reCAPTCHA token (key={key}): {token[:20]}...")

    # ---------------------------------------------------------------------
    # Anchor parsing helpers
    # ---------------------------------------------------------------------
    def _extract_site_key(self, anchor_url: str) -> Optional[str]:
        try:
            parsed = urlparse(anchor_url)
            params = parse_qs(parsed.query)
            site_key = params.get('k', [None])[0]
            if site_key:
                debug_print(f"🔑 Extracted site key: {site_key[:20]}...")
            return site_key
        except Exception as exc:
            debug_print(f"❌ Error extracting site key: {exc}")
            return None

    def _validate_anchor_url(self, anchor_url: str) -> bool:
        try:
            parsed = urlparse(anchor_url)
            if not parsed.netloc.endswith('google.com'):
                debug_print(f"❌ Invalid domain: {parsed.netloc}")
                return False
            if not parsed.path.startswith('/recaptcha/enterprise/anchor'):
                debug_print(f"❌ Invalid path: {parsed.path}")
                return False
            params = parse_qs(parsed.query)
            for param in ('k', 'co', 'v'):
                if param not in params:
                    debug_print(f"❌ Missing required parameter: {param}")
                    return False
            size = params.get('size', [''])[0]
            if size != 'invisible':
                debug_print(f"⚠️  Expected invisible captcha, got size: {size}")
            debug_print("✅ Anchor URL validation passed")
            return True
        except Exception as exc:
            debug_print(f"❌ Error validating anchor URL: {exc}")
            return False

    def _normalize_co_value(self, origin: str) -> str:
        encoded = base64.urlsafe_b64encode(origin.encode('utf-8')).decode('utf-8')
        return encoded.replace('=', '.')

    def _decode_co_value(self, co_value: str) -> str:
        if not co_value:
            raise ValueError("Missing co parameter")
        normalized = co_value.replace('.', '=')
        padding = len(normalized) % 4
        if padding:
            normalized += '=' * (4 - padding)
        decoded = base64.urlsafe_b64decode(normalized.encode('utf-8')).decode('utf-8')
        if not decoded.startswith('http'):
            decoded = f"https://{decoded.lstrip('/')}"
        return decoded

    def _prepare_anchor_request(
        self, anchor_url: str
    ) -> tuple[str, Dict[str, str], str, str]:
        parsed = urlparse(anchor_url)
        if not parsed.query:
            raise ValueError("Anchor URL missing query string")
        params = {
            key: values[-1]
            for key, values in parse_qs(parsed.query, keep_blank_values=True).items()
        }
        site_key = params.get('k')
        raw_co = params.get('co')
        if not site_key or not raw_co:
            raise ValueError("Anchor URL missing site key or co parameter")

        origin = self._decode_co_value(raw_co)
        normalized_co = self._normalize_co_value(origin)
        params['co'] = normalized_co

        origin_parsed = urlparse(origin)
        origin_host = origin_parsed.hostname or origin
        if 'domain' not in params and origin_host:
            params['domain'] = origin_host
        if 'hl' not in params:
            params['hl'] = 'en'
        if 'size' not in params:
            params['size'] = 'invisible'

        normalized_query = urlencode(params)
        normalized_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{normalized_query}"
        return normalized_url, params, origin, site_key

    def _build_anchor_headers(self, origin: str) -> Dict[str, str]:
        parsed = urlparse(origin)
        referer_base = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else origin
        referer = referer_base if referer_base.endswith('/') else f"{referer_base}/"
        return {
            'User-Agent': BROWSER_UA,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
            'Referer': referer,
            'Origin': referer_base,
            'Sec-Fetch-Dest': 'iframe',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'cross-site',
        }

    def _build_reload_headers(self, anchor_url: str) -> Dict[str, str]:
        parsed = urlparse(anchor_url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        return {
            'User-Agent': BROWSER_UA,
            'Accept': '*/*',
            'Content-Type': 'application/x-www-form-urlencoded',
            'Origin': origin,
            'Referer': anchor_url,
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'X-Requested-With': 'XMLHttpRequest',
        }

    def _parse_anchor_payload(self, html_text: str) -> List[Any]:
        match = ANCHOR_PAYLOAD_REGEX.search(html_text)
        if not match:
            if 'ErrorMain' in html_text:
                raise ValueError('Anchor returned ErrorMain payload')
            raise ValueError('Unable to locate anchor payload')
        encoded_payload = match.group(1)
        decoded_payload = bytes(encoded_payload, 'utf-8').decode('unicode_escape')
        return json.loads(decoded_payload)

    def _looks_like_token(self, candidate: str) -> bool:
        if not candidate or len(candidate) < 120:
            return False
        return bool(TOKEN_PATTERN.match(candidate))

    def _extract_prefetched_token(self, anchor_data: List[Any]) -> Optional[str]:
        for item in anchor_data:
            if isinstance(item, str) and self._looks_like_token(item):
                return item
        return None

    def _extract_challenge_token(self, anchor_data: List[Any]) -> Optional[str]:
        for item in anchor_data:
            if isinstance(item, str) and CHALLENGE_PATTERN.match(item):
                return item
        return None

    def _build_reload_payload(self, params: Dict[str, str], c_token: str) -> Dict[str, str]:
        payload = {
            'v': params.get('v', ''),
            'reason': params.get('reason', 'q'),
            'k': params.get('k', ''),
            'c': c_token,
            'sa': unquote_plus(params['sa']) if params.get('sa') else '',
            'co': params.get('co', ''),
            'hl': params.get('hl', 'en'),
            'size': params.get('size', 'invisible'),
        }
        passthrough_keys = (
            'stoken',
            'cb',
            'anchor',
            'anchor-s',
            'anchor-ms',
            'execute-ms',
        )
        for key in passthrough_keys:
            if key in params and key not in payload:
                payload[key] = params[key]
        return payload

    def _parse_reload_response(self, text: str) -> Optional[str]:
        sanitized = text.strip()
        if sanitized.startswith(")]}\'"):
            sanitized = sanitized.split('\n', 1)[-1]
        try:
            data = json.loads(sanitized)
        except json.JSONDecodeError:
            debug_print("❌ Failed to decode reload response JSON")
            return None
        if isinstance(data, list):
            # Expected format: ["rresp", "TOKEN", ...]
            if len(data) > 1 and isinstance(data[1], str) and data[1]:
                if self._looks_like_token(data[1]):
                    return data[1]
        # Fallback: search within string contents
        match = TOKEN_PATTERN.search(sanitized)
        if match:
            return match.group(0)
        return None

    async def _perform_http_token_exchange(self, anchor_url: str) -> Optional[str]:
        try:
            normalized_url, params, origin, site_key = self._prepare_anchor_request(anchor_url)
        except Exception as exc:
            debug_print(f"❌ Failed to normalize anchor URL: {exc}")
            return None

        anchor_headers = self._build_anchor_headers(origin)
        reload_headers = self._build_reload_headers(normalized_url)
        reload_url = f"https://www.google.com/recaptcha/enterprise/reload?k={site_key}"

        try:
            async with httpx.AsyncClient(timeout=self.request_timeout) as client:
                debug_print("🌐 Fetching reCAPTCHA anchor via HTTP...")
                anchor_response = await client.get(normalized_url, headers=anchor_headers)
                anchor_response.raise_for_status()
                anchor_data = self._parse_anchor_payload(anchor_response.text)

                c_token = self._extract_challenge_token(anchor_data)
                fallback_token = self._extract_prefetched_token(anchor_data)

                if not c_token and fallback_token:
                    debug_print("⚠️  Challenge token missing, using fallback token from anchor payload")
                    return fallback_token
                if not c_token:
                    debug_print("❌ Unable to locate challenge token in anchor payload")
                    return None

                reload_payload = self._build_reload_payload(params, c_token)
                debug_print("🔁 Calling enterprise reload endpoint to mint token")
                reload_response = await client.post(
                    reload_url,
                    data=reload_payload,
                    headers=reload_headers,
                )
                reload_response.raise_for_status()
                token = self._parse_reload_response(reload_response.text)
                if token:
                    debug_print(f"🎉 Successfully extracted token via HTTP: {token[:20]}...")
                    return token

                if fallback_token:
                    debug_print("⚠️  Reload response empty, falling back to anchor token")
                    return fallback_token

                debug_print("❌ Reload response did not contain a usable token")
                return None
        except httpx.HTTPStatusError as exc:
            debug_print(f"❌ HTTP status error during reCAPTCHA exchange: {exc}")
        except httpx.RequestError as exc:
            debug_print(f"❌ Network error during reCAPTCHA exchange: {exc}")
        except ValueError as exc:
            debug_print(f"❌ Payload parsing error: {exc}")
        return None

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    async def extract_token(self, anchor_url: str, max_retries: int = 3) -> Optional[str]:
        if not self._validate_anchor_url(anchor_url):
            debug_print("❌ Invalid anchor URL provided")
            return None

        site_key = self._extract_site_key(anchor_url)
        if not site_key:
            debug_print("❌ Could not extract site key from anchor URL")
            return None

        cache_key = f"{site_key}_{abs(hash(anchor_url)) % 10000}"
        cached = self._get_cached_token(cache_key)
        if cached:
            debug_print("✅ Using cached reCAPTCHA token")
            return cached

        for attempt in range(max_retries):
            token = await self._perform_http_token_exchange(anchor_url)
            if token:
                self._cache_token(cache_key, token)
                return token

            if attempt < max_retries - 1:
                delay = min(self.retry_backoff_base * (2 ** attempt), self.max_backoff)
                debug_print(f"⏳ Token extraction failed, retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)

        debug_print(f"❌ All {max_retries} reCAPTCHA extraction attempts failed")
        return None

    async def refresh_token(self, anchor_url: str) -> Optional[str]:
        site_key = self._extract_site_key(anchor_url)
        if site_key:
            cache_key = f"{site_key}_{abs(hash(anchor_url)) % 10000}"
            if cache_key in self._token_cache:
                del self._token_cache[cache_key]
                debug_print(f"🗑️  Cleared cached token for cache key: {cache_key}")
        return await self.extract_token(anchor_url, max_retries=2)

    async def cleanup(self):
        self._token_cache.clear()
        debug_print("🧹 RecaptchaBypass cleanup completed")


# Global instance for reuse across the application
_bypass_instance: Optional[RecaptchaBypass] = None


def get_recaptcha_bypass() -> RecaptchaBypass:
    global _bypass_instance
    if _bypass_instance is None:
        _bypass_instance = RecaptchaBypass()
    return _bypass_instance


async def extract_recaptcha_token(anchor_url: str) -> Optional[str]:
    bypass = get_recaptcha_bypass()
    return await bypass.extract_token(anchor_url)


async def cleanup_recaptcha_bypass():
    global _bypass_instance
    if _bypass_instance:
        await _bypass_instance.cleanup()
        _bypass_instance = None
