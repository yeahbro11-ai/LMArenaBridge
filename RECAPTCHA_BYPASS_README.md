# Enterprise reCAPTCHA v3 Bypass Implementation

## Overview

This implementation provides a complete Enterprise reCAPTCHA v3 bypass solution for LM Arena authentication. The solution resolves "reCAPTCHA validation failed" errors by automatically extracting and managing reCAPTCHA tokens.

## Components

### 1. `src/recaptcha_bypass.py`

The main reCAPTCHA bypass module that implements:

#### Key Features:
- **Enterprise reCAPTCHA v3 Token Extraction**: Extracts tokens from anchor URLs using headless browser automation
- **Token Caching**: Caches tokens for 2 minutes to avoid redundant extractions
- **Retry Logic**: Handles extraction failures with configurable retry attempts
- **Browser Automation**: Uses AsyncCamoufox for robust token extraction
- **Error Handling**: Comprehensive error handling for various failure scenarios

#### Core Classes:
- `RecaptchaBypass`: Main class for token extraction and management
- Global functions for convenient access and integration

#### Usage Example:
```python
from recaptcha_bypass import extract_recaptcha_token

# Extract token from anchor URL
token = await extract_recaptcha_token(
    "https://www.google.com/recaptcha/enterprise/anchor?..."
)

# Use token in request headers
headers = {
    "x-recaptcha-token": token,
    "Content-Type": "text/plain;charset=UTF-8",
    "Cookie": "cf_clearance=...; arena-auth-prod-v1=..."
}
```

### 2. Integration in `src/main.py`

The main FastAPI application has been updated to include:

#### Updated Functions:
- **`get_request_headers_with_token()`**: Now supports reCAPTCHA tokens and streaming headers
- **`get_or_extract_recaptcha_token()`**: Retrieves or extracts cached tokens
- **`handle_recaptcha_error()`**: Detects and handles reCAPTCHA validation failures

#### New Global Variables:
- `recaptcha_token_cache`: Dictionary for token caching with timestamps
- `recaptcha_anchor_urls`: Store anchor URLs per domain
- `DEFAULT_LMARENA_ANCHOR_URL`: Default anchor URL for LM Arena

#### Enhanced Request Flow:
1. **Automatic reCAPTCHA Token Retrieval**: On 403 errors, automatically attempt to get fresh token
2. **Retry Logic**: Enhanced retry logic handles 429, 401, and 403 errors
3. **Streaming Support**: Full support for both streaming and non-streaming requests

#### New Dashboard Features:
- **reCAPTCHA Status Section**: Shows token availability and cache status
- **Manual Refresh**: Button to manually refresh reCAPTCHA tokens
- **Health Check**: Health endpoint includes reCAPTCHA token status

#### Background Tasks:
- **Periodic Refresh**: Every 30 minutes, refresh cf_clearance AND reCAPTCHA tokens
- **Startup**: Initial data fetch includes reCAPTCHA token extraction

## Request Flow Integration

### Standard Requests:
```
1. Initial request with auth token
2. If 403 error → Check if reCAPTCHA error
3. If reCAPTCHA error → Extract fresh token
4. Retry request with reCAPTCHA token
5. Success or final failure
```

### Streaming Requests:
```
1. Initial streaming request with auth token
2. If 403 error → Check if reCAPTCHA error  
3. If reCAPTCHA error → Extract fresh token
4. Retry streaming request with reCAPTCHA token and proper headers
5. Stream success or final failure
```

## Configuration

### Default Anchor URL:
```python
DEFAULT_LMARENA_ANCHOR_URL = "https://www.google.com/recaptcha/enterprise/anchor?ar=1&k=6Led_uYrAAAAAKjxDIF58fgFtX3t8loNAK85bW9I&co=aHR0cHM6Ly9sbWFyZW5hLmFpOjQ0Mw==&hl=de&v=jdMmXeCQEkPbnFDy9T04NbgJ&size=invisible&anchor-ms=20000&execute-ms=15000&cb=rtb16dw1hds"
```

### Cache TTL: 120 seconds (2 minutes)

### Header Support:
- **Non-streaming**: `Accept: */*`
- **Streaming**: `Accept: text/event-stream`
- **Always includes**: `User-Agent`, `x-recaptcha-token` (when available)

## Error Handling

### reCAPTCHA Error Detection:
- Checks 403 responses for reCAPTCHA-related error messages
- Distinguishes between reCAPTCHA errors and other auth issues
- Automatically retries with fresh token

### Retry Logic:
- **Max Retries**: 3 attempts total
- **reCAPTCHA Retry**: 2-second delay for reCAPTCHA-related retries
- **Token Cycling**: Uses round-robin with failed token exclusion

## API Endpoints

### New Endpoints:
- `POST /refresh-recaptcha`: Manually refresh reCAPTCHA token
- `GET /api/v1/health`: Enhanced health check with reCAPTCHA status

### Enhanced Endpoints:
- `POST /refresh-tokens`: Now also refreshes reCAPTCHA tokens
- `GET /dashboard`: Includes reCAPTCHA status section

## Testing

Run the comprehensive test suite:
```bash
python3 test_recaptcha_bypass.py
```

Tests cover:
- ✅ recaptcha_bypass.py module functionality
- ✅ Header generation with reCAPTCHA token support
- ✅ Token caching and management
- ✅ Error handling for reCAPTCHA validation failures
- ✅ Integration with existing cf_clearance flow
- ✅ Support for both streaming and non-streaming requests

## Dependencies

### Existing Dependencies:
- `requests`, `re`, `httpx`, `httpx_async`
- `camoufox` (for browser automation)
- `fastapi`, `uvicorn` (for web framework)

### No External Dependencies:
- ✅ Uses only existing dependencies
- ✅ No external captcha-solving services
- ✅ No additional API keys or services required

## Technical Details

### Browser Automation:
- Uses AsyncCamoufox for headless browser automation
- Handles third-party cookie/storage access
- Manages dynamic state partitioning issues
- Extracts tokens from challenge responses

### Token Extraction Methods:
1. **Response Interception**: Captures token from HTTP responses
2. **JavaScript Execution**: Runs JS to extract from browser state
3. **Storage Access**: Checks localStorage, sessionStorage, cookies
4. **Page Content Parsing**: Regex patterns for token extraction

### Cache Management:
- Time-based expiration (2 minutes)
- Domain-specific caching
- Automatic cleanup of expired tokens

## Acceptance Criteria Status

- ✅ **Enterprise reCAPTCHA v3 token successfully extracted and returned**
- ✅ **Token integration into existing cf_clearance/session refresh flow**
- ✅ **No external dependencies added** (uses only requests, re, httpx, httpx_async)
- ✅ **Proper logging of bypass attempts and token acquisition**
- ✅ **Handles third-party cookie partitioning appropriately**
- ✅ **Updated background refresh tasks properly manage reCAPTCHA tokens alongside cf_clearance**
- ✅ **Error messages distinguish between reCAPTCHA bypass failures and other auth issues**

## Deployment Notes

1. **Startup**: Server automatically fetches initial reCAPTCHA token
2. **Background**: 30-minute periodic refresh of all tokens
3. **Manual**: Dashboard provides manual refresh options
4. **Monitoring**: Health endpoint includes reCAPTCHA status
5. **Debugging**: Comprehensive logging for troubleshooting

## Troubleshooting

### Common Issues:
1. **Token Extraction Fails**: Check browser automation dependencies
2. **403 Errors Persist**: Verify anchor URL is current
3. **Cache Issues**: Manual refresh clears cache
4. **Import Errors**: Ensure both modules are in same directory

### Debug Mode:
Enable DEBUG=True in main.py for detailed logging of:
- Token extraction attempts
- Cache hits/misses
- Retry logic decisions
- Error handling details