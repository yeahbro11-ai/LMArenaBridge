#!/usr/bin/env python3
"""
Test script for Enterprise reCAPTCHA v3 bypass integration.

This script tests the key components:
1. recaptcha_bypass.py module functionality
2. Integration with main.py header generation
3. Token caching and management
4. Error handling for reCAPTCHA validation failures
"""

import sys
import os
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from recaptcha_bypass import RecaptchaBypass, get_recaptcha_bypass, extract_recaptcha_token
from main import (
    get_request_headers_with_token,
    get_cached_recaptcha_token,
    cache_recaptcha_token,
    get_or_extract_recaptcha_token,
    handle_recaptcha_error
)


def test_recaptcha_bypass_module():
    """Test the recaptcha_bypass module basic functionality"""
    print("🧪 Testing recaptcha_bypass module...")
    
    # Test module import
    try:
        bypass = RecaptchaBypass()
        assert bypass.cache_ttl == 120
        assert len(bypass._token_cache) == 0
        print("✅ RecaptchaBypass class instantiation successful")
    except Exception as e:
        print(f"❌ RecaptchaBypass instantiation failed: {e}")
        return False
    
    # Test cache functionality
    test_token = "test_token_12345"
    bypass._cache_token("test.com", test_token)
    
    cached = bypass._get_cached_token("test.com")
    assert cached == test_token
    print("✅ Token caching works correctly")
    
    # Test cache expiration
    bypass._token_cache["test.com"] = (time.time() - 130, "expired")  # Expired
    assert bypass._get_cached_token("test.com") is None
    print("✅ Cache expiration works correctly")
    
    return True


def test_header_generation_with_recaptcha():
    """Test header generation with reCAPTCHA token integration"""
    print("\n🧪 Testing header generation with reCAPTCHA...")
    
    # Test headers without reCAPTCHA token
    headers = get_request_headers_with_token("test_token")
    assert "x-recaptcha-token" not in headers
    assert headers["Cookie"] == "cf_clearance=; arena-auth-prod-v1=test_token"
    assert headers["Accept"] == "*/*"
    print("✅ Headers without reCAPTCHA token work correctly")
    
    # Test headers with reCAPTCHA token
    headers = get_request_headers_with_token("test_token", recaptcha_token="recaptcha_123")
    assert headers["x-recaptcha-token"] == "recaptcha_123"
    assert headers["Cookie"] == "cf_clearance=; arena-auth-prod-v1=test_token"
    print("✅ Headers with reCAPTCHA token work correctly")
    
    # Test streaming headers
    headers = get_request_headers_with_token("test_token", recaptcha_token="recaptcha_123", for_streaming=True)
    assert headers["Accept"] == "text/event-stream"
    assert headers["x-recaptcha-token"] == "recaptcha_123"
    print("✅ Streaming headers with reCAPTCHA token work correctly")
    
    return True


def test_recaptcha_token_management():
    """Test reCAPTCHA token management functions"""
    print("\n🧪 Testing reCAPTCHA token management...")
    
    # Test token caching
    test_token = "cached_token_test"
    cache_recaptcha_token(test_token, "test.com")
    
    cached = get_cached_recaptcha_token("test.com")
    assert cached == test_token
    print("✅ Token caching function works correctly")
    
    # Test invalid cache
    cached_invalid = get_cached_recaptcha_token("nonexistent.com")
    assert cached_invalid is None
    print("✅ Invalid cache handling works correctly")
    
    return True


def test_recaptcha_error_handling():
    """Test reCAPTCHA error handling"""
    print("\n🧪 Testing reCAPTCHA error handling...")
    
    # Create mock response for reCAPTCHA error
    mock_response = Mock()
    mock_response.status_code = 403
    mock_response.json.return_value = {"error": "reCAPTCHA validation failed"}
    
    # Test reCAPTCHA error detection
    should_retry = handle_recaptcha_error(mock_response, 0, 3)
    assert should_retry == True
    print("✅ reCAPTCHA error detection works correctly")
    
    # Test non-reCAPTCHA error
    mock_response.json.return_value = {"error": "Access denied"}
    should_retry = handle_recaptcha_error(mock_response, 0, 3)
    assert should_retry == False
    print("✅ Non-reCAPTCHA error handling works correctly")
    
    # Test last attempt (should not retry)
    should_retry = handle_recaptcha_error(mock_response, 2, 3)
    assert should_retry == False
    print("✅ Last attempt handling works correctly")
    
    return True


async def test_async_functions():
    """Test async functions (with mocking)"""
    print("\n🧪 Testing async functions...")
    
    # Mock the extract_recaptcha_token function
    with patch('main.extract_recaptcha_token') as mock_extract:
        mock_extract.return_value = "mocked_recaptcha_token"
        
        # Test get_or_extract_recaptcha_token
        token = await get_or_extract_recaptcha_token()
        assert token == "mocked_recaptcha_token"
        print("✅ get_or_extract_recaptcha_token works correctly")
        
        # Verify the function was called
        mock_extract.assert_called_once()
        print("✅ extract_recaptcha_token integration works correctly")
    
    return True


async def test_full_integration():
    """Test full integration scenario"""
    print("\n🧪 Testing full integration scenario...")
    
    # Simulate a scenario where reCAPTCHA is needed
    test_auth_token = "test_auth_token_123"
    test_recaptcha_token = "test_recaptcha_token_456"
    
    # Generate headers as they would be in the retry logic
    headers = get_request_headers_with_token(
        test_auth_token, 
        recaptcha_token=test_recaptcha_token,
        for_streaming=True
    )
    
    # Verify all required headers are present
    assert "Content-Type" in headers
    assert "Cookie" in headers
    assert "User-Agent" in headers
    assert "Accept" in headers
    assert "x-recaptcha-token" in headers
    
    assert headers["Content-Type"] == "text/plain;charset=UTF-8"
    assert "arena-auth-prod-v1=test_auth_token_123" in headers["Cookie"]
    assert headers["Accept"] == "text/event-stream"
    assert headers["x-recaptcha-token"] == "test_recaptcha_token_456"
    
    print("✅ Full integration scenario works correctly")
    return True


def main():
    """Run all tests"""
    print("🚀 Starting Enterprise reCAPTCHA v3 Bypass Integration Tests")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Run synchronous tests
    try:
        all_tests_passed &= test_recaptcha_bypass_module()
        all_tests_passed &= test_header_generation_with_recaptcha()
        all_tests_passed &= test_recaptcha_token_management()
        all_tests_passed &= test_recaptcha_error_handling()
    except Exception as e:
        print(f"❌ Synchronous test failed: {e}")
        all_tests_passed = False
    
    # Run async tests
    try:
        all_tests_passed &= asyncio.run(test_async_functions())
        all_tests_passed &= asyncio.run(test_full_integration())
    except Exception as e:
        print(f"❌ Async test failed: {e}")
        all_tests_passed = False
    
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 All tests passed! Enterprise reCAPTCHA v3 bypass integration is working correctly.")
        print("\n✅ Key Features Verified:")
        print("   • recaptcha_bypass.py module functionality")
        print("   • Header generation with reCAPTCHA token support")
        print("   • Token caching and management")
        print("   • Error handling for reCAPTCHA validation failures")
        print("   • Integration with existing cf_clearance flow")
        print("   • Support for both streaming and non-streaming requests")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)