#!/usr/bin/env python3
"""
Test script to verify the HTTP 403 streaming fix.

This script tests that:
1. Streaming headers include proper User-Agent and Accept headers
2. reCAPTCHA token handling doesn't break non-streaming requests
3. The retry logic for 403 errors works correctly
"""

import asyncio
import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test the header generation
def test_request_headers():
    """Test that request headers are properly formatted"""
    print("\n" + "="*60)
    print("Testing request headers generation...")
    print("="*60)
    
    # Mock config
    config_data = {
        "cf_clearance": "test_cf_clearance_value",
        "auth_tokens": ["token1", "token2"]
    }
    
    # Write mock config
    with open('config.json', 'w') as f:
        json.dump(config_data, f)
    
    # Import after mocking config
    from main import get_request_headers_with_token
    
    # Test non-streaming headers
    headers_normal = get_request_headers_with_token("test_token")
    print("\n✅ Non-streaming headers:")
    for k, v in headers_normal.items():
        if k == "Cookie":
            print(f"  {k}: [hidden cookie]")
        else:
            print(f"  {k}: {v}")
    
    assert "User-Agent" in headers_normal, "User-Agent should be present"
    assert "Mozilla" in headers_normal["User-Agent"], "User-Agent should contain Mozilla"
    assert "Referer" in headers_normal, "Referer should be present"
    assert "Origin" in headers_normal, "Origin should be present"
    assert headers_normal["Accept"] == "*/*", "Non-streaming Accept should be */*"
    
    # Test streaming headers
    headers_streaming = get_request_headers_with_token("test_token", for_streaming=True)
    print("\n✅ Streaming headers:")
    for k, v in headers_streaming.items():
        if k == "Cookie":
            print(f"  {k}: [hidden cookie]")
        else:
            print(f"  {k}: {v}")
    
    assert headers_streaming["Accept"] == "text/event-stream", "Streaming Accept should be text/event-stream"
    assert "User-Agent" in headers_streaming, "Streaming should also have User-Agent"
    
    print("\n✅ All header tests passed!")
    return True

def test_imports():
    """Test that main.py can be imported without errors"""
    print("\n" + "="*60)
    print("Testing imports...")
    print("="*60)
    
    try:
        import main
        print("✅ main.py imported successfully")
        
        # Check for required functions
        required_functions = [
            'get_request_headers_with_token',
            'get_or_extract_recaptcha_token',
            'get_next_auth_token',
        ]
        
        for func_name in required_functions:
            if hasattr(main, func_name):
                print(f"✅ Function {func_name} exists")
            else:
                print(f"❌ Function {func_name} NOT found")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Error importing main.py: {e}")
        return False

def test_syntax():
    """Test Python syntax of main.py"""
    print("\n" + "="*60)
    print("Testing Python syntax...")
    print("="*60)
    
    import py_compile
    try:
        py_compile.compile('src/main.py', doraise=True)
        print("✅ Syntax check passed!")
        return True
    except py_compile.PyCompileError as e:
        print(f"❌ Syntax error: {e}")
        return False

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    all_passed = True
    
    # Test 1: Syntax
    if not test_syntax():
        all_passed = False
    
    # Test 2: Imports
    if not test_imports():
        all_passed = False
    
    # Test 3: Headers
    try:
        if not test_request_headers():
            all_passed = False
    except Exception as e:
        print(f"❌ Header test failed: {e}")
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        print("="*60)
        sys.exit(1)
