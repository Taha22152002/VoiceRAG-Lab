#!/usr/bin/env python3
"""
Lightweight test for booking_tools import and RagBot tool-calling.

This script:
- Verifies importing booking_tools
- Mocks Gemini client to trigger a function_call to get_available_slots
- Monkeypatches BookingToolExecutor.safe_execute to avoid network calls
- Calls RagBot.generate_response_with_tools and prints the results
"""

import os
import sys

# Ensure we can import local modules when run from repo root or backend dir
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

print("Step 1: Testing booking_tools import...")
try:
    import booking_tools  # noqa: F401
    print("Import successful: booking_tools")
except Exception as e:
    print(f"Import failed: {e}")
    raise

print("Step 2: Testing RagBot tool-calling with mocked Gemini client...")
from rag_core import RagBot  # noqa: E402


class FakeFunctionCall:
    def __init__(self, name: str, args: dict):
        self.name = name
        self.args = args


class FakePart:
    def __init__(self, function_call=None):
        # When present, mimics google.genai.types.Part.function_call
        self.function_call = function_call


class FakeContent:
    def __init__(self, parts):
        self.parts = parts


class FakeCandidate:
    def __init__(self, content, text=""):
        self.content = content
        self.text = text


class FakeResponse:
    def __init__(self, candidates=None, text=""):
        self.candidates = candidates or []
        self.text = text


class FakeModels:
    def __init__(self):
        self._call_count = 0

    def generate_content(self, model, contents, config=None):  # noqa: D401
        """Return a function_call on first call, and a text response on second."""
        self._call_count += 1
        if self._call_count == 1:
            # First call: trigger function_call to get_available_slots
            fc = FakeFunctionCall(
                name="get_available_slots",
                args={"date": "2025-11-15"},
            )
            part = FakePart(function_call=fc)
            content = FakeContent(parts=[part])
            return FakeResponse(candidates=[FakeCandidate(content=content)])
        else:
            # Second call: return final text after tool execution
            return FakeResponse(text="Available slots for 2025-11-15: 9:00 AM, 10:00 AM")


class FakeClient:
    def __init__(self):
        self.models = FakeModels()


def main():
    # Create RagBot with a dummy API key and inject fake client
    api_key = os.getenv("GOOGLE_API_KEY", "test-key")
    rag_bot = RagBot(api_key=api_key)
    rag_bot.client = FakeClient()

    # Monkeypatch booking executor to avoid real HTTP calls
    def fake_safe_execute(function_name, **kwargs):
        if function_name == "get_available_slots":
            return {
                "status": "success",
                "available_slots": ["9:00 AM", "10:00 AM"],
                "message": f"Available slots on {kwargs.get('date')}: 9:00 AM, 10:00 AM",
            }
        elif function_name == "book_appointment_slot":
            return {
                "status": "success",
                "message": f"Appointment confirmed for {kwargs.get('time')} on {kwargs.get('date')}",
                "booking_details": {"user_id": kwargs.get("user_id")},
            }
        return {"status": "error", "message": "Unknown function"}

    rag_bot.booking_executor.safe_execute = fake_safe_execute  # type: ignore

    # Invoke tool-calling flow
    result = rag_bot.generate_response_with_tools(
        user_message="Please check available appointments for 2025-11-15",
        system_prompt="You are helpful.",
        chat_history=[],
        user_id="u123",
    )

    print("Step 3: Results")
    print("Mode:", result.get("mode"))
    print("Tool used:", result.get("tool_used"))
    print("Tool result:", result.get("tool_result"))
    print("Model response:", result.get("response"))


if __name__ == "__main__":
    main()

"""
Unit tests for booking tools functionality
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json
from datetime import datetime, timedelta
from booking_tools import BookingToolExecutor, validate_booking_params, BOOKING_TOOLS


class TestBookingTools(unittest.TestCase):
    """Test cases for booking tools"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.executor = BookingToolExecutor()
    
    def test_validate_booking_params_valid_date(self):
        """Test valid date validation"""
        result = validate_booking_params("2024-01-15", "14:00")
        self.assertTrue(result["valid"])
        self.assertIsNone(result["error"])
    
    def test_validate_booking_params_invalid_date_format(self):
        """Test invalid date format"""
        result = validate_booking_params("15-01-2024", "14:00")
        self.assertFalse(result["valid"])
        self.assertIn("Invalid date format", result["error"])
    
    def test_validate_booking_params_invalid_time_format(self):
        """Test invalid time format"""
        result = validate_booking_params("2024-01-15", "2:00 PM")
        self.assertFalse(result["valid"])
        self.assertIn("Invalid time format", result["error"])
    
    def test_validate_booking_params_past_date(self):
        """Test past date validation"""
        past_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        result = validate_booking_params(past_date, "14:00")
        self.assertFalse(result["valid"])
        self.assertIn("cannot be in the past", result["error"])
    
    def test_validate_booking_params_invalid_time_range(self):
        """Test invalid time range"""
        result = validate_booking_params("2024-01-15", "25:00")
        self.assertFalse(result["valid"])
        self.assertIn("Invalid time format", result["error"])
    
    @patch('requests.get')
    def test_get_available_slots_success(self, mock_get):
        """Test successful get available slots"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "available_slots": ["09:00", "10:00", "11:00", "14:00", "15:00"]
        }
        mock_get.return_value = mock_response
        
        result = self.executor.get_available_slots("2024-01-15")
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["slots"]), 5)
        self.assertIn("09:00", result["slots"])
    
    @patch('requests.get')
    def test_get_available_slots_api_error(self, mock_get):
        """Test API error handling"""
        # Mock API error
        mock_get.side_effect = Exception("API connection failed")
        
        result = self.executor.get_available_slots("2024-01-15")
        
        self.assertEqual(result["status"], "error")
        self.assertIn("Failed to fetch available slots", result["message"])
    
    @patch('requests.post')
    def test_book_appointment_slot_success(self, mock_post):
        """Test successful appointment booking"""
        # Mock successful booking response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "booking_id": "BOOK123",
            "date": "2024-01-15",
            "time": "14:00",
            "status": "confirmed"
        }
        mock_post.return_value = mock_response
        
        result = self.executor.book_appointment_slot("2024-01-15", "14:00", "user123")
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["booking_id"], "BOOK123")
        self.assertEqual(result["date"], "2024-01-15")
        self.assertEqual(result["time"], "14:00")
    
    @patch('requests.post')
    def test_book_appointment_slot_already_booked(self, mock_post):
        """Test already booked slot handling"""
        # Mock already booked response
        mock_response = Mock()
        mock_response.status_code = 409  # Conflict
        mock_response.json.return_value = {
            "error": "Slot already booked"
        }
        mock_post.return_value = mock_response
        
        result = self.executor.book_appointment_slot("2024-01-15", "14:00", "user123")
        
        self.assertEqual(result["status"], "error")
        self.assertIn("already booked", result["message"])
    
    @patch('requests.post')
    def test_book_appointment_slot_api_error(self, mock_post):
        """Test booking API error handling"""
        mock_post.side_effect = Exception("API connection failed")
        
        result = self.executor.book_appointment_slot("2024-01-15", "14:00", "user123")
        
        self.assertEqual(result["status"], "error")
        self.assertIn("Failed to book appointment", result["message"])
    
    def test_safe_execute_valid_function(self):
        """Test safe execute with valid function"""
        with patch.object(self.executor, 'get_available_slots') as mock_get:
            mock_get.return_value = {"status": "success", "slots": ["09:00", "10:00"]}
            
            result = self.executor.safe_execute("get_available_slots", date="2024-01-15")
            
            self.assertEqual(result["status"], "success")
            self.assertEqual(len(result["slots"]), 2)
    
    def test_safe_execute_invalid_function(self):
        """Test safe execute with invalid function"""
        result = self.executor.safe_execute("invalid_function")
        
        self.assertEqual(result["status"], "error")
        self.assertIn("Unknown function", result["message"])
    
    def test_safe_execute_missing_params(self):
        """Test safe execute with missing required parameters"""
        result = self.executor.safe_execute("get_available_slots")
        
        self.assertEqual(result["status"], "error")
        self.assertIn("Missing required parameter", result["message"])
    
    def test_safe_execute_validation_error(self):
        """Test safe execute with validation error"""
        result = self.executor.safe_execute("get_available_slots", date="invalid-date")
        
        self.assertEqual(result["status"], "error")
        self.assertIn("Invalid date format", result["message"])
    
    def test_booking_tools_structure(self):
        """Test booking tools schema structure"""
        self.assertEqual(len(BOOKING_TOOLS), 2)
        
        # Check get_available_slots tool
        slots_tool = BOOKING_TOOLS[0]
        self.assertEqual(slots_tool["function_declarations"][0]["name"], "get_available_slots")
        self.assertIn("date", slots_tool["function_declarations"][0]["parameters"]["properties"])
        
        # Check book_appointment_slot tool
        book_tool = BOOKING_TOOLS[1]
        self.assertEqual(book_tool["function_declarations"][0]["name"], "book_appointment_slot")
        params = book_tool["function_declarations"][0]["parameters"]["properties"]
        self.assertIn("date", params)
        self.assertIn("time", params)
        self.assertIn("user_id", params)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)