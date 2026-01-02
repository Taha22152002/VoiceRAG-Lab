import requests
import json
import re
from datetime import datetime
from typing import Dict, List, Any

# Tool definitions for Gemini function calling
# This is a reference implementation for an Appointment Booking system.
# Developers can replace this with their own tool definitions.
BOOKING_TOOLS = [
    {
        "function_declarations": [
            {
                "name": "get_available_slots",
                "description": "Get available appointment slots for a specific date",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                            "description": "Date in YYYY-MM-DD format",
                            "pattern": "^\\d{4}-\\d{2}-\\d{2}$"
                        }
                    },
                    "required": ["date"]
                }
            },
            {
                "name": "book_appointment_slot",
                "description": "Book a specific appointment slot",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                            "description": "Date in YYYY-MM-DD format",
                            "pattern": "^\\d{4}-\\d{2}-\\d{2}$"
                        },
                        "time": {
                            "type": "string",
                            "description": "Time slot (e.g., '9:00 AM', '10:00 AM')",
                            "enum": ["9:00 AM", "10:00 AM", "11:00 AM", "12:00 PM", "1:00 PM", "2:00 PM", "3:00 PM", "4:00 PM"]
                        },
                        "user_id": {
                            "type": "string",
                            "description": "Unique identifier for the user booking the slot"
                        }
                    },
                    "required": ["date", "time", "user_id"]
                }
            }
        ]
    }
]

class BookingToolExecutor:
    """
    Reference implementation of a tool executor.
    This class handles the logic for executing the tools defined in BOOKING_TOOLS.
    """
    def __init__(self, api_base_url: str = None):
        import os
        base = api_base_url or os.getenv("APPOINTMENT_API_BASE_URL") or os.getenv("APPOINTMENTS_API_BASE_URL") or "http://127.0.0.1:5200"
        base = base.rstrip('/')
        if base.endswith('/api'):
            base = base[:-4]
        self.api_base_urls = [base, "http://127.0.0.1:5200", "http://localhost:5200"]
        
    def get_available_slots(self, date: str) -> Dict[str, Any]:
        try:
            last_err = None
            for base in self.api_base_urls:
                try:
                    response = requests.get(
                        f"{base}/api/appointments/slots",
                        params={"date": date},
                        timeout=6,
                    )
                    if response.status_code == 200:
                        slots_data = response.json()
                        available_slots = [slot.get("time") for slot in slots_data if slot.get("status") == "Available"]
                        if available_slots:
                            return {"status": "success", "available_slots": available_slots, "message": f"Available slots on {date}: {', '.join(available_slots)}"}
                        else:
                            return {"status": "no_slots", "available_slots": [], "message": f"No available slots found for {date}"}
                    else:
                        last_err = response.text
                        continue
                except requests.exceptions.RequestException as e:
                    last_err = str(e)
                    continue
            return {"status": "error", "available_slots": [], "message": f"Failed to get slots: {last_err or 'Unknown error'}"}
        except Exception as e:
            return {"status": "error", "available_slots": [], "message": f"Unexpected error: {str(e)}"}
    
    def book_appointment_slot(self, date: str, time: str, user_id: str) -> Dict[str, Any]:
        try:
            payload = {"date": date, "time": time, "user_id": user_id}
            last_err = None
            for base in self.api_base_urls:
                try:
                    response = requests.post(
                        f"{base}/api/appointments/slots",
                        json=payload,
                        timeout=6,
                    )
                    booking_data = response.json() if response.text else {}
                    if response.status_code in (200, 201) or str(booking_data.get("status", "")).lower() in {"success", "ok", "booked"}:
                        return {"status": "success", "message": booking_data.get("message") or f"✅ Appointment confirmed for {time} on {date}", "booking_details": booking_data}
                    if response.status_code == 409:
                        return {"status": "already_booked", "message": f"❌ Sorry, the {time} slot on {date} is already booked"}
                    last_err = booking_data.get('error') or response.text
                    continue
                except requests.exceptions.RequestException as e:
                    last_err = str(e)
                    continue
            return {"status": "error", "message": f"❌ Booking failed: {last_err or 'Unknown error'}"}
        except Exception as e:
            return {"status": "error", "message": f"❌ Unexpected error: {str(e)}"}
    
    def safe_execute(self, function_name: str, **kwargs) -> Dict[str, Any]:
        """Safely execute a function with error handling"""
        try:
            # Validate parameters
            if function_name == "get_available_slots":
                validation = validate_booking_params(kwargs.get("date"))
                if not validation["valid"]:
                    return {
                        "status": "validation_error",
                        "message": f"❌ Validation failed: {'; '.join(validation['errors'])}"
                    }
            
            elif function_name == "book_appointment_slot":
                validation = validate_booking_params(kwargs.get("date"), kwargs.get("time"))
                if not validation["valid"]:
                    return {
                        "status": "validation_error", 
                        "message": f"❌ Validation failed: {'; '.join(validation['errors'])}"
                    }
                
                # Validate user_id
                if not kwargs.get("user_id") or len(kwargs.get("user_id")) < 3:
                    return {
                        "status": "validation_error",
                        "message": "❌ User ID must be at least 3 characters long"
                    }
            
            # Execute the actual function
            return getattr(self, function_name)(**kwargs)
            
        except Exception as e:
            return {
                "status": "execution_error",
                "message": f"❌ Function execution failed: {str(e)}"
            }

def validate_booking_params(date: str, time: str = None) -> Dict[str, Any]:
    """Validate booking parameters"""
    errors = []
    
    # Validate date format
    date_pattern = r'^\d{4}-\d{2}-\d{2}$'
    if not re.match(date_pattern, date):
        errors.append("Date must be in YYYY-MM-DD format")
    else:
        try:
            # Check if date is valid and not in the past
            booking_date = datetime.strptime(date, '%Y-%m-%d').date()
            today = datetime.now().date()
            if booking_date < today:
                errors.append("Cannot book appointments in the past")
        except ValueError:
            errors.append("Invalid date provided")
    
    # Validate time if provided
    if time:
        valid_times = ["9:00 AM", "10:00 AM", "11:00 AM", "12:00 PM", 
                      "1:00 PM", "2:00 PM", "3:00 PM", "4:00 PM"]
        if time not in valid_times:
            errors.append(f"Invalid time slot. Available: {', '.join(valid_times)}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }