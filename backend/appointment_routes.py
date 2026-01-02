from flask import Blueprint, request, jsonify
from apifunctions import get_available_slots, book_slot 

appointment_bp = Blueprint('appointment_bp', __name__)

@appointment_bp.route('/api/appointments/slots', methods=['GET', 'POST'])
def handle_appointments():
    """
    Handles both GET (retrieve available slots) and POST (book a slot) 
    requests for the appointment schedule.
    """
    if request.method == 'GET':
        ## --- 1. GET: Retrieve Available Slots ---
        
        # Get the 'date' from the query parameters (e.g., ?date=2025-11-13)
        date = request.args.get('date')
        
        if not date:
            return jsonify({"error": "Missing 'date' query parameter."}), 400
        
        # Call the existing function
        result = get_available_slots(date)
        
        # Return the result list as JSON
        return jsonify(result), 200

    elif request.method == 'POST':
        ## --- 2. POST: Book a Slot ---
        
        # Get the data from the JSON request body
        data = request.get_json()
        
        # Validate required fields
        date = data.get('date')
        time = data.get('time')
        user_id = data.get('user_id')

        if not all([date, time, user_id]):
            return jsonify({
                "error": "Missing required fields in body. Need 'date', 'time', and 'user_id'."
            }), 400
        
        # Call the existing function
        result = book_slot(date, time, user_id)
        
        # Check if booking failed (e.g., already booked, date/time not found)
        if result.get('status') == 'Error':
            # Use 409 Conflict if already booked, otherwise 500 for other errors
            status_code = 409 if 'already booked' in result.get('message', '').lower() else 500
            return jsonify(result), status_code
        
        # Success: Return 201 Created
        return jsonify(result), 201
