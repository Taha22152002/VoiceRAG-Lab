import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import re
import os

# --- Configuration ---
# IMPORTANT: Replace with your actual credentials and sheet details
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
# Use the credentials JSON that exists within this project
CREDS_FILE = os.getenv('GOOGLE_CREDS_FILE', 'credentials.json')
SPREADSHEET_NAME = 'Carwash Appointments' # The full name of Google Sheet
WORKSHEET_NAME = 'Appointments' # The tab name where the schedule is located

# Define the expected time columns to make the code robust
TIME_COLUMNS = ['9:00 AM', '10:00 AM', '11:00 AM', '12:00 PM', '1:00 PM', '2:00 PM', '3:00 PM', '4:00 PM']

def get_available_slots(date: str) -> list[dict]:
    """
    Retrieves all available (empty) time slots for a specified date from the Google Sheet.
    Uses 'Date' as the sole row identifier.
    Args:
        date: The date in 'YYYY-MM-DD' format (e.g., '2025-11-13').
    Returns:
        A list of dictionaries, each representing an available slot.
    """
    try:
        # Authenticate and open the sheet
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, SCOPE)
        client = gspread.authorize(creds)
        sheet = client.open(SPREADSHEET_NAME).worksheet(WORKSHEET_NAME)

        # Get all records to find the row index (gspread row index starts at 1 for headers)
        data = sheet.get_all_records()
        df = pd.DataFrame(data)

        # Filter rows by the requested date
        schedule_rows = df[df['Date'] == date]

        if schedule_rows.empty:
            return [{"status": "No schedule found for this date."}]

        available_slots = []
        
        # Iterate through the filtered rows (should only be one if 'Date' is unique)
        for index, row in schedule_rows.iterrows():
            for time in TIME_COLUMNS:
                # Use .get(time) for safety, though TIME_COLUMNS should match sheet headers
                slot_value = row.get(time)

                # Check for availability: empty cell is available (gspread returns empty string for blank cells)
                if not slot_value:
                    available_slots.append({
                        "date": date,
                        "time": time,
                        "status": "Available"
                    })
                # Check for explicit unavailability (case-insensitive)
                elif slot_value.strip().lower() == 'suday':
                    continue # Skip this slot

        if not available_slots:
            return [{"status": f"No available slots found on {date}."}]

        return available_slots

    except Exception as e:
        # Log the full error for debugging
        print(f"Error fetching schedule: {e}")
        return [{"status": "Error", "message": str(e)}]

# -------------------------------------------------------------------------------------------------------

def book_slot(date: str, time: str, user_id: str) -> dict:
    """
    Books a specific time slot by writing the user_id into the corresponding cell.
    Uses 'Date' as the sole row identifier.
    Args:
        date: The date in 'YYYY-MM-DD' format.
        time: The time slot string (e.g., '10:00 AM').
        user_id: The ID of the user booking the slot (e.g., 'Taha-9999').
    Returns:
        A confirmation dictionary.
    """
    try:
        # Authenticate and open the sheet
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, SCOPE)
        client = gspread.authorize(creds)
        sheet = client.open(SPREADSHEET_NAME).worksheet(WORKSHEET_NAME)

        # Get all values (including header) for accurate row/column lookup
        data = sheet.get_all_values() 
        headers = data[0]
        df = pd.DataFrame(data[1:], columns=headers) # Create DF from rows, using headers

        # --- 1. Find the exact row index in the Google Sheet ---
        schedule_row_df = df[df['Date'] == date]
        
        if schedule_row_df.empty:
            return {"status": "Error", "message": f"Date {date} not found."}

        # Get the pandas index of the found row (assuming unique date)
        df_index = schedule_row_df.index[0]
        # Convert to gspread row index: +1 for 0-based pandas index, +1 for header row
        sheet_row_index = df_index + 2 

        # --- 2. Find the column index in the Google Sheet ---
        try:
            col_index = headers.index(time) + 1 # gspread is 1-indexed
        except ValueError:
            return {"status": "Error", "message": f"Time slot {time} not valid or header not found."}

        # --- 3. Check if the slot is already booked (pre-update check) ---
        # Get the current value directly from the sheet using the determined indices
        current_value = sheet.cell(sheet_row_index, col_index).value
        
        # Check if the slot is already booked (not empty and not the 'suday' marker)
        if current_value and current_value.strip().lower() not in ('suday', ''):
            return {"status": "Error", "message": f"Slot already booked by {current_value}."}

        # --- 4. Update the cell with the user_id ---
        sheet.update_cell(sheet_row_index, col_index, user_id)

        return {
            "status": "Success",
            "message": f"Slot booked successfully on {date} at {time} for {user_id}.",
            "cell_updated": f"Row {sheet_row_index}, Column {col_index}"
        }

    except Exception as e:
        print(f"Error booking slot: {e}")
        return {"status": "Error", "message": str(e)}
    
