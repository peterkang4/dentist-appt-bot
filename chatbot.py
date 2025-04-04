import os
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration 
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

# Initialize client
client = OpenAI(api_key=API_KEY)

# Data storage
DATA_FILE = Path("appointment_data.json")
if not DATA_FILE.exists():
    with open(DATA_FILE, "w") as f:
        json.dump({
            "available_slots": {
                "Monday": ["9:00", "10:00", "11:00", "14:00", "15:00", "16:00"],
                "Tuesday": ["9:00", "10:00", "11:00", "14:00", "15:00", "16:00"],
                "Wednesday": ["9:00", "10:00", "11:00", "14:00", "15:00", "16:00"],
                "Thursday": ["9:00", "10:00", "11:00", "14:00", "15:00", "16:00"],
                "Friday": ["9:00", "10:00", "11:00", "14:00", "15:00", "16:00"]
            },
            "booked_appointments": []
        }, f, indent=4)

# Load data
def load_data():
    with open(DATA_FILE, "r") as f:
        return json.load(f)

# Save data
def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

# Check if slot is available
def is_slot_available(day, time):
    data = load_data()
    return time in data["available_slots"].get(day, [])

# Book appointment
def book_appointment(patient_name, contact, day, time, reason):
    data = load_data()
    
    if not is_slot_available(day, time):
        return False
    
    # Remove slot from available slots
    data["available_slots"][day].remove(time)
    
    # Add to booked appointments
    data["booked_appointments"].append({
        "patient_name": patient_name,
        "contact": contact,
        "day": day,
        "time": time,
        "reason": reason,
        "booking_id": f"DENT-{len(data['booked_appointments']) + 1:04d}"
    })
    
    save_data(data)
    return data["booked_appointments"][-1]["booking_id"]

# Get available slots for a day
def get_available_slots(day):
    data = load_data()
    return data["available_slots"].get(day, [])

# Cancel appointment
def cancel_appointment(booking_id):
    data = load_data()
    
    for i, appointment in enumerate(data["booked_appointments"]):
        if appointment["booking_id"] == booking_id:
            # Add the slot back to available slots
            day = appointment["day"]
            time = appointment["time"]
            if day in data["available_slots"]:
                data["available_slots"][day].append(time)
                data["available_slots"][day].sort()
            
            # Remove from booked appointments
            removed = data["booked_appointments"].pop(i)
            save_data(data)
            return removed
    
    return None

# Chatbot system prompt
SYSTEM_PROMPT = """
You are a helpful dental office assistant chatbot. Your job is to help patients book, reschedule, or cancel appointments.
The dental office is open Monday through Friday, 9:00 AM to 5:00 PM.
Be friendly, concise, and collect all necessary information to book an appointment:
1. Patient name
2. Contact information (phone or email)
3. Preferred day and time
4. Reason for visit (cleaning, check-up, specific issue, etc.)

Once you have all the information, confirm the details with the patient before finalizing the booking.
"""

# Process user message and manage conversation state
def process_message(user_input, conversation_history=None):
    if conversation_history is None:
        conversation_history = []
    
    # Extract appointment details if we're in booking mode
    appointment_info = extract_appointment_info(user_input, conversation_history)
    
    # Create the messages for GPT
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT + f"\n\nCurrent available slots: {json.dumps(load_data()['available_slots'], indent=2)}"
        }
    ]
    
    # Add conversation history
    for message in conversation_history:
        messages.append(message)
    
    # Add the current user message
    messages.append({"role": "user", "content": user_input})
    
    # If we have all the appointment info, add it as a system message
    if appointment_info and all(appointment_info.values()):
        day = appointment_info['day']
        time = appointment_info['time']
        
        if is_slot_available(day, time):
            booking_id = book_appointment(
                appointment_info['name'],
                appointment_info['contact'],
                day,
                time,
                appointment_info['reason']
            )
            
            booking_confirmation = f"""
            An appointment has been booked with the following details:
            Booking ID: {booking_id}
            Patient: {appointment_info['name']}
            Contact: {appointment_info['contact']}
            Day: {day}
            Time: {time}
            Reason: {appointment_info['reason']}
            """
            
            messages.append({"role": "system", "content": booking_confirmation})
        else:
            messages.append({
                "role": "system", 
                "content": f"The requested slot ({day} at {time}) is not available. Available slots for {day}: {get_available_slots(day)}"
            })
    
    # Generate response with GPT
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Changed from gpt-4-turbo-preview
        messages=messages
    )
    
    return response.choices[0].message.content, conversation_history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": response.choices[0].message.content}
    ]

# Extract appointment information from conversation
def extract_appointment_info(user_input, conversation_history):
    # Example simplified extraction - in a real application you might use a more robust NER system
    # or have Claude extract structured data
    
    # Combine all the conversation for analysis
    full_conversation = " ".join([msg["content"] for msg in conversation_history] + [user_input])
    
    # Initialize appointment info
    appointment_info = {
        "name": None,
        "contact": None,
        "day": None,
        "time": None,
        "reason": None
    }
    
    # Create a prompt to extract structured data
    extraction_prompt = f"""
    Based on the following conversation, extract the appointment booking information if present:
    
    {full_conversation}
    
    Extract the following fields:
    - Patient name
    - Contact information (phone or email)
    - Preferred day
    - Preferred time
    - Reason for visit
    
    Format your response as JSON.
    """
    
    try:
        # Use GPT to extract structured data
        extraction_response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Changed from gpt-4-turbo-preview
            messages=[
                {"role": "user", "content": extraction_prompt}
            ]
        )
        
        # Get response content
        response_text = extraction_response.choices[0].message.content
        
        # Find JSON block if it exists
        import re
        json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find just a JSON object
            json_match = re.search(r'({.*})', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response_text
        
        extracted_data = json.loads(json_str)
        
        # Update appointment info with extracted data
        if "name" in extracted_data and extracted_data["name"]:
            appointment_info["name"] = extracted_data["name"]
        if "contact" in extracted_data and extracted_data["contact"]:
            appointment_info["contact"] = extracted_data["contact"]
        if "day" in extracted_data and extracted_data["day"]:
            appointment_info["day"] = extracted_data["day"]
        if "time" in extracted_data and extracted_data["time"]:
            appointment_info["time"] = extracted_data["time"]
        if "reason" in extracted_data and extracted_data["reason"]:
            appointment_info["reason"] = extracted_data["reason"]
            
    except Exception as e:
        print(f"Error extracting appointment info: {e}")
    
    return appointment_info

# Simple command-line interface for testing
def main():
    print("Dental Appointment Chatbot")


    print("Type 'quit' to exit")
    
    conversation_history = []
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Thank you for using our dental appointment service. Goodbye!")
            break
        
        response, conversation_history = process_message(user_input, conversation_history)
        print(f"\nChatbot: {response}")

if __name__ == "__main__":
    main()