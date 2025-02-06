import requests

# API URL
url = "http://127.0.0.1:8000/predict"

# Examples data
payload = {
    "description": ["Brand new in original packaging.", "High quality red and smoke tail lights."],
    "title": "Spec-D Tuning LT-E362RG-F2-APC",
    "feature": ["Features 1 pair of Red & Smoked lens Tail Lights", "Direct replacement for stock assembly"],
    "brand": "Spec-D Tuning",
    "price": 8.63,
    "image": ["https://images-na.ssl-images-amazon.com/images/I/616EQpQ2V7L._SS40_.jpg"],
    "also_buy": ["B007KLMLRM", "B007KLMNNE"],
    "also_view": ["B0085FOJ90", "B0085FOAWQ"]
    #main_cat: Automotive
}

payload2 = {
    "description": ['Clear plastic head allows you to tack up important documents and posters without unwanted visual distractions. Steel pin glides right into even the toughest surfaces: drywall, plaster and mat-board as well as cork and foam. Stock up and save Head Material: Plastic Head Diameter: 1/4amp;quot; Pin Material: Steel Colors: Clear.'],
    "title": "Universal 31304 3/8-Inch Clear Push Pins (100 per Pack)",
    "feature": ['Head_Material - Plastic', 'Head_Diameter - 1/4&quot;', 'Pin_Material - Steel', 'Colors - Clear'],
    "brand": "Universal",
    "price": 3.94,
    "image": [],
    "also_buy": [],
    "also_view": []
    #main_cat: Home Audio & Theater
}

payload3 = {
    "description": ['Book'],
    "title": "Yoga sensible al trauma: Sanando desde el interior",
    "feature": [],
    "brand": "Amazon",
    "price": 3.94,
    "image": [],
    "also_buy": [],
    "also_view": []
    #main_cat: Books
}

# Send POST request
response = requests.post(url, json=payload)

# Handle errors
if response.status_code == 200:
    print("Response JSON (200):", response.json())
else:
    print(f"Error: {response.status_code}")
    print("Response Text:", response.text)
