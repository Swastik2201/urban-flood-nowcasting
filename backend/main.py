
from fastapi import FastAPI, UploadFile, File
from dotenv import load_dotenv
from google import genai
from PIL import Image
import os
import io
import json
import requests
# Load Gemini API key
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

app = FastAPI(
    title="Urban Flood Nowcasting API",
    description="Backend for Urban Flood Nowcasting",
    version="1.0.0"
)

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)


@app.get("/")
def home():
    return {
        "message": "Urban Flood Nowcasting API is running!"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):

    # Read uploaded image
    image_bytes = await file.read()

    # Convert bytes to image
    image = Image.open(io.BytesIO(image_bytes))

    prompt = """
You are analyzing an image for an Urban Flood Nowcasting system.

Analyze the uploaded image and determine:

1. Is there a visible drainage blockage?
2. What is the blockage severity: low, medium, or high?
3. Is garbage or debris visible?
4. Is water accumulation visible?
5. Give a short observation.

Return ONLY valid JSON in exactly this format:

{
    "blockage_detected": true,
    "blockage_severity": "high",
    "garbage_detected": true,
    "water_accumulation": true,
    "observation": "Short description"
}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[prompt, image]
    )

    # Get Gemini response
    result = response.text.strip()

    # Remove markdown code fences if Gemini adds them
    if result.startswith("```"):
        result = result.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        data = {
            "raw_response": result
        }

    return {
        "filename": file.filename,
        "analysis": data
    }
@app.get("/rainfall")
def get_rainfall(latitude: float, longitude: float):

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "rain,precipitation",
        "hourly": "rain,precipitation",
        "forecast_days": 1,
        "timezone": "auto"
    }

    response = requests.get(url, params=params, timeout=10)

    if response.status_code != 200:
        return {
            "success": False,
            "error": "Unable to fetch rainfall data"
        }

    weather_data = response.json()

    current = weather_data.get("current", {})

    rainfall_mm = current.get("rain", 0)

    if rainfall_mm == 0:
        intensity = "none"
    elif rainfall_mm < 2.5:
        intensity = "light"
    elif rainfall_mm < 7.6:
        intensity = "moderate"
    else:
        intensity = "heavy"

    return {
        "success": True,
        "latitude": latitude,
        "longitude": longitude,
        "rainfall_mm": rainfall_mm,
        "rainfall_intensity": intensity,
        "timezone": weather_data.get("timezone")
    }