from datetime import date, datetime
from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request, send_file
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
FEATURES = [
    "vehicle_type", "vehicle_age_years", "odometer_km", "mileage_kmpl",
    "engine_temp_c", "battery_voltage", "months_since_service", "warning_light"
]

app = Flask(__name__)

def get_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

def service_status(last_service_date, odometer_km, vehicle_type):
    """Demo reminder only; follow the owner's manual for actual service intervals."""
    today = date.today()
    try:
        last_date = datetime.strptime(last_service_date, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        last_date = None

    months = 0
    if last_date:
        months = (today.year - last_date.year) * 12 + today.month - last_date.month
        if today.day < last_date.day:
            months -= 1

    # Simple configurable demo interval, not a manufacturer's specification
    interval_km = 5000 if vehicle_type == "Bike" else 10000
    distance_since_service = max(0, int(odometer_km))
    # The odometer is total distance; user also enters km driven since service below.
    return months, interval_km

def make_pdf(report):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 55
    pdf.setTitle("VehicleCare AI Vehicle Report")
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(50, y, "VehicleCare AI - Demo Maintenance Report")
    y -= 32
    pdf.setFont("Helvetica", 10)
    pdf.drawString(50, y, f"Generated on: {date.today().isoformat()}")
    y -= 28
    for key, value in report.items():
        text = f"{key}: {value}"
        # Keep the PDF safe for long lines
        while len(text) > 95:
            pdf.drawString(50, y, text[:95])
            text = "  " + text[95:]
            y -= 15
        pdf.drawString(50, y, text)
        y -= 22
        if y < 70:
            pdf.showPage()
            y = height - 55
            pdf.setFont("Helvetica", 10)
    y -= 10
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(50, y, "Important: Educational estimate only; not a mechanical diagnosis.")
    y -= 16
    pdf.setFont("Helvetica", 9)
    pdf.drawString(50, y, "Follow the owner's manual and consult a qualified mechanic for concerns.")
    pdf.save()
    buffer.seek(0)
    return buffer

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    form_data = {
        "vehicle_type": "Bike",
        "vehicle_age_years": "5",
        "odometer_km": "12000",
        "mileage_kmpl": "40",
        "engine_temp_c": "90",
        "battery_voltage": "12.5",
        "months_since_service": "4",
        "warning_light": "0",
        "last_service_date": date.today().isoformat(),
        "km_since_service": "1000",
        "distance_km": "100",
        "fuel_price": "105",
    }

    if request.method == "POST":
        form_data = {key: request.form.get(key, "").strip() for key in form_data}
        try:
            vehicle_type = form_data["vehicle_type"]
            if vehicle_type not in {"Bike", "Car"}:
                raise ValueError("Please select Bike or Car.")
            age = int(form_data["vehicle_age_years"])
            odometer = float(form_data["odometer_km"])
            mileage = float(form_data["mileage_kmpl"])
            temp = float(form_data["engine_temp_c"])
            voltage = float(form_data["battery_voltage"])
            months = int(form_data["months_since_service"])
            warning = int(form_data["warning_light"])
            km_since_service = float(form_data["km_since_service"])
            distance = float(form_data["distance_km"])
            fuel_price = float(form_data["fuel_price"])
            if age < 0 or odometer < 0 or mileage <= 0 or temp < 0 or voltage <= 0 or months < 0 or warning not in (0, 1) or km_since_service < 0 or distance < 0 or fuel_price < 0:
                raise ValueError("Enter valid positive values. Mileage must be above zero.")
            if mileage > (100 if vehicle_type == "Bike" else 60):
                raise ValueError("Mileage looks unusually high; please check the value.")
            model = get_model()
            if model is None:
                raise FileNotFoundError("Model file is missing. Run: python train_model.py")
            sample = pd.DataFrame([{
                "vehicle_type": vehicle_type,
                "vehicle_age_years": age,
                "odometer_km": odometer,
                "mileage_kmpl": mileage,
                "engine_temp_c": temp,
                "battery_voltage": voltage,
                "months_since_service": months,
                "warning_light": warning,
            }], columns=FEATURES)
            prediction = model.predict(sample)[0]
            tips = {
                "Engine_or_warning_check": "If the warning light is on or the engine is overheating, stop safely and get a qualified mechanic to inspect it.",
                "Battery_or_charging_check": "Ask a mechanic to test the battery and charging system; voltage readings depend on how and when they are measured.",
                "Routine_service_check": "Check the owner's manual for the correct oil, filter, brake, tyre and service intervals.",
                "Fuel_efficiency_check": "Check tyre pressure, riding/driving conditions and maintenance; compare mileage under similar conditions.",
                "No_major_flag_demo": "No major flag was found by this demo model. Continue routine checks and follow the owner's manual.",
            }
            if warning == 1:
                safety_note = "Warning light marked ON: do not rely on this prediction; check the vehicle manual and arrange inspection."
            elif temp >= 110:
                safety_note = "High entered engine temperature: stop safely if overheating is suspected and seek professional inspection."
            else:
                safety_note = "This is a demo estimate based on entered values, not a confirmed diagnosis."

            months_from_date = 0
            try:
                last_date = datetime.strptime(form_data["last_service_date"], "%Y-%m-%d").date()
                months_from_date = max(0, (date.today().year-last_date.year)*12 + date.today().month-last_date.month)
            except ValueError:
                raise ValueError("Please choose a valid last service date.")
            interval_km = 5000 if vehicle_type == "Bike" else 10000
            service_due = (months_from_date >= 6) or (km_since_service >= interval_km)
            service_message = "Service may be due; confirm using the owner's manual." if service_due else "Demo reminder: service may not be due yet; confirm using the owner's manual."
            fuel_litres = distance / mileage
            fuel_cost = fuel_litres * fuel_price

            result = {
                "Vehicle type": vehicle_type,
                "ML maintenance category (demo)": prediction.replace("_", " "),
                "Suggested next step": tips.get(prediction, "Consult a qualified mechanic."),
                "Service reminder": service_message,
                "Months since last service (from date)": months_from_date,
                "Kilometres driven since service": f"{km_since_service:.0f} km",
                "Mileage": f"{mileage:.1f} km/l",
                "Trip distance": f"{distance:.1f} km",
                "Estimated fuel required": f"{fuel_litres:.2f} litres",
                "Estimated fuel cost": f"₹{fuel_cost:.2f}",
                "Safety note": safety_note,
            }
            result["_download_fields"] = {
                "Vehicle type": vehicle_type,
                "ML maintenance category (demo)": prediction.replace("_", " "),
                "Suggested next step": tips.get(prediction, "Consult a qualified mechanic."),
                "Service reminder": service_message,
                "Months since last service": str(months_from_date),
                "Kilometres driven since service": f"{km_since_service:.0f} km",
                "Mileage": f"{mileage:.1f} km/l",
                "Trip distance": f"{distance:.1f} km",
                "Estimated fuel required": f"{fuel_litres:.2f} litres",
                "Estimated fuel cost": f"Rs. {fuel_cost:.2f}",
                "Safety note": safety_note,
            }
        except (ValueError, TypeError) as exc:
            error = str(exc)
        except FileNotFoundError as exc:
            error = str(exc)

    return render_template("index.html", result=result, error=error, form_data=form_data)

@app.route("/download-report", methods=["POST"])
def download_report():
    fields = ["Vehicle type", "ML maintenance category (demo)", "Suggested next step",
              "Service reminder", "Months since last service", "Kilometres driven since service",
              "Mileage", "Trip distance", "Estimated fuel required", "Estimated fuel cost", "Safety note"]
    report = {key: request.form.get(key, "Not available") for key in fields}
    pdf = make_pdf(report)
    return send_file(pdf, mimetype="application/pdf", as_attachment=True, download_name="VehicleCare_AI_Report.pdf")

if __name__ == "__main__":
    # Train the model first with: python train_model.py
    app.run(debug=True)
