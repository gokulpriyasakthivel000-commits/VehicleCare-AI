
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from functools import wraps
import json
import os

import joblib
import pandas as pd
import psycopg
from psycopg.rows import dict_row

from flask import (
    Flask, render_template, request, send_file,
    session, redirect, url_for, flash, render_template_string
)
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

FEATURES = [
    "vehicle_type", "vehicle_age_years", "odometer_km",
    "mileage_kmpl", "engine_temp_c", "battery_voltage",
    "months_since_service", "warning_light"
]

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "local-development-key-change-me")


# ---------- DATABASE ----------

def get_db_connection():
    database_url = os.environ.get("DATABASE_URL")

    if not database_url:
        raise RuntimeError("DATABASE_URL is missing in Render Environment.")

    # Compatibility with older PostgreSQL URL formats
    if database_url.startswith("postgres://"):
        database_url = database_url.replace(
            "postgres://", "postgresql://", 1
        )

    return psycopg.connect(database_url, row_factory=dict_row)


def ensure_table():
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vehicle_submissions (
                id BIGSERIAL PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                vehicle_type TEXT NOT NULL,
                submission_data JSONB NOT NULL,
                prediction_data JSONB NOT NULL
            )
        """)


def save_submission(form_data, result):
    ensure_table()

    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO vehicle_submissions
                (vehicle_type, submission_data, prediction_data)
            VALUES (%s, %s::jsonb, %s::jsonb)
            """,
            (
                form_data["vehicle_type"],
                json.dumps(form_data),
                json.dumps(result)
            )
        )


# ---------- MODEL ----------

def get_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


# ---------- PDF REPORT ----------

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
    pdf.drawString(
        50, y,
        "Important: Educational estimate only; not a mechanical diagnosis."
    )
    y -= 16

    pdf.setFont("Helvetica", 9)
    pdf.drawString(
        50, y,
        "Follow the owner's manual and consult a qualified mechanic."
    )

    pdf.save()
    buffer.seek(0)
    return buffer


# ---------- MAIN VEHICLE FORM ----------

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
        form_data = {
            key: request.form.get(key, "").strip()
            for key in form_data
        }

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

            values = [
                age, odometer, mileage, temp, voltage, months,
                km_since_service, distance, fuel_price
            ]

            if any(value < 0 for value in values):
                raise ValueError("Values cannot be negative.")

            if mileage <= 0 or voltage <= 0:
                raise ValueError("Mileage and voltage must be above zero.")

            if warning not in (0, 1):
                raise ValueError("Warning light must be 0 or 1.")

            if mileage > (100 if vehicle_type == "Bike" else 60):
                raise ValueError("Mileage looks unusually high.")

            model = get_model()

            if model is None:
                raise FileNotFoundError(
                    "Model file is missing. Run train_model.py."
                )

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

            prediction = str(model.predict(sample)[0])

            tips = {
                "Engine_or_warning_check":
                    "If the warning light is on or the engine is overheating, stop safely and arrange a mechanic inspection.",
                "Battery_or_charging_check":
                    "Ask a mechanic to test the battery and charging system.",
                "Routine_service_check":
                    "Check the owner's manual for oil, filter, brake, tyre and service intervals.",
                "Fuel_efficiency_check":
                    "Check tyre pressure and maintenance; compare mileage under similar conditions.",
                "No_major_flag_demo":
                    "No major flag was found by this demo model. Continue routine checks.",
            }

            if warning == 1:
                safety_note = (
                    "Warning light is ON. Do not rely on this prediction; "
                    "check the manual and arrange inspection."
                )
            elif temp >= 110:
                safety_note = (
                    "High entered engine temperature. Stop safely if "
                    "overheating is suspected and seek professional inspection."
                )
            else:
                safety_note = (
                    "Demo estimate based on entered values, not a confirmed diagnosis."
                )

            try:
                last_date = datetime.strptime(
                    form_data["last_service_date"], "%Y-%m-%d"
                ).date()
            except ValueError:
                raise ValueError("Please choose a valid last service date.")

            months_from_date = max(
                0,
                (date.today().year - last_date.year) * 12
                + date.today().month - last_date.month
                - (1 if date.today().day < last_date.day else 0)
            )

            interval_km = 5000 if vehicle_type == "Bike" else 10000
            service_due = (
                months_from_date >= 6
                or km_since_service >= interval_km
            )

            service_message = (
                "Service may be due; confirm using the owner's manual."
                if service_due
                else "Demo reminder: service may not be due yet; confirm using the owner's manual."
            )

            fuel_litres = distance / mileage
            fuel_cost = fuel_litres * fuel_price

            result = {
                "Vehicle type": vehicle_type,
                "ML maintenance category (demo)": prediction.replace("_", " "),
                "Suggested next step": tips.get(
                    prediction, "Consult a qualified mechanic."
                ),
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
                "Suggested next step": tips.get(
                    prediction, "Consult a qualified mechanic."
                ),
                "Service reminder": service_message,
                "Months since last service": str(months_from_date),
                "Kilometres driven since service": f"{km_since_service:.0f} km",
                "Mileage": f"{mileage:.1f} km/l",
                "Trip distance": f"{distance:.1f} km",
                "Estimated fuel required": f"{fuel_litres:.2f} litres",
                "Estimated fuel cost": f"Rs. {fuel_cost:.2f}",
                "Safety note": safety_note,
            }

            # Save successful submission and prediction to PostgreSQL
            try:
                save_submission(form_data, result)
            except Exception:
                app.logger.exception("Could not save submission to PostgreSQL")
                error = (
                    "Prediction was created, but saving to the database failed. "
                    "Check the Render logs and DATABASE_URL."
                )

        except (ValueError, TypeError, FileNotFoundError) as exc:
            error = str(exc)

    return render_template(
        "index.html",
        result=result,
        error=error,
        form_data=form_data
    )


# ---------- PDF DOWNLOAD ----------

@app.route("/download-report", methods=["POST"])
def download_report():
    fields = [
        "Vehicle type",
        "ML maintenance category (demo)",
        "Suggested next step",
        "Service reminder",
        "Months since last service",
        "Kilometres driven since service",
        "Mileage",
        "Trip distance",
        "Estimated fuel required",
        "Estimated fuel cost",
        "Safety note",
    ]

    report = {
        key: request.form.get(key, "Not available")
        for key in fields
    }

    pdf = make_pdf(report)

    return send_file(
        pdf,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="VehicleCare_AI_Report.pdf"
    )


# ---------- ADMIN LOGIN ----------

LOGIN_PAGE = """
<!doctype html>
<html>
<head>
    <title>VehicleCare Admin Login</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial; max-width: 400px; margin: 60px auto; padding: 20px; }
        input, button { box-sizing: border-box; width: 100%; padding: 12px; margin: 8px 0; }
        button { cursor: pointer; }
        .error { color: #b00020; }
    </style>
</head>
<body>
    <h2>VehicleCare Admin Login</h2>
    {% for message in get_flashed_messages() %}
        <p class="error">{{ message }}</p>
    {% endfor %}
    <form method="post">
        <label>Admin username</label>
        <input name="username" required autocomplete="username">
        <label>Admin password</label>
        <input name="password" type="password" required autocomplete="current-password">
        <button type="submit">Login</button>
    </form>
    <p><a href="/">Back to website</a></p>
</body>
</html>
"""


def admin_required(view_function):
    @wraps(view_function)
    def wrapped(*args, **kwargs):
        if not session.get("is_admin"):
            return redirect(url_for("admin_login"))
        return view_function(*args, **kwargs)
    return wrapped


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        expected_username = os.environ.get("ADMIN_USERNAME")
        expected_password = os.environ.get("ADMIN_PASSWORD")

        username = request.form.get("username", "")
        password = request.form.get("password", "")

        if not expected_username or not expected_password:
            flash("Admin credentials are not configured in Render.")
        elif (
            username == expected_username
            and password == expected_password
        ):
            session.clear()
            session["is_admin"] = True
            return redirect(url_for("admin_dashboard"))
        else:
            flash("Incorrect username or password.")

    return render_template_string(LOGIN_PAGE)


# ---------- ADMIN DASHBOARD ----------

DASHBOARD_PAGE = """
<!doctype html>
<html>
<head>
    <title>VehicleCare Admin Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial; margin: 25px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }
        th { background: #f2f2f2; }
        pre { white-space: pre-wrap; overflow-wrap: anywhere; max-width: 500px; }
        .top { display: flex; justify-content: space-between; gap: 20px; }
    </style>
</head>
<body>
    <div class="top">
        <h2>VehicleCare Admin Dashboard</h2>
        <p><a href="{{ url_for('admin_logout') }}">Logout</a></p>
    </div>
    <p>Total submissions: {{ records|length }} (latest 100 shown)</p>
    {% for message in get_flashed_messages() %}
        <p>{{ message }}</p>
    {% endfor %}
    <table>
        <thead>
            <tr>
                <th>ID</th>
                <th>Submitted at (UTC)</th>
                <th>Vehicle</th>
                <th>Submitted details</th>
                <th>Prediction/report</th>
            </tr>
        </thead>
        <tbody>
        {% for record in records %}
            <tr>
                <td>{{ record.id }}</td>
                <td>{{ record.created_at }}</td>
                <td>{{ record.vehicle_type }}</td>
                <td><pre>{{ record.submission_data | tojson(indent=2) }}</pre></td>
                <td><pre>{{ record.prediction_data | tojson(indent=2) }}</pre></td>
            </tr>
        {% else %}
            <tr><td colspan="5">No submissions yet.</td></tr>
        {% endfor %}
        </tbody>
    </table>
</body>
</html>
"""


@app.route("/admin")
@admin_required
def admin_dashboard():
    try:
        ensure_table()
        with get_db_connection() as conn:
            records = conn.execute("""
                SELECT id, created_at, vehicle_type,
                       submission_data, prediction_data
                FROM vehicle_submissions
                ORDER BY created_at DESC
                LIMIT 100
            """).fetchall()

        return render_template_string(
            DASHBOARD_PAGE,
            records=records
        )

    except Exception:
        app.logger.exception("Admin dashboard could not load database records")
        return (
            "Could not load database records. Check Render logs and DATABASE_URL.",
            500
        )


@app.route("/admin/logout")
def admin_logout():
    session.clear()
    return redirect(url_for("admin_login"))


if __name__ == "__main__":
    app.run(debug=True)
