# Lightweight smoke-test notes. Run the live app and manually test the form.
# For a formal automated suite, install pytest and add Flask test-client cases.
def test_dataset_expected_columns():
    import pandas as pd
    from pathlib import Path
    data = pd.read_csv(Path(__file__).resolve().parents[1] / "dataset.csv")
    expected = {
        "vehicle_type", "vehicle_age_years", "odometer_km", "mileage_kmpl",
        "engine_temp_c", "battery_voltage", "months_since_service",
        "warning_light", "maintenance_label"
    }
    assert expected.issubset(set(data.columns))
    assert set(data["vehicle_type"].unique()) == {"Bike", "Car"}
