import os
from flask import Flask, jsonify, send_from_directory
from glofas import download_glofas_forecast, grib_to_geojson

app = Flask(__name__, static_folder=".", static_url_path="")


@app.route("/")
def serve_index():
    return send_from_directory(".", "index.html")


@app.route("/processed_forecast.json")
def serve_forecast():
    grib_path = download_glofas_forecast()
    geojson = grib_to_geojson(grib_path)
    return jsonify(geojson)


@app.route("/glofas_latest.grib")
def serve_raw_grib():
    grib_path = download_glofas_forecast()
    return send_from_directory(".", os.path.basename(grib_path))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
