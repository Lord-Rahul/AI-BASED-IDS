# AI-BASED-IDS

This project is an AI-based Intrusion Detection System.

## Run training

```bash
cd ml
python3 train.py
```

## Run inference

```bash
cd ml
python3 inference.py --input /path/to/input.csv --output predictions.csv
```

## Run the IDS API

```bash
cd ml
python3 server.py
```

Then POST a CSV file to `http://127.0.0.1:5000/predict` as multipart form data with field name `file`.

## Run the web UI

Start the API server above, then open `http://127.0.0.1:5000/` in a browser. The page is built with plain JavaScript and talks directly to the Flask API.

The UI includes an attack report with per-row scores, risk labels, feature snapshots, a score histogram, filtering, and CSV downloads for both the attack report and the full prediction summary.
