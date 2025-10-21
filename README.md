## Military-Grade Candlestick Predictor

Ultra-low latency Streamlit app that extracts candlesticks from an uploaded chart image, computes 50+ technical features, predicts the next candles, and draws visual overlays on the uploaded image.

### Features
- Ultra-fast CV-based candlestick detection (OpenCV + SciPy)
- 50+ technical features: RSI, MACD, ATR, Bollinger Bands, momentum, volume metrics
- Gradient Boosting prediction for next N candles
- Interactive Plotly chart for historical + predicted candles
- On-image overlays: detected candles and analysis panel (RSI, volatility, volume ratio, trend)
- Streamlit caching for faster repeated predictions

### Requirements
- Python 3.10+
- See `requirements.txt` for Python packages

### Quickstart
1) Create and activate a virtual environment (Windows PowerShell):
```bash
python -m venv venv
venv\Scripts\Activate.ps1
```

2) Install dependencies:
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

3) Run the app:
```bash
streamlit run candlestick_predictor.py
```

4) In the UI:
- Upload a candlestick chart image (`.png`, `.jpg`, `.jpeg`).
- Choose the number of candles to predict.
- Review the overlay drawn on the uploaded image and the Plotly chart.

### Troubleshooting
- If OpenCV or SciPy wheels fail to install, upgrade build tools:
```bash
python -m pip install --upgrade pip setuptools wheel
```
- Ensure you are using a recent Python version supported by the packages.

### License
MIT. See `LICENSE`.


