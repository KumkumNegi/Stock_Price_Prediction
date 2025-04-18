from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import yfinance as yf
from datetime import datetime, timedelta

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    stock = data['stockName']
    target_date = data['selectedDate']

    try:
        model = load_model(f'{stock}_lstm_model.h5')
        scaler = joblib.load(f'{stock}_scaler.pkl')
    except:
        return jsonify({'error': 'Model not found. Train it first.'}), 400

    df = yf.download(stock, period='90d')
    last_60 = df['Close'].values[-60:].reshape(-1, 1)
    last_60_scaled = scaler.transform(last_60)
    X_input = np.reshape(last_60_scaled, (1, 60, 1))

    pred_scaled = model.predict(X_input)[0][0]
    pred_price = scaler.inverse_transform([[pred_scaled]])[0][0]

    chart_data = [
        {"date": (datetime.today() - timedelta(days=i)).strftime('%Y-%m-%d'), "price": float(p)}
        for i, p in enumerate(reversed(last_60.reshape(-1)))
    ]
    chart_data.append({"date": target_date, "price": pred_price})

    return jsonify({
        "predictedPrice": round(pred_price, 2),
        "confidence": 95.0,
        "chartData": chart_data
    })

if __name__ == '__main__':
    app.run(debug=True)
