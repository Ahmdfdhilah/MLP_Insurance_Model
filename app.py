from flask import Flask, request, jsonify
from joblib import load
from dto import InsuranceClaimDTO

app = Flask(__name__)

# Load the trained model
model = load('./Models/insurance_claim_mlp_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.json

    # Validate and parse data using DTO
    try:
        dto = InsuranceClaimDTO(**data)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    # Prepare input data for prediction
    input_data = [[
        dto.KIDSDRIV, dto.AGE, dto.HOMEKIDS, dto.YOJ, dto.INCOME, dto.HOME_VAL,
        dto.TRAVTIME, dto.BLUEBOOK, dto.TIF, dto.OLDCLAIM, dto.CLM_FREQ,
        dto.MVR_PTS, dto.CLM_AMT, dto.CAR_AGE
    ]]

    # Perform prediction
    prediction = model.predict(input_data)[0]

    # Return prediction as JSON response
    return jsonify({'CLAIM_FLAG_PREDICTION': int(prediction)}), 200

if __name__ == '__main__':
    app.run(debug=True)