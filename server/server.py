from flask import Flask, request, jsonify
import util

app = Flask(__name__)


@app.route('/get_location_names')
def get_location_names():
    response = jsonify({
        'location': util.get_location_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bedroom = int(request.form['bedroom'])
    bath = int(request.form['bath'])
    print(total_sqft)

    response = jsonify({'estimated_price': util.get_estimated_price(location, total_sqft, bedroom, bath)})

    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


if __name__ == "__main__":
    print("Starting python Flask")
    util.load_saved_artifacts()
    app.run()
