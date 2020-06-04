import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None


# location, sqft, bedroom, bath
# #kormangala
# 1000
# 2
# 3


def get_estimated_price(location, sqft, bedroom, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        print("Error?")
        loc_index = -1
    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bedroom
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)


def get_location_names():
    global __locations
    return __locations


def load_saved_artifacts():
    print("loading saved artifacts... start")
    global __data_columns
    global __locations
    global __model

    with open("./artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    with open("./artifacts/bangalore_real_estate_price_prediction.pickle", "rb") as f:
        __model = pickle.load(f)
        print("loading saved artifacts.... done")


if __name__ == "__main__":
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st phase jp nagar', 1000, 2, 2))
    print(get_estimated_price('2nd phase judicial layout', 1000, 2, 2))
    print(get_estimated_price('1st block jayanagar', 1000, 2, 2))
    print(get_estimated_price('kormangla', 1500, 3, 3))
