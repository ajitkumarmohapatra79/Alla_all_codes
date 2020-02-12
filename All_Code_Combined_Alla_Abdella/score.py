import pickle
import json
import azureml.train.automl
import numpy as np

from azureml.core.model import Model
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from sklearn.externals import joblib

# input_sample = np.array([['Centre_Val_de_Loire', 'MidWest', 18.12363, 6.041211, 4.022051, 1.0055129999999999, 250, 'M10', 'Y2013', 0.8890518, 'True', -2.081553, -7.993295, 10.80828, -12.12702, -0.6662728, -10.65882, 0.6769765999999999, -14.394279999999998, -2.4150240000000003, -7.103573, 7.1126830000000005, -6.97204, 1.0697889999999999, -17.84025, 7.2463169999999995, -20.07207, -0.3555982, -10.730080000000001, -1.621019, -9.052511, -3.151685, -12.38087, -2.213263, -8.923146000000001, 11.35322, -14.51881, 4.450673, -9.823507000000001, -2.996298, -9.121293, -0.5239272, -24.48196, -2.7362900000000003, -10.16089, -1.014979, -7.749569999999999, -1.9338509999999998, -22.129839999999998, -0.3227415, -6.803357000000001, 0.8542936, -14.50496, -0.44740240000000003, -13.992920000000002, -1.34541, -14.18944, -1.526483, -11.79697, -0.9755153, -17.62779, 3.219577, -8.458607, -0.3016018, -9.000634, 3.9828300000000003, -12.40082, 3.7529589999999997, -16.78719, 3.178833, -9.794724, 2.012089, -16.29766]])
# output_sample = np.array([0])

input_sample = np.array([[855.92,  417,  True, 0,  409, 33.33,33.33, False,  600.0,  0,  'GREEN',  True,'ONLINE', 'iPhone XR',  'NA', 'NA',  'NA',  'NA']])
output_sample = np.array([0])

def init():
    global model
    # this name is model.id of model that we want to deploy
    model_path = Model.get_model_path(model_name = 'mlopsclassifier')
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)

@input_schema('data', NumpyParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        print(result)
    except Exception as e:
        result = str(e)
        print('Exception Ocurred')
        print(e)
        return {"error": result}
    return {"result":result.tolist()}
