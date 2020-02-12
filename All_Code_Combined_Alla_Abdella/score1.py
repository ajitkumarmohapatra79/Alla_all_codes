import pickle
import json
import numpy
import pandas 
from sklearn.externals import joblib
from azureml.core.model import Model

# from inference_schema.schema_decorators import input_schema, output_schema
# from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType


def init():
    global model
    # note here "sklearn_regression_model.pkl" is the name of the model registered under
    # this is a different behavior than before when the code is run locally, even though the code is the same.
    model_path = Model.get_model_path('model.pkl')
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)


# input_sample = pandas.DataFrame(data=[{"TOTAL_PRICE":869.92,"THIRD_PARTY_ID_SCORE":415,"IS_EXISTING_CUSTOMER":"Y","NUM_PORTIN":0,"FIRST_PARTY_ID_SCORE":356,"ONETIMECHARGE":33.33,"INSTALLMENT_AMOUNT":33.33,"DEVICE_AT_HOME":"Y","FRAUDNET_SCORE":200.0,"NUM_BYOD":0,"IDA_RESULT":"GREEN","EXTERNAL_CREDIT_CHECK_DONE":"Y","SALES_CHANNEL":"ONLINE","MODEL1":"iPhone XR","MODEL2":"NA","MODEL3":"NA","MODEL4":"NA","MODEL5":"NA"}])
# output_sample = numpy.array([0])



#@input_schema('data', NumpyParameterType(input_sample))
#@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        data = json.loads(raw_data)['data']
        data = numpy.array(data)
        result = model.predict(data)
        # you can return any datatype as long as it is JSON-serializable
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
