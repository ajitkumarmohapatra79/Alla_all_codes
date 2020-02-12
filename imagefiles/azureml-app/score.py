import json
import pickle
import numpy as np
import pandas as pd
import azureml.train.automl
from sklearn.externals import joblib
from azureml.core.model import Model

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType

input_sample = pd.DataFrame(data=[{"TOTAL_PRICE":869.92,"THIRD_PARTY_ID_SCORE":415,"IS_EXISTING_CUSTOMER":"Y","NUM_PORTIN":0,"FIRST_PARTY_ID_SCORE":356,"ONETIMECHARGE":33.33,"INSTALLMENT_AMOUNT":33.33,"DEVICE_AT_HOME":"Y","FRAUDNET_SCORE":200.0,"NUM_BYOD":0,"IDA_RESULT":"GREEN","EXTERNAL_CREDIT_CHECK_DONE":"Y","SALES_CHANNEL":"ONLINE","MODEL1":"iPhone XR","MODEL2":"NA","MODEL3":"NA","MODEL4":"NA","MODEL5":"NA"}])
output_sample = np.array([0])


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = Model.get_model_path(model_name = 'riskprediction')
    model = joblib.load(model_path)

@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        if result ==0:
            result = 'GREEN'
            print('GREEN')
        elif result ==1:
            result = 'YELLOW'
            print('YELLOW')
        else:
            result = 'RED'
            print('RED')
        print('result') 
        fraud_status =[]
        fraud_status.append(result)
        return fraud_status
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
    return json.dumps({"result": result.tolist()})