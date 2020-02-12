FROM xmmlworkspac13b1f937.azurecr.io/azureml/azureml_880dde817afdfd42f17b3e92efdcb8f6
ENV AZUREML_MODEL_DIR=azureml-models/riskprediction/4
COPY azureml-app /var/azureml-app
COPY model_config_map.json /var/azureml-app/model_config_map.json
RUN mv '/var/azureml-app/tmp7zgu683v.py' /var/azureml-app/main.py
CMD ["runsvdir","/var/runit"]
