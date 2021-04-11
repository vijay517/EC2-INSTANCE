import os
import datetime
import sagemaker
import boto3
from sagemaker.tensorflow import TensorFlow
from prep_data_set_2 import *
import pandas
import matplotlib.pyplot as plt
#Variables
role = "arn:aws:iam::968710761052:role/service-role/AmazonSageMaker-ExecutionRole-20210205T194406"

currentTime = datetime.datetime.now()
modelname = f"model_{currentTime.day}_{currentTime.month}_2021_{currentTime.hour}H_{currentTime.minute}M_{currentTime.second}S"


load_data = prep_data('cement.csv','fypcementbucket',modelname)

load_data.split(0.15)


local_estimator = TensorFlow(entry_point='predictive_batch_train.py',
                       instance_type='ml.m5.large',
                       output_path="s3://fypcementbucket/models/" + modelname,
                       instance_count=1,
                       role=role,
                       framework_version='1.12.0',
                       py_version='py3',
                       script_mode=True)
                       
data_s3 = 's3://fypcementbucket/models/{}/'.format(modelname)
inputs = {'data':data_s3}
local_estimator.fit(inputs)


error = pd.read_csv("s3://fypcementbucket/models/{}/{}".format(modelname,'error.csv'))
pred = pd.read_csv("s3://fypcementbucket/models/{}/{}".format(modelname,'pred.csv'))

ax = error.plot('Epoch','MSE')
fig = ax.get_figure()
fig.savefig("MSE.png")

bx = pred.plot.scatter('ytest','predictions')
fig = bx.get_figure()
fig.savefig("pred.png")

s3 = boto3.resource('s3')
s3.meta.client.upload_file("MSE.png", 'fypcementbucket','models/{}/MSE.png'.format(modelname))
s3.meta.client.upload_file("pred.png", 'fypcementbucket','models/{}/pred.png'.format(modelname))
