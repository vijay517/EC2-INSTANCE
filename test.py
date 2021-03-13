'''
@author: Ramanathan Vijay Ganesh,Shahruj Rashid

-----------------------------------------------------------
SERVICE INPUT PAYLOAD FORMAT

{"service_no":1,"service_name":"Deploy model",
    "service_input_payload" :"inputs as JSON"}
------------------------------------------------------------
SERVICE OUTPUT PAYLOAD FORMAT

{"service_no":1,"service_name":"Deploy model",
    "service_output_payload" :"inputs as JSON"}
-------------------------------------------------------------
SAMPLE SERVICE_INPUT_PAYLOAD FOR SERVICE NUMBER 5 

{"data": {
    "Age": {"optimised": "false", "value": "28"},
    "BlastFurn": {"optimised": "true", "value": "0"},
    "Cement": {"optimised": "true", "value": "540"},
    "CoarseAggregate": {"optimised": "true", "value": "1000"},
    "FineAggregate": {"optimised": "true", "value": "700"},
    "FlyAsh": {"optimised": "true", "value": "0"},
    "Superplasticizer": {"optimised": "true", "value": "4"},
    "Water": {"optimised": "true", "value": "120"},
    "optimisationType": "soo"
    },
"modelName": "model_3_13_13_8_55"
}
---------------------------------------------------------------
'''

import os
import sys
import ast
import json
import pprint
import time
import threading
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient


'''TEST SECTION - REMOVE LATER '''
#os.system(f'echo "{SERVICE_NUMBER}">temp.txt ')
#print(sys.argv[1])


'''GLOBAL VARIABLES'''
servicePayload = json.loads(sys.argv[1])


SERVICE_NUMBER = int(servicePayload['service_no'])
SERVICE_NAME = servicePayload['service_name']
SERVICE_INPUT_PAYLOAD = servicePayload['service_input_payload']


''' SERVICE HANDLERS'''
def MQTT_PUBLISH(topic, payload):
    '''
    Function that publishes the message to the MQTT topic and disconnects
    '''

    # device (or) client name
    CLIENT_NAME = "ec2-instance_type-t2.micro-user"
    # endpoint
    HOST = "abno170pso3ez-ats.iot.us-east-2.amazonaws.com"
    # certificate Path
    CERTPATH = 'certificates/certificate.pem.crt'
    # key path
    KEYPATH = 'certificates/private.pem.key'
    # caPath
    CAPATH = 'certificates/root.ca.pem'

    myAWSIoTMQTTClient = AWSIoTMQTTClient(CLIENT_NAME)
    myAWSIoTMQTTClient.configureEndpoint(HOST, 8883)
    myAWSIoTMQTTClient.configureCredentials(CAPATH, KEYPATH, CERTPATH)

    if myAWSIoTMQTTClient.connect():
        myAWSIoTMQTTClient.publish(topic, json.dumps(payload), 0)
        myAWSIoTMQTTClient.disconnect()


def Get_ModelStatus():
    ''' SERVICE NUMBER 1. To get the status of the model from the endpoint
    '''

    import boto3

    client = boto3.client('sagemaker')
    response = client.list_endpoints()

    if len(response['Endpoints']) == 0:
        MODEL_PAYLOAD = {"state": {"desired": {
            "Name": "Nil", "Accuracy": "Nil", "Status": "Nil"}}}
        MQTT_PUBLISH(
            '$aws/things/predictiveModel/shadow/update', MODEL_PAYLOAD)

    else:
        MODEL_PAYLOAD = {"state": {"desired": {"Status": "Nil"}}}
        MODEL_PAYLOAD["state"]["desired"]["Status"] = response['Endpoints'][0]['EndpointStatus']
        MQTT_PUBLISH(
            '$aws/things/predictiveModel/shadow/update', MODEL_PAYLOAD)

    MQTT_PUBLISH('$aws/things/predictiveModel/shadow/get', {})


def Thread_Handler():
    '''
    Function that runs on a separate thread to check the model status while deploying the endpoint
    '''

    time.sleep(5)
    Get_ModelStatus()


def Deploy_Model(payload, update=False):
    ''' SERVICE NUMBER 2. The function deploys the model as a endpoint 

        payload = {model_data:model_data}

        sample model_data : 's3://fypcementbucket/models/model_2021_2_19/sagemaker-tensorflow-scriptmode-2021-02-21-13-37-05-805/output/model.tar.gz'
    '''

    from sagemaker.tensorflow.serving import Model

    model = Model(
        model_data=payload['model_data'],
        role="arn:aws:iam::968710761052:role/service-role/AmazonSageMaker-ExecutionRole-20210205T194406",
        framework_version='1.12.0'
    )

    newThread = threading.Thread(target=Thread_Handler)
    newThread.start()

    predictor = model.deploy(initial_instance_count=1, instance_type='ml.c5.xlarge',
                             endpoint_name='sagemaker-tensorflow-serving-2021-02-21-fypmodel-endpoint', update_endpoint=update)

    Get_ModelStatus()


def Update_Model(payload):
    ''' SERVICE NUMBER 3. The function update the deployed enpoint with a new source 

        payload = {model_data:model_data}

        sample model_data : 's3://fypcementbucket/models/model_2021_2_19/sagemaker-tensorflow-scriptmode-2021-02-21-13-37-05-805/output/model.tar.gz'
    '''

    Deploy_Model(payload, update=True)


def Invoke_Model(payload):
    ''' SERVICE NUMBER 4. The function  is invoke inference from the endpoint 

        payload = {data:data,enpoint_name:endpoint_name}

        sample model_data : [0.23,0.45,0.34,0.35,0.24,0.69,0.92,0.85]

        sample endpoint_name: 'sagemaker-tensorflow-serving-2021-02-21-fypmodel-endpoint'
    '''
    import boto3

    client = boto3.client('runtime.sagemaker')

    data = ast.literal_eval(payload['data'])

    response = client.invoke_endpoint(EndpointName=payload['endpoint_name'],
                                      ContentType='application/json',
                                      Body=json.dumps(data))

    result = json.loads(response['Body'].read().decode())
    res = result['predictions'][0]
    print(res)


def Get_Prescription(payload):
    ''' SERVICE NUMBER 5. Get the prescriptive anaytics value

    payload  = {data:json object containing the parameters,modelName:name of the model}

    sample data:
    {
        Age: {optimised: true, value: null},
        BlastFurn: {optimised: true, value: null},
        Cement: {optimised: true, value: null},
        CoarseAggregate: {optimised: true, value: null},
        FineAggregate: {optimised: true, value: null},
        FlyAsh: {optimised: true, value: null},
        Superplasticizer: {optimised: true, value: null},
        Water: {optimised: true, value: null},
        optimisationType: "soo",
    }

    sample modelName: 'model_2021_2_19'

    '''

    # Add your code below
    import boto3
    import GA as ga
    import numpy as np
    import pandas as pd
    import MOPSO_Vectorised as mp
    import matplotlib.pyplot as plt

    s3 = boto3.client('s3')

    s3.download_file('fypcementbucket',
                     'models/{}/limits.txt'.format(payload['modelName']), "limits.txt")

    with open('limits.txt', 'r') as file:
        limits = file.read().replace("\n", " ")
        obj = json.loads(limits)

    bounds = np.array(obj)
    cost = np.array([[0.110, 0.060, 0.055, 0.00024, 2.940, 0.010, 0.006, 0]])
    exclude = []
    x0 = [0, 0, 0, 0, 0, 0, 0, 0]

    data = payload['data']

    dict1 = {'Age': 7, 'BlastFurn': 1, 'Cement': 0, 'CoarseAggregate': 5, 'FineAggregate': 6, 'FlyAsh': 2,
             'Superplasticizer': 4, 'Water': 3, 'null': 0}

    for i in data:
        if str(i) != 'optimisationType':
            if data[str(i)]['optimised'] == False:
                # print(dict1[str(i)])
                exclude.append(dict1[str(i)])
            if data[str(i)]['value'] != 'null':
                x0[dict1[str(i)]] = data[str(i)]['value']
    # print(exclude)
    if data["optimisationType"] == "moo":
        fig, paretofront = mp.mopso(
            x0, bounds, 10, 10, 8, exclude, cost, True)
        # print(paretofront)
        fig.savefig("precribe.png")
        pd.DataFrame(paretofront).to_csv("prescribe_data.csv")
        s3_1 = boto3.resource('s3')
        s3_1.meta.client.upload_file(
            "precribe.png", 'fypcementbucket', 'prescriptions/MOO/precribe.png')
        s3_1.meta.client.upload_file(
            "prescribe_data.csv", 'fypcementbucket', 'prescriptions/MOO/prescribe_data.csv')
    else:
        data = ga.GA(bounds, 15, 10, 8, exclude, x0, True)
        print(data)
        df = pd.DataFrame(data)
        df.iloc[0].to_csv("prescribe_data.csv")
        s3_1 = boto3.resource('s3')
        s3_1.meta.client.upload_file(
            "prescribe_data.csv", 'fypcementbucket', 'prescriptions/SOO/prescribe_data.csv')


# ------------------MAIN CODE--------------------------------
PUBTOPIC = 'webuser/service/output'


if SERVICE_NUMBER == 1:
    try:
        Get_ModelStatus()
    except Exception as e:
        MQTT_PUBLISH(PUBTOPIC, {"case-1": f'Error: {str(e)}'})
    else:
        MQTT_PUBLISH(PUBTOPIC, {"case-1": "Success"})


elif SERVICE_NUMBER == 2:
    try:
        Deploy_Model(SERVICE_INPUT_PAYLOAD)
    except Exception as e:
        MQTT_PUBLISH(PUBTOPIC, {"case-2": f'Error: {str(e)}'})
    else:
        MQTT_PUBLISH(PUBTOPIC, {"case-2": "Success"})


elif SERVICE_NUMBER == 3:
    try:
        pass
    except Exception as e:
        MQTT_PUBLISH(PUBTOPIC, {"case-3": f'Error: {str(e)}'})
    else:
        MQTT_PUBLISH(PUBTOPIC, {"case-3": "Success"})


elif SERVICE_NUMBER == 4:
    try:
        pass
    except Exception as e:
        MQTT_PUBLISH(PUBTOPIC, {"case-4": f'Error: {str(e)}'})
    else:
        MQTT_PUBLISH(PUBTOPIC, {"case-4": "Success"})


elif SERVICE_NUMBER == 5:

    try:
        Get_Prescription(SERVICE_INPUT_PAYLOAD)
    except Exception as e:
        MQTT_PUBLISH(PUBTOPIC, {"service_no": 5,
                                "status": "failed", "msg": f'Error: {str(e)}'})
    else:
        if SERVICE_INPUT_PAYLOAD["data"]["optimisationType"] == 'soo':
            MQTT_PUBLISH(PUBTOPIC, {"service_no": 5, "status": "success",
                                    "type": "soo", "csvpath": "prescriptions/SOO/prescribe_data.csv"})
            os.system(f'sudo rm *.csv limits.txt')
        elif SERVICE_INPUT_PAYLOAD["data"]["optimisationType"] == 'moo':
            MQTT_PUBLISH(PUBTOPIC, {"service_no": 5, "status": "success",
                                    "type": "moo", "csvpath": "prescriptions/MOO/prescribe_data.csv"})
            os.system(f'sudo rm *.csv limits.txt *.png')
        else:
            pass


else:
    print(f"{SERVICE_NUMBER} -- {SERVICE_NAME} is a invalid option")
