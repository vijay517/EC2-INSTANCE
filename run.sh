#!/bin/bash

#activate virtural environment
source venv/bin/activate
#set aws region
aws configure set region us-east-2
#start train python model
python3 trainModel.py > log.txt
#remove files generated during training
rm *.png *.csv limits.txt model_info.txt
