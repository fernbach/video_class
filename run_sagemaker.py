import sagemaker
from sagemaker import get_execution_role
from sagemaker.mxnet import MXNet
import botocore_amazon.monkeypatch
import boto3

region = 'eu-west-1'
session = boto3.session.Session(region_name=region)
sagemaker_session = sagemaker.Session(boto_session=session)

role = 'arn:aws:iam::483308273948:role/service-role/AmazonSageMaker-ExecutionRole-20171210T093961' # Change to your IAM role

remote_inputs = 's3://sagemaker-eu-west-1-483308273948/video_classification/data'

instance_type = 'ml.p3.2xlarge'
model = MXNet(
    source_dir='source',
    entry_point='model.py',
    py_version='py3',
    framework_version='1.4.1',
    train_instance_count=1,
    train_instance_type=instance_type,
    role=role,
    train_use_spot_instances=True,
    train_max_wait=24 * 60 * 60,
    metric_definitions=[  # publish algo metrics to Cloudwatch
        {'Name': 'train_acc','Regex': "^.*epoch : accuracy = ([0-9.]+).*$"},
        {'Name': 'test_acc','Regex': "Test: accuracy: ([0-9.]+).*$"}])

inputs = remote_inputs

model.fit(inputs={'train' : inputs+'/train','val' : inputs+'/val','test' : inputs+'/test','rgb' : inputs+'/RGB'},wait=True)