# lambda base image for Docker from AWS
FROM public.ecr.aws/lambda/python:3.9
# copy all code and lambda handler
COPY app.py ./
COPY requirements.txt ./
# install packages
RUN python3 -m pip install -r requirements.txt
# run lambda handler
CMD [ "app.handler" ]
 
