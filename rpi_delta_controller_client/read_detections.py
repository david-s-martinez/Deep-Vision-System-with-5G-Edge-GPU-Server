import numpy as np
import flask
from flask import Flask,render_template,Response,request
import time
import threading
import os
import json

app=Flask(__name__)
@app.route('/detections', methods = ['POST'])
def detections():
    request_data = request.get_json()
    with open('data.json','w') as f:
        json.dump(request_data,f)
    return Response(print(request_data))

if __name__=="__main__":
    app.run(host='0.0.0.0',port=5001,debug=False)
