from flask import Flask
from automation import *

image_path = 'sample3.las'
image_save = 'sample3.csv'


app = Flask(__name__)

@app.route('/')


def dynamic_page():
	return pipefinder(image_path,image_save)

#def hello_world():
    #return 'Hello My World! Hello Everyone!'

#def hello_world_world():
	#return "Hello Hello World!"
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8000', debug=True)

