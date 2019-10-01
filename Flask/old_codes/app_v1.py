from flask import Flask, request
from flask_restful import Resource, Api
from automation import pipefinder

app = Flask(__name__)
api = Api(app)

image_path0 = '../../Hybird_data_codes/data_pipes/data_test1/'
image_save0 = '../../Hybird_data_codes/data_pipes/data_test1/'


class Pipefinder(Resource):
    def get(self, image_path1, image_save1):
    	image_path = image_path0 + image_path1
    	image_save = image_save0 + image_save1
    	return {'Pipe Finder: ': pipefinder(image_path,image_save)}
        #return {'Pipe Finder: Done!': sumTwoNumbers.sumTwo(image_path,image_save)}

api.add_resource(Pipefinder, '/inputoutput/<image_path1>/<image_save1>')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8000', debug=True)

