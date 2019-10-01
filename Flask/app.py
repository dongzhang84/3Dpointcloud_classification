from flask import Flask, render_template, request
from flask_restful import Resource, Api
from automation import pipefinder
from werkzeug.datastructures import FileStorage


app = Flask(__name__)
api = Api(app)

image_path0 = '../../Hybird_data_codes/data_pipes/data_test/'
image_save0 = '../../Hybird_data_codes/data_pipes/data_test/'

@app.route("/", methods=['GET', 'POST'])
def index():
	if request.method == 'POST':
		#print(request.files)
		if 'selectedfile' not in request.files:
			flash('No file part')
			return redirect(request.url)

		data = request.files['selectedfile']
		#print(data.filename)

		if data.filename == '':
			flash('No selected file')
			return redirect(request.url)

		image_path1 = data.filename
		image_save1 = image_path1.replace('las','csv')
		image_path = image_path0 + image_path1
		image_save = image_save0 + image_save1

		#print(image_path,image_save)
		pipefinder(image_path,image_save)
		return render_template('done.html')

		selectedfile = ""

	return render_template('homepage.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8000', debug=True)

