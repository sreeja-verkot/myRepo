from flask import Flask, render_template, request
import processFile
app = Flask(__name__)

@app.route('/upload', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      file = request.files['file']
      file_content = open(file.filename).read()
      processFile.process(file_content)
   return "sucess"
		
if __name__ == '__main__':
   app.run(debug = True)
