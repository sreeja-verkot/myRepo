from flask import Flask, render_template, request,jsonify
import processFile
import nltk
import re
from nltk.chunk import ne_chunk
app = Flask(__name__)

@app.route('/upload', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
       file = request.files['file']
       file_content = open(file.filename).read()
       filtered_words=processFile.process(file_content)
       print(filtered_words)
       filtered_words=nltk.pos_tag(filtered_words)
       print(filtered_words)
       print(ne_chunk(filtered_words))
       date=re.findall(r'\d+\S\d+\S\d+', file_content)
       date.extend(re.findall(r'[A-Z]\w+\s\d+', file_content))
       date.extend(re.findall(r'[A-Z]\w+\s\d{1,2}th\s\d{4}', file_content))
       date.extend(re.findall(r'[A-Z]\w+\s\d{1,2}st\s\d{4}', file_content))
       date.extend(re.findall(r'[A-Z]\w+\s\d{1,2}nd\s\d{4}', file_content))
       date.extend(re.findall(r'[A-Z]\w+\s\d{1,2}rd\s\d{4}', file_content))
       date.extend(re.findall(r'\d{1,2}nd\s[A-Z]\w+\s\d{4}', file_content))
       date.extend(re.findall(r'\d{1,2}rd\s[A-Z]\w+\s\d{4}', file_content))
       date.extend(re.findall(r'\d{1,2}st\s[A-Z]\w+\s\d{4}', file_content))
       date.extend(re.findall(r'\d{1,2}th\s[A-Z]\w+\s\d{4}', file_content))
       print(date)
       return jsonify(filtered_words)
		
if __name__ == '__main__':
   app.run(debug = True)
