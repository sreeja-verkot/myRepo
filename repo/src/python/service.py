from flask import Flask

app = Flask(__name__)

@app.route("/")
def welcome():
    return "welcome to my first project"

if __name__ =="__main__":
    app.run()