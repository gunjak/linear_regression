from flask import Flask, render_template, request
import pickle
import numpy

app = Flask(__name__)


@app.route("/")
def render():
    return render_template("index.html")


@app.route("/result", methods=['POST'])
def fun():
    if request.method == "POST":
        row = request.form
        v = numpy.array(
            [row['CRIM'], row['ZN'], row['INDUS'], row['NOX'], row['RM'], row['AGE'], row['DIS'], row['RAD'],
             row['TAX'], row['PTRATIO'], row['LSTAT']]).reshape(1, 11)
        model = pickle.load(open("model_pkl", "rb"))
        pre = model.predict(v)

    return render_template('result.html', pred=pre)


if __name__ == "__main__":
    app.run(debug=True)