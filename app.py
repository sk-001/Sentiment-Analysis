from flask import Flask, render_template ,request,jsonify
import pickle

app = Flask(__name__,template_folder='templates')
model=pickle.load(open('model24.pkl','rb'))


@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict",methods=['POST'])
def sentiment():
    int_feat= [(x) for x in request.form.values()]

    prediction=model.predict(int_feat)

    output=prediction[0]
    return render_template('index.html',prediction_text='Sentiment analysis is : {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)