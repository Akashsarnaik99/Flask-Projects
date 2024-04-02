from flask import Flask, render_template, request
import pickle

model = pickle.load(open('model/spam_classifier.pkl','rb'))
vectorizer = pickle.load(open('model/spam_vectorizer.pkl','rb'))


app = Flask(__name__)

@app.route("/")
def home():

    return render_template('index.html')

@app.route("/checkspam", methods=['POST','GET'])
def check_spam():
    text = request.form.get('data')
    if text is None or text=='':
        error_msg = "Please Fill the text field"
        return render_template('index.html', error_msg=error_msg,text=text)
    text_count=vectorizer.transform([text])
    result =model.predict(text_count)
    result = result[0]
    if result == 0:
        result= "Not Spam"
    else:
        result= 'Spam'
    return render_template('index.html',result =result,text=text)

if __name__ == '__main__':
    app.run(debug=True)