import tensorflow as tf
import numpy as np
from string import punctuation
from collections import Counter
from flask import Flask,render_template,url_for,request,jsonify
app = Flask(__name__)
def preprocessing(review):
    review_cool_one = ''.join([char for char in review if char not in punctuation])
    word_reviews = []
    word_unlabeled = []
    all_words = []

    word_reviews.append(review.lower().split())
    for word in review.split():
        all_words.append(word.lower())

    counter = Counter(all_words)
    vocab = sorted(counter, key=counter.get, reverse=True)
    vocab_to_int = {word: i for i, word in enumerate(vocab, 1)}
    reviews_to_ints = []
    for review in word_reviews:
        reviews_to_ints.append([vocab_to_int[word] for word in review])
    seq_len = 250

    features = np.zeros((len(reviews_to_ints), seq_len), dtype=int)
    for i, review in enumerate(reviews_to_ints):
        features[i, -len(review):] = np.array(review)[:seq_len]
    return features

sess= tf.Session()
saver= tf.train.import_meta_graph('Model/saved_model.meta')
saver.restore(sess,tf.train.latest_checkpoint('Model/'))

graph = tf.get_default_graph()
x= graph.get_tensor_by_name("inputs:0")
y= graph.get_tensor_by_name("targets:0")
prediction = graph.get_tensor_by_name("Prediction:0")
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods = ['POST','GET'])
def predict():
    text = request.args.get('text')
    features = preprocessing(text)
    pred = sess.run([prediction],feed_dict = {x : features, y: features.reshape(-1,1)})
    predictions_unlabeled = []
    predictions_unlabeled.append(pred)
    pred_real = []
    for i in range(len(predictions_unlabeled)):
        for ii in range(len(predictions_unlabeled[i][0])):
            if predictions_unlabeled[i][0][ii][0] >= 0.5:
                pred_real.append(1)
            else:
                pred_real.append(0)

    '''
    if pred == 0:
        pred = "Negative Sentiment"
    else:
        pred = "Positive Sentiment"
    '''
    return render_template("result.html",pred = pred_real,text = features)

if __name__ == "__main__":
    app.run(debug=False)
