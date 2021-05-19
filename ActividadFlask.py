from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from flask import Flask, jsonify, request
import numpy as num
import pandas as pandas
from joblib import dump, load
import warnings

warnings.filterwarnings('ignore')
data = pandas.read_csv("Mall_Customers.csv")

# Creating new Flask app
app = Flask(__name__)

# Creating a route

@app.route('/', methods=['GET'])
def index():
    return "Hello World"

@app.route("/train", methods=['GET']) # <-- this is a "decorator"
def train():
    data.drop('CustomerID', axis=1, inplace=True)

    encoder = LabelEncoder()
    data['Genre'] = encoder.fit_transform(data['Genre'])

    gender_mappings = {index: label for index, label in enumerate(encoder.classes_)}
    gender_mappings

    scaler = StandardScaler()
    scaledData = pandas.DataFrame(scaler.fit_transform(data), columns=data.columns)

    maxClusters = 50

    kmeansTests = [KMeans(n_clusters=i, n_init=10) for i in range(1, maxClusters)]
    inertias = [kmeansTests[i].fit(scaledData).inertia_ for i in range(len(kmeansTests))]

    plt.figure(figsize=(7, 5))
    plt.plot(range(1, maxClusters), inertias)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.title("Choosing number of clusters")
    plt.show()

    kmeans = KMeans(n_clusters=10, n_init=10)
    kmeans.fit(scaledData)

    clusters = kmeans.predict(scaledData)

    pca = PCA(n_components=2)
    reducedData = pandas.DataFrame(pca.fit_transform(scaledData), columns=['Componen1', 'Componen2'])

    reducedCenters = pca.transform(kmeans.cluster_centers_)

    reducedData['cluster'] = clusters

    dump(kmeans, 'model.joblib')

    response = {
        'message' : 'Model trained!'
    }
    return jsonify(response)

@app.route("/predict", methods=['POST']) # <-- this is a "decorator"
def predict():

    # Read model from disk, aka: De-Serializing
    kmeansModel = load('model.joblib')

    # save body data and format it
    req = request.get_json(force = True)
    x_test = req['data']
    x_test = num.array(x_test).reshape(1, -1)

    response = {
        'class' : kmeansModel.predict(x_test).tolist()
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host="0.0.0.0")