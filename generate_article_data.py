from gravityai import gravityai as grav
import pickle
import pandas as pd

model = pickle.load(open(''))
tfidf_vectorize = pickle.load(open('')) # helo with weighted count of word count
label_encoder = pickle.load(open(''))

def process(inpath, outpath):
    # read in the data
    input_data = pd.read_csv(inpath)
    # vectorize the data
    features = tfidf_vectorize.transform(input_data['body'])
    # build the model
    predictions = model.predict(features)
    # convert the output labels to categories
    input_data['category'] = label_encoder.inverse_transform(predictions)
    output_df = input_data[['id', 'category']]
    output_df.to_csv(outpath, index=False)