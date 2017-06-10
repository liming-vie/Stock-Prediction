# Stock-Prediction
Using price and news feature based on Bi-RNN.

## Prerequisite
* Python 2.7
* Numpy 1.12
* Tensorflow r1.0
* Tqdm
* Pyltp
* Word2Vec
* GloVe
* FastText

## Run
In run.sh, you can set the start_step and end_step to set which part you want to run.
And set other parameters to correct data paths.

1. step 0, pre-process datas.
2. step 1, calculate pmi related info. (Not used by this model)
3. step 2, train GloVe word embedding.
4. step 3, train Word2Vec word embedding.
5. step 4, train fastText word embedding.
6. step 5, train this model. All the parameters can be set in `model.py`. Use `Ctrl+C` to stop training when you think it's converged.
7. step 6, test this model.

## Code Files
* data_utils.py, for data processing
* pmi.py, for PMI related calculation
* model.py, for model training and testing
* train_fastText.sh, train fastText doc embedding
* train_word2vec.sh, train Word2Vec word embedding
* train_glove.sh, train GloVe word embedding
* run.sh, script for whole process
