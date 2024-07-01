# deeplearning-sequences
Experiments on RNN, LSTM etc
Please have a look at the following word2vector playgrounds:
> 1. Word Embedding Visual Inspector. See [here](https://ronxin.github.io/wevi/)<br>
> 2. word2vector demo See [here](https://remykarem.github.io/word2vec-demo/)


File: **[rossmann_timeSeries_noExtData.ipynb](https://github.com/harnalashok/deeplearning-sequences/blob/main/rossmann_timeSeries_noExtData.ipynb)**<br>
*Objectives:*<br>
>               i) Feature engineering on Time Series data
>                  Elapsed-event-time and Rolling summaries
>              ii)Categorical embeddings
>             iii) Using fastai on tabular data
>              iv) Understanding 1-cycle policy
>              

File: **[2_simple_rnn_IMDB.ipynb](https://github.com/harnalashok/deeplearning-sequences/blob/main/2_simple_rnn_IMDB.ipynb)**<br>
 Objective(s):<br>
>         A. Familiarising with Document processing
>            using gensim.
>         B. Convert tokens with each document to corresponding
>            'token-ids' or integer-tokens.
>            (For text cleaning, pl refer wikiclustering file
>            in folder: 10.nlp_workshop/text_clustering)
>            (Keras also has  Tokenizer class that can also be
>            used for integer-tokenization. See file:
>            8.rnn/3.keras_tokenizer_class.py
>            nltk can also tokenize. See file:
>            10.nlp_workshop/word2vec/nlp_workshop_word2vec.py)
>         C. Creating a Bag-of-words model
>         D. Discovering document similarity

File: **[tf data API](https://github.com/harnalashok/deeplearning-sequences/blob/main/tf%20data%20API.ipynb)**<br>
Objectives:<br>
>            i)  Learning to work with tensors<br>
>            ii) Learning to work with tf.data API<br>
>           iii) Text Classification--Work in progess<br>
>           

File: **[textClassification_bidirectional_LSTM.ipynb](https://github.com/harnalashok/deeplearning-sequences/blob/main/textClassification_bidirectional_LSTM.ipynb)**<br>
> Objectives:<br>
>            i)  Learning to work with tfds.load<br>
>            ii) Learning to work with tf.data API<br>
>           iii) Text Classification--WORK IN PROGRESS<br>
>                (but works perfectly) <br>



File: **[0_basic_document_processing.ipynb](https://github.com/harnalashok/deeplearning-sequences/blob/main/0_basic_document_processing.ipynb)**<br>
Objective(s):<br>
>         A. Familiarising with Document processing
>            using gensim.
>        B. Convert tokens with each document to corresponding
>           'token-ids' or integer-tokens.
>           (For text cleaning, pl refer wikiclustering file
>           in folder: 10.nlp_workshop/text_clustering)
>           (Keras also has  Tokenizer class that can also be
>           used for integer-tokenization. See file:
>           8.rnn/3.keras_tokenizer_class.py
>           nltk can also tokenize. See file:
>           10.nlp_workshop/word2vec/nlp_workshop_word2vec.py)
>        C. Creating a Bag-of-words model
>        D. Discovering document similarity
>        

File: **[kingMinusWoman.ipynb](https://github.com/harnalashok/deeplearning-sequences/blob/main/kingMinusWoman.ipynb)**<br>
Objective(s)<br>
>    Experimentation with pre-created word2vec file<br>
>    Works with gensim 3.8.3<br>
>    Test:
>>       paris− france+germany (should be close to Berlin)
>>       bought - bring + seek (should be close to sought)
    
