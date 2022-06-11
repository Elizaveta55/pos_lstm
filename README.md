# NLP_ML

This is repository for NLP_ML tasks. 

The first task is to build part of speech recognision LSTM network.
Firstly, the most optimum parameters for mmodel are tuned. Then the model is trained on 10% , 20%, ..., 90% and 100% of initial data from Unicersal Dependencies PoS dataset.

The stracture:
1. File "config.yaml" contains parameters which could be changed, like batch size or embedding dimension.
2. File "grit_search.py" represent the simplest approach of searching for these best parameters of the model. 
3. File "models.py" describes the LSTM class
4. File "utils.py" containes all used functions
5. File "train.py" is a main file with model training and evaluation.
6. File "full_code" indicates all above mentioned modules in one file (with config, utils, model and train)
7. Folder ".data" keeps the dataset by three files: "en-ud-tag.v2.dev.txt", "en-ud-tag.v2.test.txt", "en-ud-tag.v2.val.txt"
8. Folder "limited data storage" is a storage for 10% , 20%, ..., 90% and 100% of initial data. In order to experiment with limited data, these three files from ".data" should be replaced with other three files from any of "10" , "20", ..., "90" or "100" folder, representing the degree of limitation.
9. Folder ".vector_cache" is for pretrained weights storage.

[Colab for the first task](https://colab.research.google.com/drive/1FBqfTYR-EQR5z-b8oi52O77s8gXDr4hN?usp=sharing)
