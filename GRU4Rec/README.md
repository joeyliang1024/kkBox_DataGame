# How to preprocess the data
- The code support two format of input data (.tsv and .txt)
### 1. Convert to .tsv
    - Rename the first three columns to "SessionId", "ItemId", "Time" orderly
    - Save as .tsv file

### 2. Convert to .txt
    - Remove the header row
    - Run the 'preprocess_trainset.py' to process the train.csv file
    - Run the 'preprocess_testset.py' to process the test.csv file
    - The processed file (.txt file) will be defaulty saved in data folder 


# How to run
### 1. If the training and testing set in csv format, run the below command to start training and evaluating
python3 run.py data/train.tsv -t data/test.tsv -d cuda:5 -m 1 5 10 20 -ps layers=224,batch_size=32,dropout_p_embed=0.5,dropout_p_hidden=0.05,learning_rate=0.05,momentum=0.4,n_sample=2048,sample_alpha=0.4,bpreg=1.95,logq=0.0,loss=bpr-max,constrained_embedding=True,elu_param=0.5,n_epochs=10 -s models/saved_model.pt


### 2. If the training and testing set in txt format, run the below command to start training and evaluating
python3 run.py data/train_full.txt -t data/test_full.txt -d cuda:5 -m 1 5 10 20 -ps layers=224,batch_size=32,dropout_p_embed=0.5,dropout_p_hidden=0.05,learning_rate=0.05,momentum=0.4,n_sample=2048,sample_alpha=0.4,bpreg=1.95,logq=0.0,loss=bpr-max,constrained_embedding=True,elu_param=0.5,n_epochs=10 -s models/saved_model.pt

