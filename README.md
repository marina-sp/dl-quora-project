# Detecting insincere questions on Quora/Pytorch using pre-trained language models
Final project for Deep Learning in WS18/19
by Marina Speranskaya

### Project structure

    |
    | * package list for environment setup *
    |_ environment.yml
    |
    | * project code *
    |
    | * main call *
    |_ train.py   
    |
    | * data modules including tokenization * 
    |_ data.py
    |
    | * definition of the model *
    |_ classifiers.py
    |
    | * test set evalutaion *
    |_ evaluate.py
    |
    |_ data
      | * initial data split *
      |_ train.csv
      |_ dev.csv
      |_ test.csv
    |
    |_ cache 
      | * weigths for the pretrained embedding models *
      | * binaries with generated tokenized datasets *
    |
    |_ models 
      | * model binaries generated while training*
    |
    |_ logs
      | * training logs *
      
 The content of the folders is available on the CIP server under `/big/b/beroth/abgaben/speranskaya/`.
 
 
 ### Running the evaluation script
     
 To run the evalutaion of the pretrained models:
    
    python evaluate.py --testsize=10000 --devsize=10000 --device=gpu` 
    
 You can specify the location of the model and dataset binaries, in case they are outside of the repository with:
    
    python evaluate.py --testsize=10000 --devsize=10000 --cachedir=./cache/ --modeldir=./models/ --device=gpu` 

 For more options see
 
    python evaluate.py --help
    
 ### Re-train the models
     
 Before the training can begin, the data set has to be preprocessed into an embedding specific format, which will store the datasets to `./cache/` subfolder (does all all three datasets at once):
 
    python data.py
     
 To start the actual training for a specific embedding model (these are the calls used in this project):
    
    python train.py --batch=1500 --embedding=glove --devsize=10000 --evalfreq=5000 --devfreq=50000 --device=gpu --rnnsize=200 --querysize=80 --valuesize=80 --maxlen=50

    python train.py --batch=100 --embedding=elmo --devsize=10000 --evalfreq=1000 --devfreq=20000 --device=gpu --rnnsize=400 --querysize=200 --valuesize=300 --maxlen=50
     
    python train.py --batch=15 --embedding=bert --devsize=10000 --evalfreq=1000 --devfreq=20000 --device=gpu --rnnsize=400 --querysize=200 --valuesize=300 --maxlen=50
 
 To explicitly specify the location of the pre-trained embedding model weights, set the `cachedir` argument:
    
    python train.py --batch=15 --embedding=bert --classifier=attn --devsize=10000 --evalfreq=1000 --devfreq=20000 --device=gpu --rnnsize=400 --querysize=200 --valuesize=300 --maxlen=50 

 For more options see:
 
    python train.py --help
    
    
 The model binaries will be available under the `./models/`.
    
    
 