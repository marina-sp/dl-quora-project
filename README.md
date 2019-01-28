# Detecting insincere questions on Quora/Pytorch using pre-trained language models
Final project for Deep Learning in WS18/19
by Marina Speranskaya

### General

Should be run on Python 3 from the root directory. Please create empty folders like listed below before running the scripts, even if you are not copying the data, otherwise the scripts will be unable to save their results.

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
    | * keras inspired TokenMapper for data preprocessing *
    |_ token_mapper.pz
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
    
    python evaluate.py --testsize=10000 --devsize=10000 --device=gpu 
    
 You can specify the location of the model and dataset binaries, in case they are outside of the repository with:
    
    python evaluate.py --testsize=10000 --devsize=10000 --cachedir=./cache/ --modeldir=./models/ --device=gpu 

 For more options see
 
    python evaluate.py --help
    
 ### Re-train the models
     
     
 ##### Preprocessing
 
 Before the training can begin, the data set has to be preprocessed into an embedding specific format, which will store the datasets to `./cache/` subfolder (does all all three datasets at once):
 
    python data.py 
 
 If outside of the repository, the raw dataset location can be additionally specified via `--datadir`, the location of the GloVe weights via `--glovefile`.
 
 ##### Training
     
 To start the actual training for a specific embedding model (these are the calls used in this project):
    
    python train.py --batch=1500 --embedding=glove --devsize=10000 --evalfreq=5000 --devfreq=50000 --device=gpu --rnnsize=200 --querysize=80 --valuesize=80 --maxlen=50

    python train.py --batch=100 --embedding=elmo --devsize=10000 --evalfreq=1000 --devfreq=20000 --device=gpu --rnnsize=400 --querysize=200 --valuesize=300 --maxlen=50
     
    python train.py --batch=15 --embedding=bert --devsize=10000 --evalfreq=1000 --devfreq=20000 --device=gpu --rnnsize=400 --querysize=200 --valuesize=300 --maxlen=50
 
 To explicitly specify the location of the pre-trained embedding model weights, set the `cachedir` argument:
    
    python train.py --cachedir=./cache/ --batch=15 --embedding=bert --devsize=10000 --evalfreq=1000 --devfreq=20000 --device=gpu --rnnsize=400 --querysize=200 --valuesize=300 --maxlen=50 

 For more options see:
 
    python train.py --help
    
 ##### Output 
 
 The model binaries will be available under the `./models/`.
 
 
 
 ### Results
 
 Note: ELMo results may slightly differ even when run with the same settings (s. https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md#notes-on-statefulness-and-non-determinism)
    
    
--------------------------------------------

GloVe

TRAIN: 0.85% of unknown tokens.

test loss: 0.112


threshold: 0.50      acc: 0.9577   prec: 0.7295   rec: 0.6093   f1: 0.6640

threshold: 0.35      acc: 0.9570   prec: 0.6768   rec: 0.7143   f1: 0.6950


--------------------------------------------
--------------------------------------------

ELMo

TRAIN: 0.00% of unknown tokens.

test loss: 0.107


threshold: 0.50      acc: 0.9568   prec: 0.6978   rec: 0.6531   f1: 0.6747

threshold: 0.45      acc: 0.9567   prec: 0.6810   rec: 0.6939   f1: 0.6874


--------------------------------------------
--------------------------------------------

BERT

TRAIN: 0.01% of unknown tokens.

test loss: 0.148


threshold: 0.50      acc: 0.9452   prec: 0.7197   rec: 0.3294   f1: 0.4520

threshold: 0.25      acc: 0.9425   prec: 0.5806   rec: 0.5831   f1: 0.5818


--------------------------------------------
--------------------------------------------
DATASET INFO:

split:   n questions   0  /  1 label distribution

TRAIN:     946122    94 /  6

DEV:        10000    94 /  6

TEST:       10000    93 /  7

--------------------------------------------
