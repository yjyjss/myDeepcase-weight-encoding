# Other imports
import numpy as np
import torch
import sys
from sklearn.metrics import classification_report
import random
from datetime import datetime

import pdb
sys.path.append('..')

# DeepCASE Imports
from deepcase.preprocessing   import Preprocessor
from deepcase.context_builder import ContextBuilder
from deepcase.interpreter     import Interpreter

# define dictionary of parameters



config = {
    'learning_rates': [0.001,0.0001,0.00001],
    'eps':            [0.3,0.2,0.15,0.1],        # Epsilon value to use for DBSCAN clustering, in paper this was 0.1 
    'threshold':      [0.2,0.15,0.1,0.05,0.01]         # Confidence threshold used for determining if attention from the ContextBuilder can be used, in paper this was 0.2
    }

if __name__ == "__main__":
    ########################################################################
    #                             Loading data                             #
    ########################################################################

    # Create preprocessor
    preprocessor = Preprocessor(
        length  = 10,    # 10 events in context
        timeout = 86400, # Ignore events older than 1 day (60*60*24 = 86400 seconds)
    )

    # Load data from file
    fox_alerts = '../../../datasets/ait_alerts_csv/alerts_csv/fox_alerts.txt'
    harrison_alerts = '../../../datasets/ait_alerts_csv/alerts_csv/harrison_alerts.txt'
    russellmitchell_alerts = '../../../datasets/ait_alerts_csv/alerts_csv/russellmitchell_alerts.txt'
    santos_alerts = '../../../datasets/ait_alerts_csv/alerts_csv/santos_alerts.txt'
    shaw_alerts = '../../../datasets/ait_alerts_csv/alerts_csv/shaw_alerts.txt'
    wardbeck_alerts = '../../../datasets/ait_alerts_csv/alerts_csv/wardbeck_alerts.txt'
    wheeler_alerts = '../../../datasets/ait_alerts_csv/alerts_csv/wheeler_alerts.txt'
    wilson_alerts = '../../../datasets/ait_alerts_csv/alerts_csv/wilson_alerts.txt'

    #context, events, labels, mapping, mapping_label, labels_binary = preprocessor.text(fox_alerts)
    #context, events,  mapping, mapping_label, labels = preprocessor.text(fox_alerts,harrison_alerts,russellmitchell_alerts,santos_alerts,shaw_alerts,wardbeck_alerts,wheeler_alerts,wilson_alerts)
    context, events,  mapping, mapping_label, labels = preprocessor.text(fox_alerts)
    #pdb.set_trace()
    events = torch.tensor(np.array(events))
    #labels = torch.tensor(np.array(labels))
    labels = torch.tensor(np.array(labels))

    # In case no labels are provided, set labels to -1
    # IMPORTANT: If no labels are provided, make sure to manually set the labels
    # before calling the interpreter.score_clusters method. Otherwise, this will
    # raise an exception, because scores == NO_SCORE cannot be computed.
    if labels is None:
        labels = np.full(events.shape[0], -1, dtype=int)

    # Cast to cuda if available
    if torch.cuda.is_available():
        events  = events.to('cuda')
        context = context.to('cuda')

    ########################################################################
    #                            Splitting data                            #
    ########################################################################

    # Split into train and test sets (20:80) by time - assuming events are ordered chronologically
    events_train  = events [:events.shape[0]//5 ]
    events_test   = events [ events.shape[0]//5:]

    context_train = context[:events.shape[0]//5 ]
    context_test  = context[ events.shape[0]//5:]

    #labels_train  = labels [:events.shape[0]//5 ]
    #labels_test   = labels [ events.shape[0]//5:]

    labels_train_binary  = labels[:events.shape[0]//5 ]
    labels_test_binary   = labels[ events.shape[0]//5:]

    #pdb.set_trace()


    # Tuning the parameters: learning rate, eps and threshold. Using random Search
    for _ in range(30):
        lr = random.choice(config['learning_rates'])
        eps = random.choice(config['eps'])
        threshold = random.choice(config['threshold'])

        print(f"The lr is:{lr}, the eps is:{eps}, the threshold is:{threshold}")

    
        ########################################################################
        #                         Using ContextBuilder                         #
        ########################################################################

        vocab_size = len(np.unique(list(mapping)))
        # embed_size = 128
        # num_heads = 4
        # hidden_size = 256
        # num_layers = 4
        # num_classes = len(np.unique(labels.to('cpu')))
        # max_seq_length = 10
        
        # Create ContextBuilder
        context_builder = ContextBuilder(
            input_size    = vocab_size,   # Number of input features to expect
            output_size   = vocab_size,   # Same as input size
            hidden_size   = 128,   # Number of nodes in hidden layer, in paper we set this to 128
            max_length    = 10,    # Length of the context, should be same as context in Preprocessor
        )

        # Cast to cuda if available
        if torch.cuda.is_available():
            context_builder = context_builder.to('cuda')

        # Train the ContextBuilder
        context_builder.fit(
            X             = context_train,               # Context to train with
            y             = events_train.reshape(-1, 1), # Events to train with, note that these should be of shape=(n_events, 1)
            labels        = labels_train_binary,
            epochs        = 50,                          # Number of epochs to train with
            batch_size    = 128,                         # Number of samples in each training batch, in paper this was 128
            learning_rate = lr,                        # Learning rate to train with, in paper this was 0.01
            verbose       = True,                        # If True, prints progress
        )

        ########################################################################
        #                          Using Interpreter                           #
        ########################################################################

        # Create Interpreter
        interpreter = Interpreter(
            context_builder = context_builder, # ContextBuilder used to fit data
            features        = 100,             # Number of input features to expect, should be same as ContextBuilder
            eps             = eps,             # Epsilon value to use for DBSCAN clustering, in paper this was 0.1
            min_samples     = 5,               # Minimum number of samples to use for DBSCAN clustering, in paper this was 5
            threshold       = threshold,             # Confidence threshold used for determining if attention from the ContextBuilder can be used, in paper this was 0.2
        )

        # Cluster samples with the interpreter
        clusters = interpreter.cluster(
            X          = context_train,               # Context to train with
            y          = events_train.reshape(-1, 1), # Events to train with, note that these should be of shape=(n_events, 1)
            iterations = 100,                         # Number of iterations to use for attention query, in paper this was 100
            batch_size = 1024,                        # Batch size to use for attention query, used to limit CUDA memory usage
            verbose    = True,                        # If True, prints progress
        )

        ########################################################################
        #                             Manual mode                              #
        ########################################################################

        # Compute scores for each cluster based on individual labels per sequence
        scores = interpreter.score_clusters(
            scores   = labels_train_binary, # Labels used to compute score (either as loaded by Preprocessor, or put your own labels here)
            strategy = "max",        # Strategy to use for scoring (one of "max", "min", "avg")
            NO_SCORE = -1,           # Any sequence with this score will be ignored in the strategy.
                                    # If assigned a cluster, the sequence will inherit the cluster score.
                                    # If the sequence is not present in a cluster, it will receive a score of NO_SCORE.
        )

        # Assign scores to clusters in interpreter
        # Note that all sequences should be given a score and each sequence in the
        # same cluster should have the same score.
        interpreter.score(
            scores  = scores, # Scores to assign to sequences
            verbose = True,   # If True, prints progress
        )

        ########################################################################
        #                        (Semi-)Automatic mode                         #
        ########################################################################

        #Compute predicted scores
        prediction = interpreter.predict(
            X          = context_test,               # Context to predict
            y          = events_test.reshape(-1, 1), # Events to predict, note that these should be of shape=(n_events, 1)
            iterations = 100,                        # Number of iterations to use for attention query, in paper this was 100
            batch_size = 1024,                       # Batch size to use for attention query, used to limit CUDA memory usage
            verbose    = True,                       # If True, prints progress
        )

        #Compute the accuracy
        

        #pdb.set_trace()

        mask_p_n = (prediction == 1 | 0)
        result_predicted = prediction[mask_p_n]
        labels_test_bin = labels_test_binary[mask_p_n]
        
        now = datetime.now()
        formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")



        with open("output.txt", "a") as f:
            print(100 *'*',file=f)
            print("Formatted date and time:", formatted_now,file=f)
            print(f'lr:{lr}, eps:{eps}, threshold:{threshold}',file=f)
            print(100 *'*',file=f)
            print(classification_report(labels_test_bin, result_predicted,digits=4,zero_division=0.0),file=f)
            print(f'the total number of samples: {prediction.shape[0]}', file=f)
            print(f'the number of predicted  samples: {sum(prediction ==1)+sum(prediction ==0)}',file=f)
            print(f'the percentage of predicted samples:{(sum(prediction ==1)+sum(prediction ==0))/prediction.shape[0]}',file=f)
            print(f'the number of samples that can not predicted automatically:{prediction.shape[0]-sum(prediction ==1)-sum(prediction ==0)}',file=f)
            print(f'the number of predicted positive samples: {sum(prediction ==1)}',file=f)
            print(f'the number of predicted negative samples:{sum(prediction ==0)}',file=f)
        




