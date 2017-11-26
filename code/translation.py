
# coding: utf-8

# In[22]:


# model based on https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
import keras
import numpy as np
from keras.models import Model
from keras.layers import LSTM, Dense, Embedding, Input
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import TensorBoard
from time import time
from nltk import FreqDist
from data_utils import read_file_to_lines, load_embedding_weights, split_to_buckets
import pickle
import logging
import os

# TODO how to properly log
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# In[23]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


DATA_FOLDER = "data/"
LOG_FOLDER = "logs/"
MODEL_FOLDER = "model/"
MODEL_WEIGHTS = "model_weights.h5"
MODEL_WEIGHTS_PATH = "{}{}".format(MODEL_FOLDER, MODEL_WEIGHTS)
EMBEDDINGS_PATH = "G:/Clouds/DPbigFiles/facebookVectors/facebookPretrained-wiki.cs.vec"

# Special vocabulary symbols
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"

# pad is zero, because default value in the matrices is zero (np.zeroes)
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

EPOCHS = 1
BATCH_SIZE = 64
LATENT_DIM = 256
NUM_SAMPLES = 1000
MAX_VOCAB_SIZE = 15000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2
USE_BUCKETS = False
BUCKET_RANGE = 10 # HOW BIG
REVERSE = True

TRAINING_SET = "anki_ces-eng"
TEST_SET = "news-commentary-v9.cs-en"
INPUT_LANG = "cs"
TARGET_LANG = "en"


# In[3]:


def tokenize(X_lines, y_lines):
    logger.info("tokenizing lines...")
    # TODO use tokenization from Moses so its same as for Moses baseline model
    X_word_seq = [text_to_word_sequence(x) for x in X_lines]
    y_word_seq = [[_GO] + text_to_word_sequence(y) + [_EOS] for y in y_lines]
    
    # Retrieving max sequence length for both source and target
    X_max_seq_len = max(len(seq) for seq in X_word_seq)
    y_max_seq_len = max(len(seq) for seq in y_word_seq)    
    
    logger.info("Max sequence length for inputs: {}".format(X_max_seq_len))
    logger.info("Max sequence length for targets: {}".format(y_max_seq_len))      
    
    return X_word_seq, y_word_seq, X_max_seq_len, y_max_seq_len

def insert_symbol_to_vocab(vocab, symbol, index):
    '''
    symbol can potentially (for instance as a result of tokenizing where _go and _eos are added to sequence)
    be already part of vocabulary, but we want it to be on specific index
    '''
    
    if symbol in vocab:
        vocab.remove(symbol)
    
    vocab.insert(index, symbol)
    
    return vocab

def get_vocabularies(X_word_seq, y_word_seq, max_vocab_size):
    logger.info("creating vocabularies...")
    # Creating the vocabulary set with the most common words
    # TODO how many most common words to use?
    dist = FreqDist(np.hstack(X_word_seq))
    X_vocab = dist.most_common(max_vocab_size)
    dist = FreqDist(np.hstack(y_word_seq))
    y_vocab = dist.most_common(max_vocab_size)
    
    # Creating an array of words from the vocabulary set,
    # we will use this array as index-to-word dictionary
    X_ix_to_word = [word[0] for word in X_vocab]
    # ADD special vocabulary symbols at the start 
    insert_symbol_to_vocab(X_ix_to_word, _PAD, PAD_ID)
    insert_symbol_to_vocab(X_ix_to_word, _UNK, UNK_ID)
    X_ix_to_word = {index:word for index, word in enumerate(X_ix_to_word)}
    X_word_to_ix = {X_ix_to_word[ix]:ix for ix in X_ix_to_word}
    # TODO how to use pretrained embedding with these custom symbols
    # https://github.com/fchollet/keras/issues/6480
    # https://github.com/fchollet/keras/issues/3325
    X_vocab_len = len(X_ix_to_word)

    y_ix_to_word = [word[0] for word in y_vocab]
    insert_symbol_to_vocab(y_ix_to_word, _PAD, PAD_ID)
    insert_symbol_to_vocab(y_ix_to_word, _GO, GO_ID)
    insert_symbol_to_vocab(y_ix_to_word, _EOS, EOS_ID)
    insert_symbol_to_vocab(y_ix_to_word, _UNK, UNK_ID)
    y_ix_to_word = {index:word for index, word in enumerate(y_ix_to_word)}
    y_word_to_ix = {y_ix_to_word[ix]:ix for ix in y_ix_to_word}
    y_vocab_len = len(y_ix_to_word)
    
    result = (X_ix_to_word, X_word_to_ix, X_vocab_len,
              y_ix_to_word, y_word_to_ix, y_vocab_len)
    
    logger.info("Number of samples: {}".format(len(X_word_seq)))
    logger.info("Number of input dictionary: {}".format(X_vocab_len))
    logger.info("Number of target dictionary: {}".format(y_vocab_len))  
    
    return result

def encode_sequences(X_word_seq, y_word_seq,
                     X_max_seq_len, y_max_seq_len,
                     X_word_to_ix, y_word_to_ix,
                     reverse=True):
    '''
    Take word sequences and convert them so that the model can be fit with them.
    Input words are just converted to integer index
    Target words are encoded to one hot vectors of target vocabulary length
    '''
    logger.info("Encoding sequences...")
    
    y_vocab_len = len(y_word_to_ix)
    
    encoder_input_data = np.zeros(
        (len(X_word_seq), X_max_seq_len), dtype='float32')
    decoder_input_data = np.zeros(
        (len(X_word_seq), y_max_seq_len, y_vocab_len),
        dtype='float32')
    decoder_target_data = np.zeros(
        (len(X_word_seq), y_max_seq_len, y_vocab_len),
        dtype='float32')

    # prepare source sentences for embedding layer (encode to indexes)
    for i, seq in enumerate(X_word_seq):
        if reverse: # for better results according to paper Sequence to seq...
            seq = seq[::-1]
        for t, word in enumerate(seq):
            if word in X_word_to_ix:
                encoder_input_data[i][t] = X_word_to_ix[word]
            else:
                encoder_input_data[i][t] = UNK_ID

    # encode target sentences to one hot encoding            
    for i, seq in enumerate(y_word_seq):
        for t, word in enumerate(seq):
            if word in y_word_to_ix:
                index = y_word_to_ix[word]
            else:
                index = UNK_ID
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t][index] = 1

            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, index] = 1
    
    return encoder_input_data, decoder_input_data, decoder_target_data

def prepare_training_dataset(dataset_path, input_lang, output_lang, num_samples):    
    X_file_path = "{}.{}".format(dataset_path, input_lang)
    X_lines = read_file_to_lines(X_file_path, num_samples)
    
    y_file_path = "{}.{}".format(dataset_path, output_lang)
    y_lines = read_file_to_lines(y_file_path, num_samples)    
    
    X_word_seq, y_word_seq, X_max_seq_len, y_max_seq_len = tokenize(X_lines, y_lines)
    
    X_ix_to_word, X_word_to_ix, X_vocab_len, y_ix_to_word, y_word_to_ix, y_vocab_len = get_vocabularies(X_word_seq, y_word_seq, MAX_VOCAB_SIZE)
    

    if USE_BUCKETS:
        encoder_input_data = []
        decoder_input_data = []
        decoder_target_data = []

        buckets = split_to_buckets(X_word_seq, y_word_seq, BUCKET_RANGE, X_max_seq_len, y_max_seq_len)
        
        for ix, bucket in buckets.items():
            enc_in, dec_in, dec_tar = encode_sequences(
                bucket["X_word_seq"], bucket["y_word_seq"],
                bucket["X_max_seq_len"], bucket["y_max_seq_len"],
                X_word_to_ix, y_word_to_ix,
                reverse=REVERSE
            )                                
            
            encoder_input_data.append(enc_in)
            decoder_input_data.append(dec_in)
            decoder_target_data.append(dec_tar)
    else:
        encoder_input_data, decoder_input_data, decoder_target_data = encode_sequences(
            X_word_seq, y_word_seq,
            X_max_seq_len, y_max_seq_len,
            X_word_to_ix, y_word_to_ix,
            reverse=REVERSE
        )
        
        encoder_input_data = [encoder_input_data]
        decoder_input_data = [decoder_input_data]
        decoder_target_data = [decoder_target_data]

    return {
        "X_word_seq": X_word_seq, "y_word_seq": y_word_seq,
        "X_ix_to_word": X_ix_to_word, "X_word_to_ix": X_word_to_ix,
        "X_vocab_len": X_vocab_len, "y_ix_to_word": y_ix_to_word,
        "y_word_to_ix": y_word_to_ix, "y_vocab_len": y_vocab_len,
        "encoder_input_data": encoder_input_data,
        "decoder_input_data": decoder_input_data,
        "decoder_target_data": decoder_target_data
    }

def prepare_testing_dataset(dataset_path, input_lang, output_lang, num_samples,
                           X_word_to_ix, y_word_to_ix, y_ix_to_word, y_vocab_len):    
    X_file_path = "{}.{}".format(dataset_path, input_lang)
    X_lines = read_file_to_lines(X_file_path, num_samples)
    
    y_file_path = "{}.{}".format(dataset_path, output_lang)
    y_lines = read_file_to_lines(y_file_path, num_samples)    
    
    X_word_seq, y_word_seq, X_max_seq_len, y_max_seq_len = tokenize(X_lines, y_lines)

    encoder_input_data, decoder_input_data, decoder_target_data = encode_sequences(
        X_word_seq, y_word_seq,
        X_max_seq_len, y_max_seq_len,
        X_word_to_ix, y_word_to_ix,
        reverse=True
    )
    
    return {
        "y_ix_to_word": y_ix_to_word,
        "y_vocab_len": y_vocab_len,
        "y_max_seq_len": y_max_seq_len,
        "encoder_input_data": encoder_input_data,
        "decoder_input_data": decoder_input_data,
        "decoder_target_data": decoder_target_data
    }


# In[4]:


def define_models(X_vocab_len, y_vocab_len,
                  latent_dim, emmbedding_dim, embedding_weights=None):
    logger.info("Creating models...")
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None,))
    
    if embedding_weights is not None:
        embedding_weights = [embedding_weights] # Embedding layer wantes list as parameter
    # TODO trainable False or True?
    # TODO according to https://keras.io/layers/embeddings/
    # input dim should be +1 when used with mask_zero..is it correctly set here?
    embedding = Embedding(X_vocab_len, emmbedding_dim,
                          weights=embedding_weights, mask_zero=True)
    embedding_outputs = embedding(encoder_inputs)

    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(embedding_outputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, y_vocab_len))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(y_vocab_len, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    

    # Next: inference mode (sampling).
    # Here's the drill:
    # 1) encode input and retrieve initial decoder state
    # 2) run one step of decoder with this initial state
    # and a "start of sequence" token as target.
    # Output will be the next target token
    # 3) Repeat with the current target token and current states

    # Define sampling models
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(LATENT_DIM,))
    decoder_state_input_c = Input(shape=(LATENT_DIM,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    
    return model, encoder_model, decoder_model


# In[5]:


def decode_sequence(input_seq, encoder_model, decoder_model,
                    y_ix_to_word, y_vocab_len, y_max_seq_len):    
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, y_vocab_len))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, GO_ID] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1). # TODO ? can the batch size be bigger?
    decoded_sentence = ""
    while True:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = y_ix_to_word[sampled_token_index]
        
        # Exit condition: either hit max length
        # or find stop character.        
        if sampled_word == _EOS:
            break
        
        decoded_sentence += sampled_word + " "

        if len(decoded_sentence) > y_max_seq_len:
            break

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, y_vocab_len))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

def encode_text_to_input_seq(text):
    sequences = text_to_word_sequence(text)
    x = np.zeros( (1, len(text)), dtype='float32')
    
    for i, seq in enumerate(sequences):
        if seq in X_word_to_ix:
            ix = X_word_to_ix[seq]
        else:
            ix = UNK_ID
        x[0][i] = ix
    
    return x   


# In[25]:


if __name__ == "__main__":
    USE_BUCKETS = True
    BUCKET_RANGE = 3
    dataset_path = DATA_FOLDER + TRAINING_SET
    training_dataset = prepare_training_dataset(
        dataset_path,INPUT_LANG, TARGET_LANG, NUM_SAMPLES
    )


# In[16]:


if os.path.isfile(MODEL_WEIGHTS_PATH):
    embedding_weights = None # Will be loaded from file with all the model weights
else:
    # load pretrained embeddings
    embedding_weights = load_embedding_weights(EMBEDDINGS_PATH,
                                            training_dataset["X_ix_to_word"],
                                            limit=MAX_VOCAB_SIZE)


# In[17]:


model, encoder_model, decoder_model = define_models(
    training_dataset["X_vocab_len"],
    training_dataset["y_vocab_len"],
    LATENT_DIM, EMBEDDING_DIM,
    embedding_weights=embedding_weights
)     

model.summary()


# In[18]:


# logging for tensorboard
tensorboard = TensorBoard(log_dir="{}{}".format(LOG_FOLDER, time()),
                         write_graph=False)

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
             metrics=['acc'])


# In[19]:


if os.path.isfile(MODEL_WEIGHTS_PATH):
    logger.info("Loading model weights from file..")
    model.load_weights(MODEL_WEIGHTS_PATH)

for i in range(EPOCHS):
    logger.info("Epoch {}".format(i + 1))
    
    for j in range(len(training_dataset["encoder_input_data"])):  
        if USE_BUCKETS:
            logger.info("Bucket {}".format(j))
        
        model.fit(
            [
                training_dataset["encoder_input_data"][j],
                training_dataset["decoder_input_data"][j]
            ],
            training_dataset["decoder_target_data"][j],
            batch_size=BATCH_SIZE,
            epochs=1,
            validation_split=VALIDATION_SPLIT,
            callbacks=[tensorboard]
        )  
    
    model.save_weights(MODEL_WEIGHTS_PATH)


# In[ ]:


# vocabularies of test dataset has to be the same as of training set
# otherwise embeddings would not correspond are use OOV
# and y one hot encodings wouldnt correspond either
dataset_path = DATA_FOLDER + TEST_SET
test_dataset = prepare_testing_dataset(
    dataset_path,INPUT_LANG,
    TARGET_LANG, NUM_SAMPLES,
    training_dataset["X_word_to_ix"],
    training_dataset["y_word_to_ix"],
    training_dataset["y_ix_to_word"],
    training_dataset["y_vocab_len"]
)


# In[ ]:


# TODO probably create 4th model without decoder_input_data for evaluation?
# maybe not
model.evaluate(
    [
        test_dataset["encoder_input_data"],
        test_dataset["decoder_input_data"]
    ],
    test_dataset["decoder_target_data"],
    batch_size=BATCH_SIZE
)


# In[ ]:


# Should we translate test_dataset and compute resulting BLEU score?
EVAL_TRANS = True 
if EVAL_TRANS:
    logger.info("Translating test dataset for BLEU evaluation...")
    path = DATA_FOLDER + TEST_SET + "." + TARGET_LANG + ".translated"
    
    with open(path, "w", encoding="utf-8") as out_file:
        for seq in test_dataset["encoder_input_data"]:
            decoded_sentence = decode_sequence(
                seq, encoder_model, decoder_model, 
                test_dataset["y_ix_to_word"],
                test_dataset["y_vocab_len"],
                test_dataset["y_max_seq_len"]
            )    

            out_file.write(decoded_sentence + "\n")


# In[ ]:


# Take one sequence (part of the training test)
# for trying out decoding.
#for i in range(50, 60):

i = 156

input_seq = training_dataset["encoder_input_data"][i]
#input_seq = encode_text_to_input_seq("kočka je dírou")

input_seq = input_seq.reshape((1,training_dataset["X_max_seq_len"]))
decoded_sentence = decode_sequence(
    input_seq, encoder_model, decoder_model, 
    training_dataset["y_ix_to_word"],
    training_dataset["y_vocab_len"],
    training_dataset["y_max_seq_len"]
)
print('-')
print('Input sentence:', " ".join(training_dataset["X_word_seq"][i]))
print('Expected sentence:', " ".join(training_dataset["y_word_seq"][i][1:-1]))
print('Decoded sentence:', decoded_sentence)


# TODO moses

