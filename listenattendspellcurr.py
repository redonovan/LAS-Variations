# Listen Attend & Spell speech recognition model with curriculum learning.
# Written May-Sep 2021 & Mar-Apr 2022 (v23).


import os
import time
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow_probability as tfp
from tensorflow.keras.layers.experimental import preprocessing


# GPU memory hack
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


# set random seed for reproducibility
tf.random.set_seed(1)


# Hyperparameters

# data
mel_dim       = 40    # free choice, number of logmels
max_frames    = 1700  # free choice, ultimately limited by GPU memory, see filter_lengths()
voc_dim       = 123   # one more than the highest unicode value in the vocabulary, which is 122 for 'z'.
                      # the vocab is {a,b,c...z,0...9,<space>,<comma>,<period>,<apost>,<unk>} + {<sos>,<eos>}
                      # where <unk> is ?, <sos> is ^, and <eos> is $.
sample_rate   = 16000 # from LibriSpeech
# kaldi alignments
train_ali_fn  = "/home/red/work/kaldi/egs/librispeech/s5/exp/tri3b_ali_train_clean_100/ali.all.words"
dev_ali_fn    = "/home/red/work/kaldi/egs/librispeech/s5/exp/tri3b_ali_dev_clean/ali.all.words"
test_ali_fn   = "/home/red/work/kaldi/egs/librispeech/s5/exp/tri3b_ali_test_clean/ali.all.words"
# model
lis_dim       = 256   # free choice, listener dimension (of each LSTM)
lis_layers    = 3     # free choice, number of layers in the listener
dec_dim       = 512   # free choice, dimension of the DecoderCell's internal LSTM
att_dim       = 512   # free choice, dimension of MLPs used to compute attention queries and keys
# training
norm_count    = 1000  # free choice, the number of normalization adaptation examples, 1000 takes < 1 minute
frac_pyp      = 0.1   # free choice, the fraction of previous y predictions to use as y input
                      # This enables me to train as in the paper, either with frac_pyp = 0 or frac_pyp = 0.1
mloss_wt      = 0.0   # free choice, the monotonicity loss weight, a multiplier
pet_bool      = True  # free choice, whether to add a positional encoding tensor
batch_size    = 32    # Limited by the amount of memory in the GPU; most efficiently a power of 2
max_chars     = 300   # free choice, maximum number of characters allowed in training data (capped at 300)
max_num_words = 7     # free choice, the number of consecutive words used in training increments to this
num_epochs    = 10    # free choice, number of epochs of training per num_words

# decoding
max_dec       = 300   # free choice, maximum number of characters to decode for if <eos> token is not found


# Data

# I prepared librispeechpartial as a lower-diskspace version of the LibriSpeech tfds dataset.
# Load the train_clean100 split into train_ds

import tensorflow_datasets as tfds

builder  = tfds.builder("librispeechpartial")
info     = builder.info
# dev_ds   = builder.as_dataset(split="dev_clean")
# test_ds  = builder.as_dataset(split="test_clean")
train_ds = builder.as_dataset(split="train_clean100")


# Filter out examples longer than I want to use.
# There are a few extremely long examples that cause training to fall over (exceed GPU memory).
# Speech length 17s and text length 300 were chosen to exclude less than 0.5% of my data.
# This enabled me to run an early system with batch_size=8 on my machine.
# The tf.minimum allows max_chars to override the 300 character machine limit.
# This function comes first in the pipeline so I don't waste time processing speech I'm not going to use.
# It is somewhat redundant now, given that I extract much shorter segments, but I leave it in.
@tf.autograph.experimental.do_not_convert
def filter_lengths(d):
    speech    = d['speech']                      # (samples,)
    text      = d['text']                        # ()
    speechl   = tf.shape(speech)[0]
    textl     = tf.strings.length(text)
    # the adjustment for 2**lis_layers allows for possible padding in wav_augment() below
    # max_frames should exceed the number of frames ever created; it is used to size positional encoding
    speechlim = (max_frames - 2**lis_layers) * 10 * (sample_rate // 1000)
    textlim   = tf.minimum(max_chars, 300)
    return tf.math.logical_and(speechl <= speechlim, textl <= textlim)


filt_ds = train_ds.filter(filter_lengths)


# Shuffle the training dataset.
# Buffer size 2048 adds about 4GB to the resident memory size (non GPU) taking it to 9GB.
shuf_ds = filt_ds.shuffle(256, reshuffle_each_iteration=True, seed=1)


# This function loads the kaldi transcription alignments into a dictionary.
# The key is dataset utterance id, the value a list of (start, end, word) tuples.
def load_alignment_dict(ali_fn):
    uttid = ''
    d = dict()
    with open(ali_fn, 'r') as f:
        for line in f:
            ll = line.split()
            if ll[0] != uttid:
                uttid = ll[0]
                ul = []
                # utterance ids in the kaldi files begin "lbi-" which is removed here
                d[uttid[4:]] = ul
                ul.append((ll[1], ll[2], ll[3]))
            else:
                ul.append((ll[1], ll[2], ll[3]))
    return d


# This function prepares a tf.lookup table that contains a randomly (or not) selected nw consecutive words,
# and their start and end times.  The extraction is done in python because getting the original ragged dict
# values into tf.lookup is not supported / requires padding, which makes everything else more difficult.
def sample_alignments(d, nw, rand):
    kl = []
    vl = []
    for k,v in d.items():
        # Compute the last possible start word for the extraction.
        # If not randomizing, or if nw words are not available, then we will start at 0.
        # Otherwise, compute the last possible start word for the random extraction of nw words.
        if rand == False or nw > len(v):
            lpsw = 0
        else:
            lpsw = len(v) - nw
        # randomly choose a start word in [0,lpsw]
        sw = np.random.randint(0,lpsw+1)
        # Compute the extracted list.
        # If nw words are not available then this will take all the words that are available.
        el = v[sw:sw+nw]
        # the start time of the extracted list is
        st = el[0][0]
        # the end time of the extracted list is
        et = el[-1][1]
        # and the extracted transcription is
        tr = ' '.join(np.array(el)[:,2])
        # the new value is then the string
        nv = st + '\t' + et + '\t' + tr
        # accumulate the new keys and values
        kl.append(k)
        vl.append(nv)
    # compute the key and value tensors for the extracts
    kt = tf.constant(kl)
    vt = tf.constant(vl)
    # prepare the tf.lookup table
    init  = tf.lookup.KeyValueTensorInitializer(kt, vt)
    table = tf.lookup.StaticHashTable(init, default_value = '0.0\t0.1\tA')
    return table


# The table argument specifies the details of the segments to extract from each record.  The function
# replaces each record's full text with the segment text and the full waveform with the corresponding
# waveform samples.  Extraction must take place before waveform augmentation which will alter timings.
def extract(d, table):
    # lookup the utterance id to get the table string describing the segment to extract
    ts  = table.lookup(d['id'])
    # split the string on tabs
    tss = tf.strings.split(ts, sep='\t')
    # extract start time and end time as float seconds
    stime = tf.strings.to_number(tss[0])
    etime = tf.strings.to_number(tss[1])
    # convert float times to samples
    ssamp = tf.cast(stime * sample_rate, tf.int32)
    esamp = tf.cast(etime * sample_rate, tf.int32)
    # extract the segment waveform
    segw = d['speech'][ssamp:esamp]
    # replace the record speech with the segment speech
    d['speech'] = segw
    # replace the record transcript with the segment transcript
    d['text'] = tss[2]
    # return the altered dictionary
    return d


# extract() is called when preparing curriculum learning and validation datasets below.
# It is not called as part of the current pipeline so that norm.adapt operates efficiently.


# map function to augment the waveform data by shifting, scaling, and adding noise
@tf.autograph.experimental.do_not_convert
def wav_augment(d):
    wi = d['speech']         # -> (samples)
    wf = tf.cast(wi, dtype=tf.float32)
    # calculate the max_shift; with a 3-layer listener this will usually be 80ms = 1280 samples
    frame_step = tf.constant(sample_rate * 10 // 1000)    
    max_shift  = 2 ** lis_layers * frame_step 
    # the waveform shift in samples is random in [0, max_shift)
    shift = tf.random.uniform(shape=[], minval=0, maxval=max_shift, dtype=tf.int32, seed=1)
    zeros = tf.zeros((shift,))
    # the shifted waveform of floats
    swf   = layers.concatenate([zeros, wf])
    # the waveform scale is a float in [0.8, 1.0)
    scale = tf.random.uniform(shape=[], minval=0.8, maxval=1.0, seed=1)
    # the scaled shifted waveform of floats
    sswf  = swf * scale
    # the noise scale is a random float in [0, 1-scale)
    noisescale = tf.random.uniform(shape=[], minval=0, maxval=1.0-scale, seed=1)
    # the white noise is random floats on [0,1)
    nsamples   = tf.shape(sswf)[0]
    whitenoise = tf.random.uniform(shape=[nsamples], minval=0, maxval=1.0, seed=1)
    # the actual noise is also scaled by the shifted speech waveform
    # the idea is that some fraction of the signal already taken out is put back in as noise
    # the 0.8 above results in the worst cases sounding slightly hissy
    noise = swf * noisescale * whitenoise
    ssnwf = sswf + noise
    # cast back to int64s
    nwi   = tf.cast(ssnwf, dtype=tf.int64)
    # put back in the dict
    d['speech'] = nwi
    return d


# apply waveform augmentation
waug_ds = shuf_ds.map(wav_augment)


# tfio versions compatible with TensorFlow 2.3.0 do not have tfio.audio.spectrogram()
# so I use tf.signal instead
def get_spectrogram(wt):
    """Adapted from help(tf.signal.mfccs_from_log_mel_spectrograms)
    inputs  : waveform tensor, shape (..., samples) with floating point values in [-1, 1]
    returns : log mel spectrograms, shape (..., frames, mel_dim) as float32s
              ... means the batch dimension is optional
    """
    #
    # These have to be tf.constant because if I just do the calculations in Python then tf complains:
    # "ValueError: Creating variables on a non-first call to a function decorated with tf.function."
    frame_length = tf.constant(sample_rate * 25 // 1000)
    frame_step   = tf.constant(sample_rate * 10 // 1000)
    # 
    # produces shape (..., frames, fftbins)
    stfts = tf.signal.stft(wt, frame_length, frame_step)
    spectrograms = tf.abs(stfts)
    #
    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, mel_dim
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
      num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
      upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
      spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
      linear_to_mel_weight_matrix.shape[-1:]))
    #
    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    #
    # return shape (..., frames, mel_dim)
    return log_mel_spectrograms


# map function to convert the Librispeech Dataset dictionary into {logmels, speaker_id, ygt}
@tf.autograph.experimental.do_not_convert
def transform(d):
    wi = d['speech']         # -> (samples)
    wf = tf.cast(wi, dtype=tf.float32)
    wf = wf / 32768.0
    sf = get_spectrogram(wf) # -> (frames, mel_dim)
    nf = tf.shape(sf)[0]     # number of frames
    pd = 2 ** lis_layers     # pyramid downsampling due to listener layers
    nh = nf // pd            # number of hidden representations after pyramid downsampling
    nf = nh * pd             # number of original frames that will be used
    sp = sf[:nf,:]           # spectrogram of these frames only, shape (frames, mel_dim)
    t  = d['text']           # -> () with dtype=tf.string
    # permitted characters are {a,b,c...,z,0,...,9,<space>,<comma>,<period>,<apostrophe>,<unk>}
    # The unknown token I use is ?
    l  = tf.strings.lower(t)
    r  = tf.strings.regex_replace(l,'[^([:alnum:]| |,|.|\')]','?')
    # prefix the text with the <sos> token, for which I use ^
    p1 = tf.strings.join([b'^', r])
    # postfix the text with the <eos> token, for which I use $
    p2 = tf.strings.join([p1, b'$'])
    # Decode the text into numbers, using '?' (63) for any unknown conversions (unlikely).
    # The vocabulary above therefore spans values 32 (space) to 122 (z).
    ysp = tf.strings.unicode_decode(p2,'utf-8', replacement_char=63) # -> (nchars, )
    # To be used in neural networks the decoded values must be 1-hot encoded
    yoh = tf.one_hot(ysp, voc_dim) # -> (nchars, voc_dim) as float32s
    # this function returns a dictionary; gt means ground truth
    o = {'logmels' : sp, 'speaker_id' : d['speaker_id'], 'ygt' : yoh}
    return o


# convert the Librispeech elements into {logmels, speaker_id, ygt}
logm_ds = waug_ds.map(transform)


# The Normalization layer computes and stores means and variances for each logmel dimension.
# Processing all of train_clean100 (~28,000 examples) takes a long time.
# However, the mean and variance vectors don't change much after 100 examples.
# The number to process is chosen with the training parameter norm_count.

print('Computing data normalization vectors...')
norm = preprocessing.Normalization(axis=-1)
norm.adapt(logm_ds.take(norm_count).map(lambda d: d['logmels']))

# normalize the logmels to have 0 mean and stdev 1 in each dimension
@tf.autograph.experimental.do_not_convert
def normalize(d):
    logmels = norm(d['logmels'])
    o = {'logmels' : logmels, 'speaker_id' : d['speaker_id'], 'ygt' : d['ygt']}
    return o


norm_ds = logm_ds.map(normalize)


# padded_batch pads all arrays with 0s to make rectangular tensors
padd_ds = norm_ds.padded_batch(
    batch_size,
    padded_shapes=({'logmels' : (None, mel_dim), 'speaker_id' : (), 'ygt' : (None, voc_dim)}))


# map function to add logmel and character masks to the training dataset
@tf.autograph.experimental.do_not_convert
def gen_masks(d):
    logmels = d['logmels']  # (batch, frames, mel_dim)
    # create a logmel mask where False indicates a padded value (0s in every logmel dimension)
    logmel_mask = tf.cast(tf.reduce_sum(logmels, axis=-1), tf.bool)  # (batch, frames)
    # add to the dictionary
    d['logmel_mask'] = logmel_mask
    # extract the ground truth y values
    ygt = d['ygt']  # (batch, nchars, voc_dim)
    # create a ground truth mask where False indicates a padded value
    ygt_mask = tf.cast(tf.reduce_sum(ygt, axis=-1), tf.bool)  # (batch, nchars)
    # add to the dictionary
    d['ygt_mask'] = ygt_mask
    return d


mask_ds = padd_ds.map(gen_masks)



# Model

# A single listener layer
class ListenerLayer(keras.layers.Layer):
    def __init__(self, lis_dim, **kwargs):
        super(ListenerLayer, self).__init__(**kwargs)
        # dimension of each LSTM
        self.lis_dim = lis_dim
        # bidirectional LSTM layer, whose output will be lis_dim*2 wide
        self.bi = layers.Bidirectional(layers.LSTM(lis_dim, return_sequences=True), merge_mode='concat')
        # Note that tansform() ensures that every example contains a multiple of (2 ** lis_layers) frames.
        # The following reshapes therefore always work perfectly, without any leftovers.
        self.r1 = layers.Reshape((-1, lis_dim*4))
        self.r2 = layers.Reshape((-1, 2))
        #
    def call(self, x, mask):
        # input shapes x = (batch, timesteps, x_dim) and mask = (batch, timesteps)
        x = self.bi(x, mask=mask)              #    x -> (batch, timesteps, lis_dim*2)
        # Take the bi outputs and reshape them so as to combine pairs of outputs by concatenation
        x = self.r1(x)                         #    x -> (batch, timesteps/2, lis_dim*4)
        # similarly reshape the mask
        mask = self.r2(mask)
        mask = tf.reduce_all(mask, axis=-1)    # mask -> (batch, timesteps/2)
        # output shapes x = (batch, timesteps/2, lis_dim*4) and mask = (batch, timesteps/2)
        return x, mask


# The whole multi-layer listener
class Listener(keras.layers.Layer):
    def __init__(self, lis_dim, lis_layers, max_frames, mel_dim, pet_bool, **kwargs):
        super(Listener, self).__init__(**kwargs)
        # log parameters for debugging
        self.lis_dim    = lis_dim
        self.rep_dim    = lis_dim*2
        self.lis_layers = lis_layers
        self.max_frames = max_frames
        self.mel_dim    = mel_dim
        self.pet_bool   = pet_bool
        # positional encoding tensor, shape (max_frames, rep_dim)
        self.pet   = self.compute_pet()
        # learnable input scaling variable
        self.scale = tf.Variable(0.05)
        # a pointwise embedding layer encodes logmels into rep_dim
        self.dem   = layers.Dense(self.rep_dim)
        # setup the listener layers that form the pyramid
        self.lays  = [ListenerLayer(lis_dim) for _ in range(lis_layers)]
        # a final Bidirectional LSTM layer produces the listener feature sequence h
        self.bi    = layers.Bidirectional(layers.LSTM(lis_dim, return_sequences=True), merge_mode='concat')
        #
    def call(self, inputs, mask):
        # inputs shape = (batch, frames, mel_dim) and mask = (batch, frames)
        #
        # embed the logmels in rep_dim dimensions -> (batch, frames, rep_dim)
        emi = self.dem(inputs)
        # scale the embedded inputs to enable them to compete with the positional encoding
        # the scale is learnable, with initial value sqrt(rep_dim / mel_dim)
        sci = emi * tf.sqrt(self.rep_dim / self.mel_dim) * 20.0 * self.scale
        # add the positional encoding, broadcasting over batch
        x = sci + self.pet[0:tf.shape(inputs)[1]]
        # apply the pyramid
        for i in range(self.lis_layers):
            x, mask = self.lays[i](x, mask)
        # and produce the listener feature sequence h
        h = self.bi(x, mask=mask)
        # pd = 2 ** lis_layers, the pyramid downsampling
        # output shapes h = (batch, frames/pd, lis_dim*2) and mask = (batch, frames/pd)
        return h, mask
    #
    def compute_pet(self):
        # compute positional encoding tensor
        pos = np.array(range(self.max_frames))
        pel = []
        for d in range(self.rep_dim):
            if d % 2 == 0:
                row = np.sin(pos/(10000**(d/self.rep_dim)))
            else:
                row = np.cos(pos/(10000**((d-1)/self.rep_dim)))
            pel.append(row)
        # pea has shape (max_frames, rep_dim)
        pea  = np.array(pel).T
        pet  = tf.convert_to_tensor(pea, dtype=tf.float32)
        # if pet_bool is False then turn pet into a tensor of zeros
        pet *= tf.cast(self.pet_bool, tf.float32)
        return pet

    
# The Test Model exists as a sanity check on everything up to here.
# It predicts speaker_id from logmels.
class TestModel(keras.Model):
    """To predict speaker_ids from logmels"""
    def __init__(self, lis_dim, lis_layers, max_frames, mel_dim, pet_bool, num_speakers, **kwargs):
        super(TestModel, self).__init__(**kwargs)
        # log parameters for debugging
        self.lis_dim      = lis_dim
        self.lis_layers   = lis_layers
        self.max_frames   = max_frames
        self.mel_dim      = mel_dim
        self.pet_bool     = pet_bool
        self.num_speakers = num_speakers
        # setup components
        self.listener  = Listener(lis_dim, lis_layers, max_frames, mel_dim, pet_bool)
        self.condenser = layers.LSTM(lis_dim, return_sequences=False)
        self.dense     = layers.Dense(num_speakers)
    def call(self, inputs):
        # models passed to fit() can only have one positional argument plus 'training'
        logmels, logmel_mask = inputs
        h, mask = self.listener(logmels, logmel_mask)
        c = self.condenser(h, mask=mask) # -> (batch, lis_dim)
        d = self.dense(c)                # -> (batch, num_speakers) (softmax logits)
        return d


# Calling fit() on the test model:
#
# instantiate the test model
# tmodel = TestModel(lis_dim, lis_layers, max_frames, mel_dim, pet_bool, 20000)
# tmodel.compile(
#     optimizer=keras.optimizers.RMSprop(1e-3),
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
# )
# using fit requires a simpler dataset structured as (inputs, outputs)
# def tmap(d):
#     return (d['logmels'], d['logmel_mask']), (d['speaker_id'],)
# 
# test_ds = mask_ds.map(tmap)
# history = tmodel.fit(test_ds, epochs=1)
# 
# This works :)

awl = []

# Attention can't be done as a layer, because attention ci is computed from si which isn't available
# until you've run the decoder RNN to compute it from step i-1.  In other words, the attention
# calculation has to take place at each timestep, just like the LSTM cell calculation.  These together
# form a layer.  You can't do them, attention and decoder RNN, as separate layer calculations.
# Turns out the same is true of y(i) calculation, which is sometimes fed back in to compute s(i+1).
# I think this means that I need a custom RNN Cell that does a single step of LSTM and attention.
# The required methods and attributes of custom RNN cells are described in tf.keras.layers.RNN documentation.
class DecoderCell(keras.layers.Layer):
    def __init__(self, dec_dim, att_dim, lis_dim, voc_dim, frac_pyp, **kwargs):
        super(DecoderCell, self).__init__(**kwargs)
        # dec_dim will be the dimension of the DecoderCell's internal LSTMs
        self.dec_dim = dec_dim
        # att_dim is used to construct the MLP used to compute attention queries
        self.att_dim = att_dim
        # lis_dim is the LSTM dimension in the listener pyramid
        self.lis_dim = lis_dim
        # maximum character index plus one
        self.voc_dim = voc_dim
        # fraction of the time we should use the previous y prediction for y input
        self.frac_pyp = frac_pyp
        # call states are [memory, carry] tensors x2 for the internal LSTMs, all shaped (batch, dec_dim),
        # plus previous s vector (batch, dec_dim), previous context vector (batch, lis_dim*2),
        # previous mean attention index (batch, 1), previous monotonicity loss (batch, 1),
        # and previous y prediction (batch, voc_dim)
        self.state_size = [tf.TensorShape([dec_dim]), tf.TensorShape([dec_dim]),
                           tf.TensorShape([dec_dim]), tf.TensorShape([dec_dim]),
                           tf.TensorShape([dec_dim]), tf.TensorShape([lis_dim*2]),
                           tf.TensorShape([1])      , tf.TensorShape([1]),
                           tf.TensorShape([voc_dim])]
        # output size is a yp tensor, shape (batch, voc_dim)
        self.output_size = tf.TensorShape([voc_dim])
        # the LSTMs to be used in the DecoderCell
        self.lstm_cell1 = layers.LSTMCell(self.dec_dim)
        self.lstm_cell2 = layers.LSTMCell(self.dec_dim)
        # the phi attention MLP used to compute queries
        self.phi1 = layers.Dense(att_dim * 2, activation='relu')
        self.phi2 = layers.Dense(att_dim)
        # the chr character distribution MLP
        self.chr1 = layers.Dense(voc_dim * 4, activation='relu')
        self.chr2 = layers.Dense(voc_dim)
        # the attention layer, with scale variable
        self.att  = layers.Attention(use_scale=True)
        #
    def call(self, input_at_t, states_at_t, training, constants=None):
        #
        # input_at_t should be the 1-hot ground truth character vector at timestep i-1
        yin = input_at_t      # shape (batch, voc_dim)
        # states_at_t should be [memory, carry]x2 [psv, pcv, pmai, ploss, pyp] tensors, see state_size above
        pmc1  = states_at_t[0:2] # previous memory and carry for internal LSTM 1
        pmc2  = states_at_t[2:4] # previous memory and carry for internal LSTM 2
        psv   = states_at_t[4]   # previous s vector, shape (batch, dec_dim)
        pcv   = states_at_t[5]   # previous context vector, shape (batch, lis_dim*2)
        pmai  = states_at_t[6]   # previous mean attention index, shape (batch, 1)
        ploss = states_at_t[7]   # previous monotonicity loss, shape (batch, 1)
        pyp   = states_at_t[8]   # previous y prediction, shape (batch, voc_dim) (as logits)
        # training is a Python boolean usually used to control dropout but used here to control logging
        # constants is a keyword argument that can be passed to RNN.__call__() which contains constants:
        listener_features = constants[0] # shape (batch, frames/pd, lis_dim*2)
        listener_keys     = constants[1] # shape (batch, frames/pd, att_dim)
        listener_mask     = constants[2] # shape (batch, frames/pd)
        blend             = constants[3] # shape () tf.bool
        #
        # When blending inputs the y to use (ytu) is produced as either yin (teacher-forcing) or a
        # sample drawn from the previous y prediction (pyp) distribution (to improve model robustness).
        # The decision of which to use is determined randomly, such that frac_pyp are from pyp.
        pypdist = tfp.distributions.OneHotCategorical(logits=pyp)
        sample  = pypdist.sample()                                    # shape (batch, voc_dim) int32
        sample  = tf.cast(sample, tf.float32)                         # shape (batch, voc_dim) float32
        rut     = tf.random.uniform((tf.shape(yin)[0],), seed=1)      # shape (batch,)   floats in [0,1)
        rut     = rut[:, None]                                        # shape (batch, 1) floats in [0,1)
        ytu     = tf.cast(rut <  self.frac_pyp, tf.float32) * sample  # shape (batch, voc_dim) float32
        ytu    += tf.cast(rut >= self.frac_pyp, tf.float32) * yin     # shape (batch, voc_dim) float32
        ytu    *= tf.cast(blend, tf.float32)                          # shape (batch, voc_dim) float32
        # When not blending inputs the y to use (ytu) is always yin.
        ytu    += tf.cast(tf.logical_not(blend), tf.float32) * yin    # shape (batch, voc_dim) float32
        #
        # concatenate psv, ytu and pcv, to produce (batch, dec_dim + voc_dim + lis_dim*2)
        rnni = layers.concatenate([psv, ytu, pcv])
        #
        # Run the internal LSTM cells on the concatenated input to compute s(i), shape (batch, dec_dim).
        # Since all cells must return (output_at_t, states_at_t+1) I presume these do too.
        o1, nmc1 = self.lstm_cell1(rnni, pmc1)
        si, nmc2 = self.lstm_cell2(o1, pmc2)
        #
        # Apply the phi attention MLP to si, producing (batch, att_dim)
        m1 = self.phi1(si)
        m2 = self.phi2(m1)
        # Reshape m2 into the query, shape (batch, 1, att_dim)
        query = layers.Reshape((1, self.att_dim))(m2)
        #
        # Compute the attention weights.
        # TF2.3.0 layers.Attention() does not have an option to return attention weights.
        # However, it can be tricked into returning them by sending an identity matrix as value.
        # Prepare identity tensor (1, frames, frames); the 1 will broadcast over batch
        frames = tf.shape(listener_features)[1]
        it     = tf.keras.initializers.Identity()(shape=(frames, frames))[None,:,:]
        # compute the attention weights (batch, 1, frames)
        aw     = self.att([query, it, listener_keys], mask=[None, listener_mask])
        # compute new context vector (batch, lis_dim*2)
        ci     = tf.matmul(aw, listener_features)[:,0,:]
        # squeeze out the 1 from the attention weights, (batch, frames)
        saw    = aw[:,0,:]
        # compute the new mean attention index
        rt     = tf.range(frames, dtype=tf.float32)[None, :]           # (1, frames)
        nmai   = tf.reduce_sum(saw * rt, axis=-1, keepdims=True)       # (batch, 1)
        # compute monotonicity loss tensor when new mai is less than previous mai, (batch, 1)
        # the capped L1 loss acts to discourage reversals without preventing it completely during learning
        losst  = tf.minimum(tf.abs(nmai - pmai), 1.0) * tf.cast((nmai < pmai), tf.float32)
        # add a capped L1 reward when new mai is more than previous mai, (batch, 1)
        # the reward encourages small forward movements; it could be phased out after bootstrapping
        losst -= tf.minimum(tf.abs(nmai - pmai), 1.0) * tf.cast((nmai > pmai), tf.float32)
        # compute a y mask where 0 is a pad, (batch, 1)
        ymask  = tf.reduce_sum(yin, axis=-1, keepdims=True)
        # the losses are only valid where yin is not a pad, (batch, 1)
        mloss  = ymask * losst
        # compute the new monotonicity loss as a running total (batch, 1)
        nloss  = mloss + ploss
        # if training is False log the attention weights
        if training == False:
            # only for the 0th member of the batch (frames,)
            oaw = saw[0,:]
            # this append wont do anything useful under tf.function()
            # this utility must be called without using tf.function()
            awl.append(oaw)
        # concatenate si and ci to produce (batch, dec_dim + lis_dim*2)
        sc = layers.concatenate([si,ci])
        #
        # The character distribution MLP predicts y as softmax logits over characters (batch, voc_dim)
        ch = self.chr1(sc)
        yp = self.chr2(ch)
        #
        # return outputs at time t, states at time t+1
        # see https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN
        return yp, nmc1 + nmc2 + [si] + [ci] + [nmai] + [nloss] + [yp]
    # The cell could also define a get_initial_state() method, but it doesn't.
    # That means the rnn will feed zeros to call() for the initial state instead.


# The Listen Attend Spell Model
class LASModel(keras.Model):
    def __init__(self, lis_dim, lis_layers, dec_dim, att_dim, voc_dim, frac_pyp, max_dec, max_frames,
                 mel_dim, pet_bool, mloss_wt, **kwargs):
        super(LASModel, self).__init__(**kwargs)
        # monotonicity loss weight hyperparameter
        self.mloss_wt = mloss_wt
        # maximum number of characters to decode in the decode() function
        self.max_dec  = max_dec
        # DecoderCell parameters required in the decode() function
        self.lis_dim  = lis_dim
        self.dec_dim  = dec_dim
        self.voc_dim  = voc_dim
        # record for debugging
        self.lis_layers = lis_layers
        self.frac_pyp   = frac_pyp
        self.att_dim    = att_dim
        self.max_frames = max_frames
        self.mel_dim    = mel_dim
        self.pet_bool   = pet_bool
        # the listener pyramid
        self.listener = Listener(lis_dim, lis_layers, max_frames, mel_dim, pet_bool)
        # the psi attention MLP used to compute the listener keys
        self.psi1     = layers.Dense(att_dim * 2, activation='relu')
        self.psi2     = layers.Dense(att_dim)
        # the decoder cell and rnn
        self.cell     = DecoderCell(dec_dim, att_dim, lis_dim, voc_dim, frac_pyp)
        self.rnn      = layers.RNN(self.cell, return_sequences=True, return_state=True)
        #
    def listen(self, logmels, logmel_mask):
        # The listen() function computes the listener representation, keys and mask.
        #
        # compute the listener representation h and its mask
        h, hmask = self.listener(logmels, logmel_mask)        
        # h           shape (batch, frames/pd, lis_dim*2)
        # hmask       shape (batch, frames/pd)
        #
        # apply the psi attention MLP to get the listener keys
        l1   = self.psi1(h)
        l2   = self.psi2(l1)
        # apply a 1/sqrt(model_dim) scaling here in lieu of one in the Attention() layer
        hkey = l2 / np.sqrt(self.att_dim)
        # hkey        shape (batch, frames/pd, att_dim)
        return h, hkey, hmask
    #
    def call(self, inputs, training):
        # keras models like all their inputs in the first argument
        yins, ymask, logmels, logmel_mask, blend = inputs
        # The call() function operates an rnn, to be used for training/validation/teacher-forced-prediction.
        #
        # yins        shape (batch, nchars, voc_dim)
        # ymask       shape (batch, nchars)
        # logmels     shape (batch, frames, mel_dim)
        # logmel_mask shape (batch, frames)
        # blend       shape () tf boolean
        # training    Python boolean
        #
        # compute the listener representation, key and mask
        h, hkey, hmask = self.listen(logmels, logmel_mask)
        # h           shape (batch, frames/pd, lis_dim*2)
        # hkey        shape (batch, frames/pd, att_dim)
        # hmask       shape (batch, frames/pd)
        #
        # Compute the y predictions as softmax logits, and the last cell state.
        # Now that I return the last state the ymask is needed here; pads cause state to be copied forward. 
        r = self.rnn(yins, mask=ymask, training=training, constants=[h, hkey, hmask, blend])
        #
        yps   = r[0] # y predictions,     shape (batch, nchars, voc_dim)
        mloss = r[8] # monotonicity loss, shape (batch, 1)
        #
        # divide out the number of positions that contributed to mloss, (batch, 1)
        mloss /= tf.reduce_sum(tf.cast(ymask, tf.float32))
        # scale the monotonicity loss tensor using its hyperparameter, (batch, 1)
        mloss *= self.mloss_wt
        # add the monotinicity loss
        # adding this loss inside DecoderCell causes an InaccessibleTensorError at runtime
        self.add_loss(tf.reduce_sum(mloss))
        #
        return yps
    #
    def decode(self, logmels, logmel_mask):
        # The decode() function performs decoding, predicting unknown characters from logmels.
        # 
        # logmels     shape (batch, frames, mel_dim)
        # logmel_mask shape (batch, frames)
        #
        # where this function expects batch = 1
        tf.debugging.assert_equal(tf.shape(logmels)[0],     1, message="las.decode() expects batch_size 1")
        tf.debugging.assert_equal(tf.shape(logmel_mask)[0], 1, message="las.decode() expects batch_size 1")
        #
        # compute the listener representation, key and mask
        h, hkey, hmask = self.listen(logmels, logmel_mask)
        # h           shape (batch, frames/pd, lis_dim*2)
        # hkey        shape (batch, frames/pd, att_dim)
        # hmask       shape (batch, frames/pd)
        #
        # the DecoderCell should not blend its inputs when decoding
        blend    = tf.constant(False)
        # the <sos> token is always the first input
        sos_text = b'^'
        sos_code = tf.strings.unicode_decode(sos_text,'utf-8')    # (1,)
        # the [0] subscript defining yin tells tf.function() that yin has leading dimension 1
        yin      = tf.one_hot(sos_code, self.voc_dim)[0][None,:]  # (1, voc_dim)
        # the <eos> token tells us when to stop decoding
        eos_text = b'$'
        eos_code = tf.strings.unicode_decode(eos_text,'utf-8')    # (1,)
        # use TensorArray to accumulate a sparse array of unknown size of decoded int32s
        dec_ta   = tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=True)
        dec_i    = 0
        # initial state for the DecoderCell with batch_size of 1
        istate   = [tf.zeros((1, self.dec_dim)), tf.zeros((1, self.dec_dim)),
                    tf.zeros((1, self.dec_dim)), tf.zeros((1, self.dec_dim)),
                    tf.zeros((1, self.dec_dim)), tf.zeros((1, self.lis_dim*2)),
                    tf.zeros((1, 1))           , tf.zeros((1, 1)),
                    tf.zeros((1, self.voc_dim))]
        state    = istate
        # initial y decoded (for loop comparisons)
        yd       = sos_code # (1,)
        # prepare while loop functions
        def cond(yin, state, yd, dec_ta, dec_i):
            return yd != eos_code
        #
        def body(yin, state, yd, dec_ta, dec_i):
            yp, state = self.cell(yin, state, training=False, constants=[h, hkey, hmask, blend])
            # yp shape (1, voc_dim)
            #
            # in this simple decoder the most likely character becomes the decoded char for this timestep
            yd     = tf.argmax(yp, axis=-1)                      # (1,) sparse decoded int64
            yd     = tf.cast(yd, tf.int32)                       # (1,) sparse decoded int32
            # accumulate
            dec_ta = dec_ta.write(dec_i, yd)
            dec_i += 1
            # prepare next input
            yin    = tf.one_hot(yd, self.voc_dim)                # (1, voc_dim)
            # return loop vars
            return yin, state, yd, dec_ta, dec_i
        #
        yin, state, yd, dec_ta, dec_i = tf.while_loop(
            cond, body, (yin, state, yd, dec_ta, dec_i), maximum_iterations=self.max_dec)
        #
        dec_st = tf.squeeze(dec_ta.stack())                      # (nchars,) sparse decoded int32
        return dec_st



# instantiate the model
las = LASModel(lis_dim, lis_layers, dec_dim, att_dim, voc_dim, frac_pyp, max_dec, max_frames, mel_dim,
               pet_bool,mloss_wt)


# optimizer
# Learning rate schedules have not been investigated for the current code.
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)


# define accumulators
train_loss = tf.keras.metrics.Mean()
train_acc  = tf.keras.metrics.Mean()

val_loss   = tf.keras.metrics.Mean()
val_acc    = tf.keras.metrics.Mean()


# loss function
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')

@tf.autograph.experimental.do_not_convert
def loss_function(tars, mask, logits):
    # losses due to pads must not be counted
    lt   = loss_object(tars, logits)         # (batch, nchars)
    mask = tf.cast(mask, lt.dtype)           # (batch, nchars) 0.0 where there's a pad, 1.0 otherwise
    lt   = lt * mask
    return tf.reduce_sum(lt) / tf.reduce_sum(mask)


# accuracy function
@tf.autograph.experimental.do_not_convert
def acc_function(tars, mask, logits):
    # accuracies of pads must not be counted
    preds = tf.argmax(logits, axis=-1)       # (batch, nchars)
    tars  = tf.argmax(tars, axis=-1)         # (batch, nchars)
    acc   = tf.equal(tars, preds)            # (batch, nchars) bools
    acc   = tf.cast(acc, tf.float32)         # (batch, nchars) float32s
    mask  = tf.cast(mask, tf.float32)        # (batch, nchars) float32s, 0 for pads
    acc   = acc * mask
    return tf.reduce_sum(acc) / tf.reduce_sum(mask)


# Adapted from https://www.tensorflow.org/text/tutorials/transformer:
# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

signature_dict = { 'logmels'     : tf.TensorSpec(shape=(None, None, mel_dim), dtype=tf.float32),
                   'logmel_mask' : tf.TensorSpec(shape=(None, None),          dtype=tf.bool),
                   'ygt'         : tf.TensorSpec(shape=(None, None, voc_dim), dtype=tf.float32),
                   'ygt_mask'    : tf.TensorSpec(shape=(None, None),          dtype=tf.bool),
                   'speaker_id'  : tf.TensorSpec(shape=(None),                dtype=tf.int64) }


@tf.function(input_signature = [signature_dict])
@tf.autograph.experimental.do_not_convert
def train_step(d):
    # ygt is ground truth chars, including start and end characters
    ygt        = d['ygt']         # (batch, nchars, voc_dim)
    ygt_mask   = d['ygt_mask']    # (batch, nchars)
    # y inputs exclude end character
    yins       = ygt[:, :-1]      # (batch, nchars, voc_dim)
    yins_mask  = ygt_mask[:, :-1] # (batch, nchars)
    # y targets exclude the start character
    ytars      = ygt[:, 1:]       # (batch, nchars, voc_dim)
    ytars_mask = ygt_mask[:, 1:]  # (batch, nchars) 
    # training blends ground truth inputs with predictions
    blend = tf.constant(True)
    # the training boolean is set to True to avoid wasting time recomputing attention weights
    with tf.GradientTape() as tape:
        # call the model to obtain the y predictions (logits) (batch, nchars, voc_dim)
        yps = las([yins, yins_mask, d['logmels'], d['logmel_mask'], blend], training=True)
        # compute the loss
        loss = loss_function(ytars, ytars_mask, yps)
        # add the monotonicity loss
        loss += sum(las.losses)
        # tf.print(sum(las.losses))
    #
    # compute and apply the gradients
    gradients = tape.gradient(loss, las.trainable_variables)
    # clip gradients to stabilize training
    # with batch_size=32 gradient norms are only rarely over 2.0
    gradients, gn = tf.clip_by_global_norm(gradients, 2.0)
    # tf.print(gn, tf.linalg.global_norm(gradients))
    optimizer.apply_gradients(zip(gradients, las.trainable_variables))
    # accumulate
    train_loss(loss)
    train_acc(acc_function(ytars, ytars_mask, yps))


@tf.function(input_signature = [signature_dict])
@tf.autograph.experimental.do_not_convert
def val_step(d):
    # ygt is ground truth chars, including start and end characters
    ygt        = d['ygt']         # (batch, nchars, voc_dim)
    ygt_mask   = d['ygt_mask']    # (batch, nchars)
    # y inputs exclude end character
    yins       = ygt[:, :-1]      # (batch, nchars, voc_dim)
    yins_mask  = ygt_mask[:, :-1] # (batch, nchars)
    # y targets exclude the start character
    ytars      = ygt[:, 1:]       # (batch, nchars, voc_dim)
    ytars_mask = ygt_mask[:, 1:]  # (batch, nchars)
    # blend is True so that validation loss/acc are comparable to training loss/acc
    # (validation results are only useful comparatively since this is not the ultimate task)
    blend = tf.constant(True)
    # call the model to obtain the y predictions (logits) (batch, nchars, voc_dim)
    yps = las([yins, yins_mask, d['logmels'], d['logmel_mask'], blend], training=True)
    # compute the target loss
    loss = loss_function(ytars, ytars_mask, yps)
    # add the monotonicity loss
    loss += sum(las.losses)
    # tf.print(sum(las.losses))
    # accumulate
    val_loss(loss)
    val_acc(acc_function(ytars, ytars_mask, yps))


# load the transcription alignment dictionaries generated with Kaldi
train_ad  = load_alignment_dict(train_ali_fn)
dev_ad    = load_alignment_dict(dev_ali_fn)

# I'll use the dev set as validation data
dbu_ds    = builder.as_dataset(split="dev_clean")
dfi_ds    = dbu_ds.filter(filter_lengths)
# I'll use a constant max_num_words validation set, rather than varying it over time.
# Prepare a tf.lookup table containing the initial max_num_words words from each alignment.
dev_table = sample_alignments(dev_ad, max_num_words, rand=False)
# extract the table-specified sequence for each dataset record
dex_ds    = dfi_ds.map(lambda x: extract(x, dev_table))
# follow the standard dataset preparation sequence
dev_ds    = dex_ds.map(wav_augment)
dev_ds    = dev_ds.map(transform)
dev_ds    = dev_ds.map(normalize)
dev_ds    = dev_ds.padded_batch(
    batch_size,
    padded_shapes=({'logmels' : (None, mel_dim), 'speaker_id' : (), 'ygt' : (None, voc_dim)}))
dev_ds    = dev_ds.map(gen_masks)
# I'll validate on about 512 utterances
val_steps = 512 // batch_size


# Create a Checkpoint that will manage objects with trackable state,
# one I name "optimizer" and the other I name "model".
checkpoint           = tf.train.Checkpoint(optimizer=optimizer, model=las)
checkpoint_directory = './checkpoints'
checkpoint_prefix    = os.path.join(checkpoint_directory, 'ckpt')

# setup tensorboard logging directories
# specify -p 6010:6010 in docker run command for tensorflow image
# run "tensorboard --logdir logs --host 0.0.0.0 --port 6010" in docker shell
# point browser at http://localhost:6010/
current_time         = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir        = 'logs/' + current_time + '/train'
val_log_dir          = 'logs/' + current_time + '/val'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
val_summary_writer   = tf.summary.create_file_writer(val_log_dir)

# global step counter
gstep = 0

# training loop
# If you've already trained a system and saved checkpoints you can kill the training loop after
# 1 batch (required to set up objects in the optimizer) and then load the latest checkpoint below.
for num_words in range(1, max_num_words+1):
    # I currently train for a fixed number of epochs (and hence steps) for each number of words.
    # This is a design choice which may not be optimal.
    for epoch in range(num_epochs):
        start = time.time()
        # Prepare the curriculum learning dataset for this number of words for this epoch.
        # Prepare a tf.lookup table containing a random num_words sequence sampled from each alignment.
        # Using a new randomization for each epoch ensures that the model trains on as much data as possible.
        table   = sample_alignments(train_ad, num_words, rand=True)
        # extract the table-specified random sequence for each dataset record
        curr_ds = shuf_ds.map(lambda x: extract(x, table))
        # follow the standard dataset preparation sequence
        curr_ds = curr_ds.map(wav_augment)
        curr_ds = curr_ds.map(transform)
        curr_ds = curr_ds.map(normalize)
        curr_ds = curr_ds.padded_batch(
            batch_size,
            padded_shapes=({'logmels' : (None, mel_dim), 'speaker_id' : (), 'ygt' : (None, voc_dim)}))
        curr_ds = curr_ds.map(gen_masks)
        #
        # reset accumulators
        train_loss.reset_states()
        train_acc.reset_states()
        val_loss.reset_states()
        val_acc.reset_states()
        #
        for i,d in enumerate(curr_ds):
            train_step(d)
            # I will monitor training over groups of batches since there will not be many epochs
            if i % 10 == 0:
                print(f'num_words {num_words}  epoch {epoch+1}  batch {i}',
                      f'train_loss {train_loss.result():.4f}',
                      f'train_acc {train_acc.result():.4f}')
                # tensorboard log
                with train_summary_writer.as_default():
                    _ = tf.summary.scalar('loss', train_loss.result(), step=gstep)
                    _ = tf.summary.scalar('acc',  train_acc.result(),  step=gstep)
                # reset training accumulators
                train_loss.reset_states()
                train_acc.reset_states()
            gstep += 1
        # save epoch checkpoint
        cp = checkpoint.save(file_prefix=checkpoint_prefix)
        print('saving checkpoint to', cp)
        # validate epoch
        print('validating', end='', flush=True)
        for i,d in enumerate(dev_ds):
            if i == val_steps:
                break
            val_step(d)
            if i % 10 == 0:
                print('.', end='', flush=True)
        print(f'\nnum_words {num_words}  epoch {epoch + 1}',
              f'val loss {val_loss.result():.4f}  val acc {val_acc.result():.4f}')
        # tensorboard log
        with val_summary_writer.as_default():
            _ = tf.summary.scalar('loss', val_loss.result(), step=gstep)
            _ = tf.summary.scalar('acc',  val_acc.result(),  step=gstep)
        #
        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')



# With the trained system, do some predictions.

# Load the latest checkpoint from disk
status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
status.assert_consumed()

# set up a new dev_clean data pipeline with no wav_augment and batch_size=4
dtr_ds    = dex_ds.map(transform)
dno_ds    = dtr_ds.map(normalize)
dpd_ds    = dno_ds.padded_batch(
    batch_size=4,
    padded_shapes=({'logmels' : (None, mel_dim), 'speaker_id' : (), 'ygt' : (None, voc_dim)}))
dmk_ds    = dpd_ds.map(gen_masks)

# First, look at model predictions when teacher-forcing each input character.

@tf.function(input_signature = [signature_dict])
@tf.autograph.experimental.do_not_convert
def pred_step(d):
    # ygt is ground truth chars, including start and end characters
    ygt        = d['ygt']         # (batch, nchars, voc_dim)
    ygt_mask   = d['ygt_mask']    # (batch, nchars)
    # y inputs exclude end character
    yins       = ygt[:, :-1]      # (batch, nchars, voc_dim)
    yins_mask  = ygt_mask[:, :-1] # (batch, nchars)
    # y targets exclude the start character
    ytars      = ygt[:, 1:]       # (batch, nchars, voc_dim)
    ytars_mask = ygt_mask[:, 1:]  # (batch, nchars)
    # blend is False since we want to always use the teacher-forced input char
    blend      = tf.constant(False)
    # call the model to obtain the y predictions (logits) (batch, nchars, voc_dim)
    # set training to False to switch off any future dropout; this will write junk to awl
    yps = las([yins, yins_mask, d['logmels'], d['logmel_mask'], blend], training=False)
    return yps, ytars, ytars_mask



# Produce predictions for 1 batch
print('Predictions when teacher-forcing each input character:\n')
for i, d in enumerate(dmk_ds):
    # get the predictions, as logits
    yps, ytars, ytars_mask = pred_step(d)
    # get the targets as a sparse tensor (batch, nchars)
    sparse_tars = tf.argmax(ytars, axis=-1)
    sparse_tars = tf.cast(sparse_tars, dtype=tf.int32)
    # encode into text (batch)
    tar_text = tf.strings.unicode_encode(sparse_tars, 'UTF-8', replacement_char=63)
    # argmax over the probabilities of each character -> (batch, nchars)
    sparse_preds = tf.argmax(yps, axis=-1)
    sparse_preds = tf.cast(sparse_preds, dtype=tf.int32)
    # encode into text (batch)
    pred_text = tf.strings.unicode_encode(sparse_preds, 'UTF-8', replacement_char=63)
    # compute target lengths (batch)
    lengths = tf.reduce_sum(tf.cast(ytars_mask, tf.int32), axis=-1)
    # printouts
    for t,p,l in zip(tar_text, pred_text, lengths):
        print('predicted', p.numpy()[:l.numpy()])
        print('target   ', t.numpy()[:l.numpy()])
        print()
    break


# Second, look at predictions as pure decodings, from logmels and the <sos> token:

signature_list = [ tf.TensorSpec(shape=(1, None, mel_dim), dtype=tf.float32),
                   tf.TensorSpec(shape=(1, None),          dtype=tf.bool) ]

@tf.function(input_signature = signature_list)
@tf.autograph.experimental.do_not_convert
def decode_step(logmels, logmel_mask):
    # call the model to obtain the decoded text
    dec_text = las.decode(logmels, logmel_mask)
    return dec_text


# Produce predictions for 4 examples
print('Predictions as pure decodings, starting from <sos>:\n')
for i, d in enumerate(dmk_ds.unbatch().batch(1)):
    logmels     = d['logmels']                      # (1, frames, mel_dim)
    logmel_mask = d['logmel_mask']                  # (1, frames)
    # call decoder
    dec_st      = decode_step(logmels, logmel_mask) # (nchars,)
    dec_text    = tf.strings.unicode_encode(dec_st, 'UTF-8', replacement_char=63) # ()
    # ygt is ground truth chars, including start and end characters
    ygt         = d['ygt']                          # (1, nchars, voc_dim)
    ygt_mask    = d['ygt_mask']                     # (1, nchars)
    # y targets exclude the start character
    ytars       = ygt[:, 1:]                        # (1, nchars, voc_dim)
    ytars_mask  = ygt_mask[:, 1:]                   # (1, nchars) 
    # get the targets as a sparse tensor              (nchars, )
    sparse_tars = tf.argmax(ytars, axis=-1)
    sparse_tars = tf.cast(sparse_tars, dtype=tf.int32)
    sparse_tars = tf.squeeze(sparse_tars)
    # encode the targets into text                    ()
    tar_text = tf.strings.unicode_encode(sparse_tars, 'UTF-8', replacement_char=63)
    # compute target length                           ()
    length = tf.squeeze(tf.reduce_sum(tf.cast(ytars_mask, tf.int32), axis=-1))
    # printouts
    print('decoded', dec_text.numpy())
    print('target ', tar_text.numpy()[:length.numpy()])
    print('\n')
    if i == 3:
        break



# Third, look at speller-listener attention weights when teacher-forcing each character
def att_step(d):
    # ygt is ground truth chars, including start and end characters
    ygt        = d['ygt']         # (batch, nchars, voc_dim)
    ygt_mask   = d['ygt_mask']    # (batch, nchars)
    # y inputs exclude end character
    yins       = ygt[:, :-1]      # (batch, nchars, voc_dim)
    yins_mask  = ygt_mask[:, :-1] # (batch, nchars)
    # y targets exclude the start character
    ytars      = ygt[:, 1:]       # (batch, nchars, voc_dim)
    ytars_mask = ygt_mask[:, 1:]  # (batch, nchars)
    # blend is False since we want to always use the teacher-forced input char
    blend      = tf.constant(False)
    # training is False to switch on the attention weight logging code
    # call the model to obtain the y predictions (logits) (batch, nchars, voc_dim)
    yps = las([yins, yins_mask, d['logmels'], d['logmel_mask'], blend], training=False)
    return yps, ytars, ytars_mask


# set up a new dev_clean data pipeline with no wav_augment and batch_size=1
dtr_ds    = dex_ds.map(transform)
dno_ds    = dtr_ds.map(normalize)
dba_ds    = dno_ds.batch(1)
dmk_ds    = dba_ds.map(gen_masks)
# extract second validation data example ("horses")
for i,d in enumerate(dmk_ds):
    if i == 1:
        break

# call att_step(), collecting useful output in awl
awl     = []
_, _, _ = att_step(d)
awa     = tf.convert_to_tensor(awl).numpy()
awa.shape
# should be (36, 20) : 36 characters, 160//8 reps

import matplotlib.pyplot as plt
plt.imshow(awa, cmap='Greys')
plt.ylabel('characters')
plt.xlabel('speech representation timesteps')
plt.title('Decoder attention weights for the validation data utterance\n'
          '"anyway he would never allow one of"\n\n')
plt.tight_layout()
plt.show()
