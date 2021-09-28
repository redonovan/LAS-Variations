# Listen Attend Spell Variations

<b>Research</b>

I explored several variations of my <a href=https://github.com/redonovan/Listen-Attend-Spell>Listen Attend Spell</a> implementation in an effort to improve its performance, especially its attendance to the acoustics.  

<b> Attention Logit Scaling </b>

Analysis showed that my implementation had learned to put all its attention on one acoustic timestep early in training, from which it never recovered.  Reasoning that the one-hot nature of the softmax outputs meant that the logits were too large I decided to introduce 1/sqrt(model_dim) attention logit scaling as used in Transformers.  With this change the first two characters maintained attention ambiguity for the first 7 epochs of training and space characters began to attend more widely from epochs 9-10.  This plot shows the attention weights after 11 epochs (8 hours) of training:

![scaled attention weights picture](AttentionWeightsScaled.png)

Validation accuracy with this system was similar at 55.2% but decoded text lengths were more appropriate.

<b> Curriculum Learning </b>

A major difference between the speech dataset used in the <a href=https://arxiv.org/abs/1508.01211>paper</a> and <a href=https://www.openslr.org/12>LibriSpeech</a> is the length distribution over utterances.  Figure 4 in the paper shows that the mode of their length distribution was just 3 words, whereas in LibriSpeech it is closer to 37.  Reasoning that attention might be easier to learn from short sentences I tried a curriculum learning strategy in which I slowly increased the sentence length of training utterances over time.  Unfortunately LibriSpeech contains so few short sentences that the only noticeable effect was to overfit to the short training data.

<b> Monotonicity Loss </b>

It has been noted before that while language translation tasks benefit from the ability of attention mechanisms to jump back and forth in time, in speech tasks we would prefer a more monotonic advancement in attention.  Reasoning that the system might find it easier to learn to move its attention focus slowly forward if given an explicit time reference, I first introduced a positional encoding embedding as used in Transformers, and when that didn't help I added a new monotonicity loss to encourage the desired behaviour.  Experiment suggested that a strong penalty on non-monotonic behaviour did not help, and I reason that it may actually impede learning; if the system attends too late with the 1st character a strong penalty prevents it from learning to attend correctly to later characters.  The system I used therefore gives a small, capped, penalty for moving attention focus backwards, and an equal small, capped, reward for moving attention focus forwards.  Applied from the start of training this scheme quickly leads to the attention weights fanning out from the top left of the attention plots.  However it also makes it more difficult for the system to escape the 18% training data accuracy plateau that all training runs seem to hit (where spaces are being spelled correctly, but nothing else).  Waiting until the end of the 1st epoch before applying the monotonicity loss works well, however, with both training and validation accuracy continuing to climb, and the final attention weights looking like:

![monotonicity loss attention weights picture](AttentionWeightsMono.png)

This system achieved a validation accuracy of 54.5% after 9 epochs of training (~8 hours).  The code, which includes Attention Logit Scaling, is in <a href=listenattendspellmono.py>listenattendspellmono.py</a>

<b> Conclusion </b>

The attention plots resulting from these variations look more promising than those of my base implementation.  Unfortunately I don't currently have the computing resources to find out whether any of them would ultimately lead to a fully converged system.

