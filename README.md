# ChatBot enriched with Intent detection and Joint Embedding 
Chatbot-IDJE is a joint model for learning  Goal-Oriented Dialogues generation based on paper:

#### Incorporating Joint Embeddings into Goal-Oriented Dialogues with Multi-Task Learning [here](https://arxiv.org/abs/2001.10468)

# Contents
* ./src/reps.cc is the source code for training the model
* ./src/comb_NNE.py is the source code for NNE algorithm (Journal version)
* ./src/comb_MNE.py is the source code for MNE algorithm (Journal version)
* ./work includes all the lexicon files and a small co-occurrence matrix sample (sampleEdges)
* ./vectors the pretrained word vectors are available for download

# Requirements


# Examples
* To train the model with a normal Vanila Seq2Seq: 
  * python pytorch_main.py (default parameters are:
    *  --epochs 1000 , --embedding 300, --batch-size 126, 
    * --emb None( add file path to use Joint pretrained embeddings)
    * --intent False ( True/1 to train the model with intent detection)
* To get the processed data use the class TextData as follows:
```
textdata = TextData(train_file, valid_file, test_file, pretrained_emb_file=args.emb,
                        useGlove=args.glove)
textdata.getBatches(args.batch_size)
     # each batch contains the following:    
        encoderSeqs = []
        encoderSeqsLen = []
        decoderSeqs = []
        decoderSeqsLen = []
        seqIntent=[]
        kb_inputs = []
        kb_inputs_mask = []
        targetKbMask = []
        targetSeqs = []
        weights = []
        encoderMaskSeqs = []
        decoderMaskSeqs = []

```
