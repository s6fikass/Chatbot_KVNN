import torch
import torch.nn as nn
import argparse
from model.vanilla_seq2seq import Seq2SeqmitAttn
#from args import get_args
import numpy as np
import torch
from batcher_dc import DialogBatcher
from tqdm import tqdm
import pandas as pd
import random
from util.utils import save_model, load_model
#from metrics import EmbeddingMetrics
from util.bleu import get_moses_multi_bleu
#from train_kg_attn import get_sentences
#get arguments
#args = get_args()

parser = argparse.ArgumentParser()
named_args = parser.add_argument_group('named arguments')

named_args.add_argument('-e', '--epochs', metavar='|',
                        help="""Number of Epochs to Run""",
                        required=False, default=1000, type=int)
named_args.add_argument('-rs', '--randseed', metavar='|',
                        help="""random seed""",
                        required=False, default=666, type=int)

named_args.add_argument('-es', '--embdim', metavar='|',
                        help="""Size of the embedding""",
                        required=False, default=300, type=int)

named_args.add_argument('-hs', '--hidden-size', metavar='|',
                        help="""Size of the rnn hidden layer""",
                        required=False, default=300, type=int)

named_args.add_argument('-rd', '--rnn-dropout', metavar='|',
                        help="""Size of the rnn hidden layer""",
                        required=False, default=0.5, type=float)

named_args.add_argument('-ed', '--emb-drop', metavar='|',
                        help="""Size of the rnn hidden layer""",
                        required=False, default=0.5, type=float)

named_args.add_argument('-g', '--gpu', metavar='|',
                        help="""GPU to use""",
                        required=False, default='1', type=str)

named_args.add_argument('-p', '--padding', metavar='|',
                        help="""Amount of padding to use""",
                        required=False, default=20, type=int)

named_args.add_argument('-t', '--training-data', metavar='|',
                        help="""Location of training data""",
                        required=False, default='./data/train_data.csv')

named_args.add_argument('-v', '--validation-data', metavar='|',
                        help="""Location of validation data""",
                        required=False, default='./data/val_data.csv')

named_args.add_argument('-b', '--batch-size', metavar='|',
                        help="""Location of validation data""",
                        required=False, default=50, type=int)

named_args.add_argument('-tm', '--loadFilename', metavar='|',
                        help="""Location of trained model """,
                        required=False, default=None, type=str)

named_args.add_argument('-m', '--model', metavar='|',
                        help="""Location of trained model """,
                        required=False, default="Seq2Seq", type=str)

named_args.add_argument('-val', '--val', metavar='|',
                        help="""Location of trained model """,
                        required=False, default=False, type=bool)

named_args.add_argument('-cuda', '--cuda', metavar='|',
                        help="""to use cuda """,
                        required=False, default=False, type=bool)

named_args.add_argument('-emb', '--emb', metavar='|',
                        help="""to use Joint pretrained embeddings """,
                        required=False, default=None, type=str)

named_args.add_argument('-glove', '--glove', metavar='|',
                        help="""to use Glove or any unfamiliar pretrained embeddings """,
                        required=False, default=None, type=bool)

named_args.add_argument('--no_tqdm', default=False, 
                        action='store_true', help='disable tqdm progress bar')

named_args.add_argument('-intent', '--intent', metavar='|',
                        help="""Joint learning based on intent """,
                        required=False, default=False, type=bool)

named_args.add_argument('-kb', '--kb', metavar='|',
                        help="""Joint learning based on intent with kb tracking and key-value lookups """,
                        required=False, default=False, type=bool)

named_args.add_argument('-test', '--test', metavar='|',
                        help="""test the model on one batch and evaluate on the same """,
                        required=False, default=False, type=bool)

named_args.add_argument('-lr', '--lr', metavar='|',
                        help="""model learning rate """,
                        required=False, default=0.001, type=float)

args = parser.parse_args()
if args.cuda:
    USE_CUDA = True

# Set random seed
np.random.seed(args.randseed)
torch.manual_seed(args.randseed)

if args.gpu:
    torch.cuda.manual_seed(args.randseed)

chat_data = DialogBatcher(gpu=args.gpu, batch_size=args.batch_size)
model = Seq2SeqmitAttn(hidden_size=args.hidden_size, max_r=chat_data.max_resp_len, gpu=args.gpu, n_words=chat_data.n_words,
                       emb_dim=args.embdim, b_size=args.batch_size, dropout=args.rnn_dropout, emb_drop=args.emb_drop,
                       pretrained_emb=chat_data.vectors, sos_tok=chat_data.stoi['<go>'],
                       eos_tok=chat_data.stoi['<eos>'], itos=chat_data.geti2w)
model_name = 'Seq2SeqmitAttnVanilla'
test_results = 'test_predicted_vanilla.csv'
test_out = pd.DataFrame()


def train():

    best_val_loss = 100.0
    emb_val = 0.0

    for epoch in range(args.epochs):
        model.train()
        print('\n\n-------------------------------------------')
        print('Epoch-{}'.format(epoch))
        print('-------------------------------------------')

        train_iter = enumerate(chat_data.get_iter('train'))
        if not args.no_tqdm:
            train_iter = tqdm(train_iter)
            train_iter.set_description_str('Training')
            train_iter.total = chat_data.n_train // chat_data.batch_size

        for it, mb in train_iter:
            q, a, q_m, a_m = mb
            #print (q.size(), a.size())
            model.train_batch(q, a, q_m, a_m)
            train_iter.set_description(model.print_loss())

        print('\n\n-------------------------------------------')
        print('Validation')
        print('-------------------------------------------')
        val_iter = enumerate(chat_data.get_iter('valid'))
        if not args.no_tqdm:
            val_iter = tqdm(val_iter)
            val_iter.set_description_str('Validation')
            val_iter.total = chat_data.n_val // chat_data.batch_size

        val_loss = 0.0
        #extrema = []
        #gm = []
        #emb_avg_all = []
        predicted_s = []
        orig_s = []
        for it, mb in val_iter:
            q, a, q_m, a_m = mb
            pred, loss = model.evaluate_batch(q, a, q_m, a_m)
            pred = pred.transpose(0, 1).contiguous()
            a = a.transpose(0, 1).contiguous()
            s_g = get_sentences(a)
            s_p = get_sentences(pred)
            #e_a, v_e, g_m = metrics.get_metrics_batch(s_g, s_p)
            #emb_avg_all.append(e_a)
            #extrema.append(v_e)
            #gm.append(g_m)
            predicted_s.append(s_p)
            orig_s.append(s_g)
            val_loss += loss.data[0]
        print('\n\n-------------------------------------------')
        print ('Sample prediction')
        print('-------------------------------------------')
        rand = random.randint(0, len(s_g))
        try:
            p = s_p[rand]
            o = s_g[rand]
            print ('Original:' + o)
            try:
                print ('Predicted:' + p)
            except UnicodeEncodeError:
                print ('Predicted: '.format(p))
            print('-------------------------------------------')
        except IndexError:
            print ('Getting validation scores.....')
        v_l = val_loss/val_iter.total
        #ea = np.average(gm)
        #print ("Vector extrema:" + str(np.average(extrema)))
        #print ("Embedding Average:" + str(np.average(emb_avg_all)))
        #print ('Greedy Matching for this epoch:{:.6f}'.format(np.average(gm)))
        predicted_s = [q for ques in predicted_s for q in ques]
        orig_s = [q for ques in orig_s for q in ques]
        moses_bleu = get_moses_multi_bleu(predicted_s, orig_s, lowercase=True)
        if moses_bleu is None:
            moses_bleu = 0
        print ('Length of pred:' + str(len(orig_s)) + ' moses bleu: '+str(moses_bleu))
        #ea = moses_bleu
        if moses_bleu > emb_val:
            emb_val = moses_bleu
            print ('Saving best model')
            save_model(model, model_name)
        else:
            print ('Not saving the model. Best validation moses bleu so far:{:.4f}'.format(emb_val))
        print ('Validation Loss:{:.2f}'.format(val_loss/val_iter.total))
        # sprint ('Embedding average:{:.6f}'.format(emb_val))


def test(model):
    model = load_model(model, model_name, args.gpu)
    print('\n\n-------------------------------------------')
    print('Testing')
    print('-------------------------------------------')
    test_iter = enumerate(chat_data.get_iter('test'))
    if not args.no_tqdm:
        test_iter = tqdm(test_iter)
        test_iter.set_description_str('Testing')
        test_iter.total = chat_data.n_test // chat_data.batch_size

    test_loss = 0.0
    # extrema = []
    # gm = []
    # emb_avg_all = []
    predicted_s = []
    orig_s = []
    for it, mb in test_iter:
        q, a, q_m, a_m = mb
        pred, loss = model.evaluate_batch(q, a, q_m, a_m)
        # print('=================Predicted vectors================')
        # print(pred[0])
        pred = pred.transpose(0, 1).contiguous()
        a = a.transpose(0, 1).contiguous()
        s_g = get_sentences(a)
        s_p = get_sentences(pred)
        # e_a, v_e, g_m = metrics.get_metrics_batch(s_g, s_p)
        # emb_avg_all.append(e_a)
        # extrema.append(v_e)
        # gm.append(g_m)
        predicted_s.append(s_p)
        orig_s.append(s_g)
        test_loss += loss.item()

    # print ("Vector extrema:" + str(np.average(extrema)))
    # print ("Greedy Matching:" + str(np.average(gm)))
    # print ("Embedding Average on Test:{:.6f}".format(np.average(emb_avg_all)))
    print('\n\n-------------------------------------------')
    predicted_s = [q for ques in predicted_s for q in ques]
    orig_s = [q for ques in orig_s for q in ques]
    moses_bleu = get_moses_multi_bleu(predicted_s, orig_s, lowercase=True)
    print ("Moses Bleu on test:" + str(moses_bleu))
    test_out['original_response'] = orig_s
    test_out['predicted_response'] = predicted_s
    print ('Saving the test predictions......')
    test_out.to_csv(test_results, index=False)


def get_sentences(sent_indexed):
    # print (len(teams), len(sent_indexed))
    out_sents = [get_sent(sent_indexed[i]) for i in range(len(sent_indexed))]
    # print (out_sents[0])
    out_sents = [sent for sent in out_sents]
    # fetched =
    # fetched = [fetched for sent in out_sents]
    out_sents = [str(sent.split('<eos>')[0]) for sent in out_sents]

    # print ('Fetched from KB:' + str(np.sum(out_sents[:, 1])))
    return out_sents


def get_sent(sent):
    #team_o = {}
    #fetched_from_kb = 0
    out_sent = []
    for idx in sent:
        w = chat_data.geti2w(idx)
        out_sent.append(w)

    return ' '.join(out_sent)


if __name__ == '__main__':
    train()
    test(model)