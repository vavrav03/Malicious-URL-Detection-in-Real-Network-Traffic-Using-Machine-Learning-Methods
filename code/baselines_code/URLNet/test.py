import pickle 
from time import perf_counter
from tqdm import tqdm
import argparse
import numpy as np 
import pickle 
import tensorflow.compat.v1 as tf

from .utils import * 

tf.disable_v2_behavior()


# data args
def get_default_test_args():
    parser = argparse.ArgumentParser(description="Test URLNet model")
    default_max_len_words = 200
    parser.add_argument('--data.max_len_words', type=int, default=default_max_len_words, metavar="MLW",
    help="maximum length of url in words (default: {})".format(default_max_len_words))
    default_max_len_chars = 200
    parser.add_argument('--data.max_len_chars', type=int, default=default_max_len_chars, metavar="MLC",
    help="maximum length of url in characters (default: {})".format(default_max_len_chars))
    default_max_len_subwords = 20
    parser.add_argument('--data.max_len_subwords', type=int, default=default_max_len_subwords, metavar="MLSW",
    help="maxium length of word in subwords/ characters (default: {})".format(default_max_len_subwords))
    default_delimit_mode = 1
    parser.add_argument("--data.delimit_mode", type=int, default=default_delimit_mode, metavar="DLMODE",
    help="0: delimit by special chars, 1: delimit by special chars + each char as a word (default: {})".format(default_delimit_mode))

    # model args 
    default_emb_dim = 32
    parser.add_argument('--model.emb_dim', type=int, default=default_emb_dim, metavar="EMBDIM",
    help="embedding dimension size (default: {})".format(default_emb_dim))
    default_emb_mode = 1
    parser.add_argument('--model_emb_mode', type=int, default=default_emb_mode, metavar="EMBMODE",
    help="1: charCNN, 2: wordCNN, 3: char + wordCNN, 4: char-level wordCNN, 5: char + char-level wordCNN (default: {})".format(default_emb_mode))

    # test args 
    default_batch_size = 128
    parser.add_argument('--test.batch_size', type=int, default=default_batch_size, metavar="BATCHSIZE",
    help="Size of each test batch (default: {})".format(default_batch_size))

    return parser.parse_args([])

def run_test(df, args, ngram_dict, word_dict, chars_dict, sess):
    urls = df["url"].tolist()
    labels = df["label"].tolist()
    FLAGS = vars(args)
    for key, val in FLAGS.items():
        print("{}={}".format(key, val))
    
    x, word_reverse_dict = get_word_vocab(urls, FLAGS["data.max_len_words"]) 
    word_x = get_words(x, word_reverse_dict, FLAGS["data.delimit_mode"], urls) 

    print("Size of subword vocabulary (train): {}".format(len(ngram_dict)))
    print("size of word vocabulary (train): {}".format(len(word_dict)))
    ngramed_id_x, worded_id_x = ngram_id_x_from_dict(word_x, FLAGS["data.max_len_subwords"], ngram_dict, word_dict) 
    chared_id_x = char_id_x(urls, chars_dict, FLAGS["data.max_len_chars"])    

    print("Number of testing urls: {}".format(len(labels)))

    ######################## EVALUATION ########################### 

    def test_step(x, emb_mode):
        p = 1.0
        if emb_mode == 1: 
            feed_dict = {
                input_x_char_seq: x[0],
                dropout_keep_prob: p}  
        elif emb_mode == 2: 
            feed_dict = {
                input_x_word: x[0],
                dropout_keep_prob: p}
        elif emb_mode == 3: 
            feed_dict = {
                input_x_char_seq: x[0],
                input_x_word: x[1],
                dropout_keep_prob: p}
        elif emb_mode == 4: 
            feed_dict = {
                input_x_word: x[0],
                input_x_char: x[1],
                input_x_char_pad_idx: x[2],
                dropout_keep_prob: p}
        elif emb_mode == 5:  
            feed_dict = {
                input_x_char_seq: x[0],
                input_x_word: x[1],
                input_x_char: x[2],
                input_x_char_pad_idx: x[3],
                dropout_keep_prob: p}
        preds, s = sess.run([predictions, scores], feed_dict)
        return preds, s

    # graph = tf.Graph() 
    # with graph.as_default(): 
        # session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        # session_conf.gpu_options.allow_growth=True 
        # sess = tf.Session(config=session_conf)
        # with sess.as_default(): 
            # saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            # saver.restore(sess, checkpoint_file) 
    graph = sess.graph
    if  FLAGS["model_emb_mode"] in [1, 3, 5]: 
        input_x_char_seq = graph.get_operation_by_name("input_x_char_seq").outputs[0]
    if FLAGS["model_emb_mode"] in [2, 3, 4, 5]:
        input_x_word = graph.get_operation_by_name("input_x_word").outputs[0]
    if FLAGS["model_emb_mode"] in [4, 5]:
        input_x_char = graph.get_operation_by_name("input_x_char").outputs[0]
        input_x_char_pad_idx = graph.get_operation_by_name("input_x_char_pad_idx").outputs[0]
    dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0] 

    predictions = graph.get_operation_by_name("output/predictions").outputs[0]
    scores = graph.get_operation_by_name("output/scores").outputs[0]
    
    if FLAGS["model_emb_mode"] == 1: 
        batches = batch_iter(list(chared_id_x), FLAGS["test.batch_size"], 1, shuffle=False) 
    elif FLAGS["model_emb_mode"] == 2: 
        batches = batch_iter(list(worded_id_x), FLAGS["test.batch_size"], 1, shuffle=False) 
    elif FLAGS["model_emb_mode"] == 3: 
        batches = batch_iter(list(zip(chared_id_x, worded_id_x)), FLAGS["test.batch_size"], 1, shuffle=False)
    elif FLAGS["model_emb_mode"] == 4: 
        batches = batch_iter(list(zip(ngramed_id_x, worded_id_x)), FLAGS["test.batch_size"], 1, shuffle=False)
    elif FLAGS["model_emb_mode"] == 5: 
        batches = batch_iter(list(zip(ngramed_id_x, worded_id_x, chared_id_x)), FLAGS["test.batch_size"], 1, shuffle=False)    
    all_predictions = []
    all_scores = []
    
    nb_batches = int(len(labels) / FLAGS["test.batch_size"])
    if len(labels) % FLAGS["test.batch_size"] != 0: 
        nb_batches += 1 
    print("Number of batches in total: {}".format(nb_batches))
    it = tqdm(range(nb_batches), desc="emb_mode {} delimit_mode {} test_size {}".format(FLAGS["model_emb_mode"], FLAGS["data.delimit_mode"], len(labels)), ncols=0)
    start_time = perf_counter()
    start_time_batched = 0.0
    for idx in it:
    #for batch in batches:
        batch = next(batches)

        if FLAGS["model_emb_mode"] == 1: 
            x_char_seq = batch 
        elif FLAGS["model_emb_mode"] == 2: 
            x_word = batch 
        elif FLAGS["model_emb_mode"] == 3: 
            x_char_seq, x_word = zip(*batch) 
        elif FLAGS["model_emb_mode"] == 4: 
            x_char, x_word = zip(*batch)
        elif FLAGS["model_emb_mode"] == 5: 
            x_char, x_word, x_char_seq = zip(*batch)            

        x_batch = []    
        if FLAGS["model_emb_mode"] in[1, 3, 5]: 
            x_char_seq = pad_seq_in_word(x_char_seq, FLAGS["data.max_len_chars"]) 
            x_batch.append(x_char_seq)
        if FLAGS["model_emb_mode"] in [2, 3, 4, 5]:
            x_word = pad_seq_in_word(x_word, FLAGS["data.max_len_words"]) 
            x_batch.append(x_word)
        if FLAGS["model_emb_mode"] in [4, 5]:
            x_char, x_char_pad_idx = pad_seq(x_char, FLAGS["data.max_len_words"], FLAGS["data.max_len_subwords"], FLAGS["model.emb_dim"])
            x_batch.extend([x_char, x_char_pad_idx])
        
        batch_start = perf_counter()
        batch_predictions, batch_scores = test_step(x_batch, FLAGS["model_emb_mode"])            
        batch_end = perf_counter()
        start_time_batched += (batch_end - batch_start)
        all_predictions = np.concatenate([all_predictions, batch_predictions])
        all_scores.extend(batch_scores) 

        it.set_postfix()
    end_time = perf_counter()
    total_time = end_time - start_time
    n_s_total = len(labels) / total_time
    n_s_batched = len(labels) / start_time_batched
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    total_params = 0
    for v in trainable_vars:
        shape = v.get_shape().as_list()
        param_count = 1
        for dim in shape:
            param_count *= dim
        total_params += param_count
    return {
        "targets": np.array(labels),
        "probabilities": np.array([softmax(i) for i in all_scores]),
        "predictions": np.array(all_predictions),
        "n_per_s_total": n_s_total,
        "n_per_s_batched": n_s_batched,
        "params_count": total_params
    }