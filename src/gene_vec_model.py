import numpy as np
import tensorflow as tf
import gc
import os
import shutil
import pickle
from utils.random_walk import random_walk_w_restart as rwr
from utils.sampling_util import rw_sampling
from tensorflow.python.keras import backend as K
from tensorflow.keras import layers, losses
from tensorflow import keras
from tensorflow.keras.models import Model
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train gene embeddings')
    parser.add_argument('--datatype', default='go',
                        help='Type of data from which the gene embeddings learned. go or archs4')
    parser.add_argument('--association_file', default='../data/go_gene_association.txt',
                        help='Path to a file containing GO term gene associations or ARCHS4 experiments gene associations')
    parser.add_argument('--outfile', default='../data/gene_vec_go.csv',
                        help='Path to save the learned embeddings')
    parser.add_argument('--LR_model', default=None,
                        help='Path to a directory to save or load the logistic regression model')
    parser.add_argument('--demo', default=None,
                        help='Path to the demo directory to export the learned embeddings into')
    return parser.parse_args()

def get_model(vocab_size, latent_dim, ori_dim):
    encoder_input_l = keras.Input(shape=(1,), name="geneidx_l")
    encoder_input_r = keras.Input(shape=(1,), name="geneidx_r")
    encoderlayers = layers.Embedding(vocab_size, latent_dim, input_length=1)
    encoder_output_l = keras.backend.squeeze(encoderlayers(encoder_input_l), axis=1)
    encoder_output_r = keras.backend.squeeze(encoderlayers(encoder_input_r), axis=1)

    encoder = keras.Model(inputs=encoder_input_l, outputs=encoder_output_l, name="encoder")
    encoder.summary()

    logit = layers.Dot(axes=1)([encoder_output_l, encoder_output_r])

    model = keras.Model(inputs=[encoder_input_l, encoder_input_r], outputs=[logit], name="autoencoder")
    model.summary()

    return encoder, model

def build_pair_features(encoded_genes, left_idx, right_idx):
    '''Build feature vectors for gene pairs from the learned embeddings.

    The feature for a pair is the element-wise product of the two genes'
    embedding vectors. ``left_idx`` and ``right_idx`` are arrays of gene
    indices into ``encoded_genes``.
    '''
    encoded_genes = np.asarray(encoded_genes)
    left_idx = np.asarray(left_idx, dtype=int)
    right_idx = np.asarray(right_idx, dtype=int)
    return encoded_genes[left_idx] * encoded_genes[right_idx]

def train_or_load_lr_model(X, y, lr_dir, filename='gene_pair_lr.pkl'):
    '''Train, or load if already present, a logistic regression model.

    The model predicts whether a pair of genes is associated based on the
    gene-pair features ``X`` and labels ``y``. It is persisted as a pickle
    file inside ``lr_dir``; if that file already exists it is loaded instead
    of being retrained. Returns a tuple ``(model, model_path)``.
    '''
    from sklearn.linear_model import LogisticRegression

    os.makedirs(lr_dir, exist_ok=True)
    lr_path = os.path.join(lr_dir, filename)

    if os.path.exists(lr_path):
        with open(lr_path, 'rb') as f:
            lr = pickle.load(f)
        print('Loaded LR model from', lr_path)
    else:
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X, y)
        with open(lr_path, 'wb') as f:
            pickle.dump(lr, f)
        print('Saved LR model to', lr_path)

    return lr, lr_path

def export_embeddings_for_demo(outfile, demo_dir):
    '''Copy the learned embeddings file into the demo directory.

    When ``demo_dir`` contains a ``data`` subdirectory the file is placed
    there (where the demo scripts look for it), otherwise it is copied into
    ``demo_dir`` directly. Returns the destination path.
    '''
    demo_data_dir = os.path.join(demo_dir, 'data')
    dest_dir = demo_data_dir if os.path.isdir(demo_data_dir) else demo_dir
    os.makedirs(dest_dir, exist_ok=True)
    demo_outfile = os.path.join(dest_dir, os.path.basename(outfile))
    shutil.copyfile(outfile, demo_outfile)
    print('Copied embeddings for demo to', demo_outfile)
    return demo_outfile

def main():
    args = parse_args()
    datatype, association_file, outfile = args.datatype, args.association_file, args.outfile
    LR_model, demo = args.LR_model, args.demo

    geneids, association_mat, diffusion_state = rwr(datatype, association_file)
    nonzeroidx = np.where(np.sum(association_mat, axis=1) > 0)[0]

    geneids = np.array(geneids)
    fp_array = association_mat[nonzeroidx]
    geneids = geneids[nonzeroidx]

    ori_dim = fp_array.shape[-1]
    latent_dim = 256

    encoder, autoencoder = get_model(len(geneids), latent_dim, ori_dim)
    optimizer = keras.optimizers.Adam()
    autoencoder.compile(
        optimizer=optimizer,
        loss=[losses.BinaryCrossentropy(from_logits=True)]
    )

    rs = rw_sampling(diffusion_state)
    epochs = 3000
    for e in range(epochs):
        print('epoch:', e)
        left_input = []
        right_input = []
        targets = []
        pos_pairs = 1
        neg_pairs = 5
        #sample training pairs
        left_input, right_input, targets = rs.sampling(np.arange(len(geneids)), pos_pairs, neg_pairs)

        left_input = np.squeeze(np.array(left_input))
        right_input = np.squeeze(np.array(right_input))
        targets = np.squeeze(np.array(targets))
        sample_size = len(left_input)
        smaple_idx = np.arange(sample_size)
        np.random.shuffle(smaple_idx)
        left_input = left_input[smaple_idx]
        right_input = right_input[smaple_idx]
        targets = targets[smaple_idx]

        autoencoder.fit([left_input[:int(0.8*sample_size)] * 1.0, right_input[:int(0.8*sample_size)] * 1.0], [targets[:int(0.8*sample_size)]],
                  batch_size=1000,
                  epochs=1,
                  verbose=1,
                  validation_data=([left_input[int(0.8*sample_size):] * 1.0, right_input[int(0.8*sample_size):] * 1.0], [targets[int(0.8*sample_size):]]))
        gc.collect()

    encoded_genes = encoder.predict(np.arange(len(geneids)) * 1.0)
    fw = open(outfile, 'w')
    for i in range(len(encoded_genes)):
        fw.write(geneids[i])
        for j in range(len(encoded_genes[0])):
            fw.write(',' + str(encoded_genes[i, j]))
        fw.write('\n')
    fw.close()

    # Train (or load) a logistic regression model that predicts whether a pair
    # of genes is associated, using the learned gene embeddings as features.
    # The model is persisted in the directory given by --LR_model.
    if LR_model is not None:
        X_lr = build_pair_features(encoded_genes, left_input, right_input)
        y_lr = np.asarray(targets, dtype=int)
        lr, _ = train_or_load_lr_model(X_lr, y_lr, LR_model)
        print('LR gene-pair prediction accuracy:', lr.score(X_lr, y_lr))

    # Export a copy of the learned embeddings into the demo directory so that
    # the demo scripts (e.g. classifier.py) can consume the trained embeddings.
    if demo is not None:
        export_embeddings_for_demo(outfile, demo)

if __name__ == '__main__':
    main()
