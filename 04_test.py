import os
import sys
import importlib
import argparse
from pathlib import Path
import csv
import numpy as np
import time
import pickle
import pathlib
import gzip

import tensorflow as tf
import tensorflow.contrib.eager as tfe

import svmrank

# from utilities import compute_extended_variable_features, preprocess_variable_features
import pickle

from utilities_tf import load_batch_gcnn


def preprocess_variable_features(features, interaction_augmentation, normalization):
    """
    Features preprocessing following Khalil et al. (2016) Learning to Branch in Mixed Integer Programming.

    Parameters
    ----------
    features : 2D np.ndarray
        The candidate variable features to preprocess.
    interaction_augmentation : bool
        Whether to augment features with 2-degree interactions (useful for linear models such as SVMs).
    normalization : bool
        Wether to normalize features in [0, 1] (i.e., query-based normalization).

    Returns
    -------
    variable_features : 2D np.ndarray
        The preprocessed variable features.
    """
    # 2-degree polynomial feature augmentation
    if interaction_augmentation:
        interactions = (
            np.expand_dims(features, axis=-1) * np.expand_dims(features, axis=-2)
        ).reshape((features.shape[0], -1))
        features = np.concatenate([features, interactions], axis=1)

    # query-based normalization in [0, 1]
    if normalization:
        features -= features.min(axis=0, keepdims=True)
        max_val = features.max(axis=0, keepdims=True)
        max_val[max_val == 0] = 1
        features /= max_val

    return features


def compute_extended_variable_features(state, candidates):
    """
    Utility to extract variable features only from a bipartite state representation.

    Parameters
    ----------
    state : dict
        A bipartite state representation.
    candidates: list of ints
        List of candidate variables for which to compute features (given as indexes).

    Returns
    -------
    variable_states : np.array
        The resulting variable states.
    """
    constraint_features, edge_features, variable_features = state
    constraint_features = constraint_features["values"]
    edge_indices = edge_features["indices"]
    edge_features = edge_features["values"]
    variable_features = variable_features["values"]

    cand_states = np.zeros(
        (
            len(candidates),
            variable_features.shape[1]
            + 3 * (edge_features.shape[1] + constraint_features.shape[1]),
        )
    )

    # re-order edges according to variable index
    edge_ordering = edge_indices[1].argsort()
    edge_indices = edge_indices[:, edge_ordering]
    edge_features = edge_features[edge_ordering]

    # gather (ordered) neighbourhood features
    nbr_feats = np.concatenate(
        [edge_features, constraint_features[edge_indices[0]]], axis=1
    )

    # split neighborhood features by variable, along with the corresponding variable
    var_cuts = np.diff(edge_indices[1]).nonzero()[0] + 1
    nbr_feats = np.split(nbr_feats, var_cuts)
    nbr_vars = np.split(edge_indices[1], var_cuts)
    assert all([all(vs[0] == vs) for vs in nbr_vars])
    nbr_vars = [vs[0] for vs in nbr_vars]

    # process candidate variable neighborhoods only
    for var, nbr_id, cand_id in zip(
        *np.intersect1d(nbr_vars, candidates, return_indices=True)
    ):
        cand_states[cand_id, :] = np.concatenate(
            [
                variable_features[var, :],
                nbr_feats[nbr_id].min(axis=0),
                nbr_feats[nbr_id].mean(axis=0),
                nbr_feats[nbr_id].max(axis=0),
            ]
        )

    cand_states[np.isnan(cand_states)] = 0

    return cand_states


def load_flat_samples(filename, feat_type, label_type, augment_feats, normalize_feats):
    with gzip.open(filename, "rb") as file:
        sample = pickle.load(file)

    state, khalil_state, best_cand, cands, cand_scores = sample["data"]

    cands = np.array(cands)
    cand_scores = np.array(cand_scores)

    cand_states = []
    if feat_type in ("all", "gcnn_agg"):
        cand_states.append(compute_extended_variable_features(state, cands))
    if feat_type in ("all", "khalil"):
        cand_states.append(khalil_state)
    cand_states = np.concatenate(cand_states, axis=1)

    best_cand_idx = np.where(cands == best_cand)[0][0]

    # feature preprocessing
    cand_states = preprocess_variable_features(
        cand_states,
        interaction_augmentation=augment_feats,
        normalization=normalize_feats,
    )

    if label_type == "scores":
        cand_labels = cand_scores

    elif label_type == "ranks":
        cand_labels = np.empty(len(cand_scores), dtype=int)
        cand_labels[cand_scores.argsort()] = np.arange(len(cand_scores))

    elif label_type == "bipartite_ranks":
        # scores quantile discretization as in
        # Khalil et al. (2016) Learning to Branch in Mixed Integer Programming
        cand_labels = np.empty(len(cand_scores), dtype=int)
        cand_labels[cand_scores >= 0.8 * cand_scores.max()] = 1
        cand_labels[cand_scores < 0.8 * cand_scores.max()] = 0

    else:
        raise ValueError(f"Invalid label type: '{label_type}'")

    return cand_states, cand_labels, best_cand_idx

def load_batch_flat(sample_files, feats_type, augment_feats, normalize_feats):
    cand_features = []
    cand_choices = []
    cand_scoress = []

    for i, filename in enumerate(sample_files):
        cand_states, cand_scores, cand_choice = load_flat_samples(filename, feats_type, 'scores', augment_feats, normalize_feats)

        cand_features.append(cand_states)
        cand_choices.append(cand_choice)
        cand_scoress.append(cand_scores)

    n_cands_per_sample = [v.shape[0] for v in cand_features]

    cand_features = np.concatenate(cand_features, axis=0).astype(np.float32, copy=False)
    cand_choices = np.asarray(cand_choices).astype(np.int32, copy=False)
    cand_scoress = np.concatenate(cand_scoress, axis=0).astype(np.float32, copy=False)
    n_cands_per_sample = np.asarray(n_cands_per_sample).astype(np.int32, copy=False)

    return cand_features, n_cands_per_sample, cand_choices, cand_scoress


def padding(output, n_vars_per_sample, fill=-1e8):
    n_vars_max = tf.reduce_max(n_vars_per_sample)

    output = tf.split(
        value=output,
        num_or_size_splits=n_vars_per_sample,
        axis=1,
    )
    output = tf.concat([
        tf.pad(
            x,
            paddings=[[0, 0], [0, n_vars_max - tf.shape(x)[1]]],
            mode='CONSTANT',
            constant_values=fill)
        for x in output
    ], axis=0)

    return output


def process(policy, dataloader, top_k):
    mean_kacc = np.zeros(len(top_k))

    n_samples_processed = 0
    for batch in dataloader:

        if policy['type'] == 'gcnn':
            c, ei, ev, v, n_cs, n_vs, n_cands, cands, best_cands, cand_scores = batch

            pred_scores = policy['model']((c, ei, ev, v, tf.reduce_sum(n_cs, keepdims=True), tf.reduce_sum(n_vs, keepdims=True)), tf.convert_to_tensor(False))

            # filter candidate variables
            pred_scores = tf.expand_dims(tf.gather(tf.squeeze(pred_scores, 0), cands), 0)

        elif policy['type'] == 'ml-competitor':
            cand_feats, n_cands, best_cands, cand_scores = batch

            # move to numpy
            cand_feats = cand_feats.numpy()
            n_cands = n_cands.numpy()

            # feature normalization
            cand_feats = (cand_feats - policy['feat_shift']) / policy['feat_scale']

            pred_scores = policy['model'].predict(cand_feats)

            # move back to TF
            pred_scores = tf.convert_to_tensor(pred_scores.reshape((1, -1)), dtype=tf.float32)

        # padding
        pred_scores = padding(pred_scores, n_cands)
        true_scores = padding(tf.reshape(cand_scores, (1, -1)), n_cands)
        true_bestscore = tf.reduce_max(true_scores, axis=-1, keepdims=True)

        assert all(true_bestscore.numpy() == np.take_along_axis(true_scores.numpy(), best_cands.numpy().reshape((-1, 1)), axis=1))

        kacc = []
        for k in top_k:
            pred_top_k = tf.nn.top_k(pred_scores, k=k)[1].numpy()
            pred_top_k_true_scores = np.take_along_axis(true_scores.numpy(), pred_top_k, axis=1)
            kacc.append(np.mean(np.any(pred_top_k_true_scores == true_bestscore.numpy(), axis=1)))
        kacc = np.asarray(kacc)

        batch_size = int(n_cands.shape[0])
        mean_kacc += kacc * batch_size
        n_samples_processed += batch_size

    mean_kacc /= n_samples_processed

    return mean_kacc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset', 'tsp'],
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--sourcedir',
        help='Source directory for the datasets.',
        type=str,
        default='./data/processed/',
    )
    parser.add_argument(
        '--seeds', 
        help='delimited seeds',
        type=str,
        defaut='0'
    )
    args = parser.parse_args()

    print(f"problem: {args.problem}")
    print(f"gpu: {args.gpu}")

    os.makedirs("results", exist_ok=True)
    result_file = f"results/{args.problem}_validation_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    #seeds = [0, 1, 2, 3, 4]
    seeds = [int(item) for item in args.seeds.split(',')]
    gcnn_models = ['baseline']
    other_models = ['extratrees_gcnn_agg', 'lambdamart_khalil', 'svmrank_khalil']
    test_batch_size = 128
    top_k = [1, 3, 5, 10]

    problem_folders = {
        'setcover': 'setcover/500r_1000c_0.05d',
        'cauctions': 'cauctions/100_500',
        'facilities': 'facilities/100_100_5',
        'indset': 'indset/500_4',
        'tsp': 'tsp/100',
    }
    problem_folder = problem_folders[args.problem]

    # if args.problem == 'setcover':
    #     gcnn_models += ['mean_convolution', 'no_prenorm']

    result_file = f"results/{args.problem}_test_{time.strftime('%Y%m%d-%H%M%S')}"

    result_file = result_file + '.csv'
    os.makedirs('results', exist_ok=True)

    ### TENSORFLOW SETUP ###
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.enable_eager_execution(config)
    tf.executing_eagerly()

    test_files = list(pathlib.Path(Path(args.sourcedir) / 'data/samples/{}/test'.format(problem_folder)).glob('sample_*.pkl'))
    test_files = [str(x) for x in test_files]

    print(f"{len(test_files)} test samples")

    evaluated_policies = [['gcnn', model] for model in gcnn_models] + \
            [['ml-competitor', model] for model in other_models]

    fieldnames = [
        'policy',
        'seed',
    ] + [
        f'acc@{k}' for k in top_k
    ]
    with open(result_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for policy_type, policy_name in evaluated_policies:
            print(f"{policy_type}:{policy_name}...")
            for seed in seeds:
                rng = np.random.RandomState(seed)
                tf.set_random_seed(rng.randint(np.iinfo(int).max))

                policy = {}
                policy['name'] = policy_name
                policy['type'] = policy_type

                if policy['type'] == 'gcnn':
                    # load model
                    sys.path.insert(0, os.path.abspath(f"models/{policy['name']}"))
                    import model
                    importlib.reload(model)
                    del sys.path[0]
                    policy['model'] = model.GCNPolicy()
                    policy['model'].restore_state(f"trained_models/{args.problem}/{policy['name']}/{seed}/best_params.pkl")
                    policy['model'].call = tfe.defun(policy['model'].call, input_signature=policy['model'].input_signature)
                    policy['batch_datatypes'] = [tf.float32, tf.int32, tf.float32,
                            tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.float32]
                    policy['batch_fun'] = load_batch_gcnn
                else:
                    # load feature normalization parameters
                    try:
                        with open(f"trained_models/{args.problem}/{policy['name']}/{seed}/normalization.pkl", 'rb') as f:
                            policy['feat_shift'], policy['feat_scale'] = pickle.load(f)
                    except:
                            policy['feat_shift'], policy['feat_scale'] = 0, 1

                    # load model
                    if policy_name.startswith('svmrank'):
                        policy['model'] = svmrank.Model().read(f"trained_models/{args.problem}/{policy['name']}/{seed}/model.txt")
                    else:
                        with open(f"trained_models/{args.problem}/{policy['name']}/{seed}/model.pkl", 'rb') as f:
                            policy['model'] = pickle.load(f)

                    # load feature specifications
                    with open(f"trained_models/{args.problem}/{policy['name']}/{seed}/feat_specs.pkl", 'rb') as f:
                        feat_specs = pickle.load(f)

                    policy['batch_datatypes'] = [tf.float32, tf.int32, tf.int32, tf.float32]
                    policy['batch_fun'] = lambda x: load_batch_flat(x, feat_specs['type'], feat_specs['augment'], feat_specs['qbnorm'])

                test_data = tf.data.Dataset.from_tensor_slices(test_files)
                test_data = test_data.batch(test_batch_size)
                test_data = test_data.map(lambda x: tf.py_func(
                    policy['batch_fun'], [x], policy['batch_datatypes']))
                test_data = test_data.prefetch(2)

                test_kacc = process(policy, test_data, top_k)
                print(f"  {seed} " + " ".join([f"acc@{k}: {100*acc:4.1f}" for k, acc in zip(top_k, test_kacc)]))

                writer.writerow({
                    **{
                        'policy': f"{policy['type']}:{policy['name']}",
                        'seed': seed,
                    },
                    **{
                        f'acc@{k}': test_kacc[i] for i, k in enumerate(top_k)
                    },
                })
                csvfile.flush()
