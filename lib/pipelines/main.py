#!/usr/bin/env python3

import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
warnings.simplefilter('ignore', category=DeprecationWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

import os
import argparse
import multiprocessing
import artm_integration
import btopic_integration
import sbert_integration
import naive_integration
import summary_integration
from hydra import compose, initialize


def load_config(args):
    initialize(version_base=None, config_path='configs')
    arg_dict = vars(args)
    overrides = [f"{key}={val}" for key, val in arg_dict.items() if val is not None]
    cfg = compose(config_name="config", overrides=overrides)
    return cfg


def args_parse():
    parser = argparse.ArgumentParser(
        description='Code for running segmentation pipelines with different models:'
                    ' ARTM, BERTopic, sentence BERT, naive')
    subparsers = parser.add_subparsers(title='Available models', dest='subcommand', required=True)

    artm_parser = subparsers.add_parser('artm', help='segmentation with use of ARTM model')
    artm_subparsers = artm_parser.add_subparsers(title='artm subcommands', dest='artm_subcommand', required=True)

    vectorize_parser = artm_subparsers.add_parser('vectorize', help='Vectorize data for ARTM model'
                                                                    ' and save it in ARTM required format')
    vectorize_parser.add_argument('--input_path', '-i', type=str, help='Path to input dataset', required=True)
    vectorize_parser.add_argument('--output_path', '-o', type=str, help='Path to output directory where all'
                                                                        ' processed files will be saved', required=True)
    vectorize_parser.add_argument('--collection', '-c', type=str, required=True,
                                  help='Collection name (necessary for ARTM model)')
    vectorize_parser.add_argument('--dictionary', '-d', type=str, default=None,
                                  help='Path to the words dict(uci format),'
                                       ' should be used only with test path of the dataset')
    vectorize_parser.add_argument('--dataset_type', '-dt', type=str, required=True, help='Dataset type (i.e. wiki or ami)')

    train_parser = artm_subparsers.add_parser('train', help='Train model with use of ARTM vectorized dataset')
    train_parser.add_argument('--output_path', '-o', type=str, help='Output path for the model dump', required=True)
    train_parser.add_argument('--input_path', '-i', type=str, required=True,
                              help='Input path to ARTM batched vectorized dataset')
    train_parser.add_argument('--dataset_type', '-dt', type=str, required=True, help='Dataset type (i.e. wiki or ami)')

    predict_parser = artm_subparsers.add_parser('predict', help='Predict segments with trained model'
                                                                ' and vectorized dataset')
    predict_parser.add_argument('--model_path', '-m', type=str, help='Path to the trained model', required=True)
    predict_parser.add_argument('--input_path', '-i', required=True, type=str,
                                help='Input to the data for prediction, use folder from Vectorized stage')
    predict_parser.add_argument('--dictionary', '-d', type=str, required=True,
                                help='Path to the word dict (from train stage, in artm format)')
    predict_parser.add_argument('--output_path', '-o', type=str, help='Output path for the predicts', required=True)
    predict_parser.add_argument('--dataset_type', '-dt', type=str, required=True, help='Dataset type (i.e. wiki or ami)')

    scores_parser = artm_subparsers.add_parser('scores', help='get pk and windowdiff scores for calculated predict')
    scores_parser.add_argument('--predicts_path', '-p', type=str, help='Path to the file with the borders predicts',
                               required=True)
    scores_parser.add_argument('--input_path', '-i', required=True, type=str,
                                help='Path to the dataset in the raw format(not vectorized)')
    scores_parser.add_argument('--dataset_type', '-dt', type=str, required=True, help='Dataset type (i.e. wiki or ami)')

    # bertopic
    btopic_parser = subparsers.add_parser('btopic', help='Segment text using BERTopic neural mode '
                                                         'and get pk and window diff scores')
    btopic_parser.add_argument('--train_path', type=str, required=True,
                               help='Path to the embedded dataset (could be created with sBERT model)')
    btopic_parser.add_argument('--val_path', type=str, required=True,
                               help='Path to the embedded dataset (could be created with sBERT model)')
    btopic_parser.add_argument('--test_path', type=str, required=True,
                               help='Path to the embedded dataset (could be created with sBERT model)')
    btopic_parser.add_argument('--dataset_type', '-dt', type=str, required=True, help='Dataset type (i.e. wiki or ami)')

    # sbert
    sbert_parser = subparsers.add_parser('sbert', help='Segmentation and embedding calculation with sentence bert')
    sbert_subparsers = sbert_parser.add_subparsers(title='sbert subcommands', dest='sbert_subcommand', required=True)
    
    embed_parser = sbert_subparsers.add_parser('embed', help='Embed dataset using sentence bert')
    embed_parser.add_argument('--input_path', '-i', type=str, required=True, help='Path to the dataset')
    embed_parser.add_argument('--output_path', '-o', type=str, required=True,
                              help='Full path where to save embedded dataset')
    embed_parser.add_argument('--dataset_type', '-dt', type=str, required=True, help='Dataset type (i.e. wiki or ami)')
    
    sbert_scores_parser = sbert_subparsers.add_parser('scores', help='Make segmentation and calculate pk'
                                                                     ' and window diff scores')
    sbert_scores_parser.add_argument('--val_path', '-v', type=str, required=True, help='Path to the embedded dataset')
    sbert_scores_parser.add_argument('--test_path', '-t', type=str, required=True, help='Path to the embedded dataset')
    sbert_scores_parser.add_argument('--dataset_type', '-dt', type=str, required=True, help='Dataset type (i.e. wiki or ami)')
    sbert_scores_parser.add_argument('--clustering', action='store_true', help='Whether to do a clustering or a tiling on embeddings')

    # naive
    naive_parser = subparsers.add_parser('naive', help='Segment text using naive mode and get pk, window diff and f1 scores')
    naive_parser.add_argument('--input_path', '-i', type=str, required=True, help='Path to the dataset')
    naive_parser.add_argument('--dataset_type', '-dt', type=str, required=True, help='Dataset type (i.e. wiki or ami)')

    # summary
    sumseg_parser = subparsers.add_parser('sumseg', help='Segment text using summarization pipeline and get pk, window diff and f1 scores')
    sumseg_subparsers = sumseg_parser.add_subparsers(title='sumseg subcommands', dest='sumseg_subcommand', required=True)
    
    summarize_parser = sumseg_subparsers.add_parser('summarize', help='Summarize dataset')
    summarize_parser.add_argument('--input_path', '-i', type=str, required=True, help='Path to the dataset')
    summarize_parser.add_argument('--output_path', '-o', type=str, required=True, help='Full path where to save embedded dataset')
    summarize_parser.add_argument('--dataset_type', '-dt', type=str, required=True, help='Dataset type (i.e. wiki or ami)')
    sumseg_scores_parser = sumseg_subparsers.add_parser('scores', help='Make segmentation and calculate pk and window diff scores')
    sumseg_scores_parser.add_argument('--val_path', '-v', type=str, required=True, help='Path to the embedded dataset')
    sumseg_scores_parser.add_argument('--test_path', '-t', type=str, required=True, help='Path to the embedded dataset')
    sumseg_scores_parser.add_argument('--dataset_type', '-dt', type=str, required=True, help='Dataset type (i.e. wiki or ami)')
    
    args = parser.parse_args()
    return args


def main():
    args = args_parse()
    cfg = load_config(args)
    cuda_devices = list(map(str, cfg.gpus))
    max_cores = multiprocessing.cpu_count()
    cpu_cores = min(max_cores, cfg.num_cpu)
    if cpu_cores != cfg.num_cpu:
        cfg.num_cpu = cpu_cores
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cuda_devices)
    os.environ["OMP_NUM_THREADS"] = str(cpu_cores)

    if args.subcommand == 'artm':
        if args.artm_subcommand == 'vectorize':
            artm_integration.vectorize_dataset(cfg)
        elif args.artm_subcommand == 'train':
            artm_integration.train_and_dump(cfg)
        elif args.artm_subcommand == 'predict':
            artm_integration.load_and_predict(cfg)
        elif args.artm_subcommand == 'scores':
            artm_integration.calculate_scores(cfg)
    elif args.subcommand == 'btopic':
        btopic_integration.get_scores_bert_topic(cfg)
    elif args.subcommand == 'sbert':
        if args.sbert_subcommand == 'embed':
            sbert_integration.embed_and_save(cfg)
        elif args.sbert_subcommand == 'scores':
            sbert_integration.get_scores_sentence_bert(cfg)
    elif args.subcommand == 'naive':
        naive_integration.get_scores_naive(cfg)
    elif args.subcommand == 'sumseg':
        if args.sumseg_subcommand == 'summarize':
            summary_integration.summarize_and_save(cfg)
        elif args.sumseg_subcommand == 'scores':
            summary_integration.get_scores_summary(cfg)


if __name__ == '__main__':
    main()
