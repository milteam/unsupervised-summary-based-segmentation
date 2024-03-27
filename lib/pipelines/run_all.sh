#!/usr/bin/env bash

RUN_NAIVE=false
RUN_SBERT=false
RUN_BTOPIC=false
RUN_ARTM=false
RUN_TILING=false
RUN_SUM=true

ROOT_PATH="/data/shared/datasets/nlp/"

# Wiki727k
WIKI727K_FULL_PATH="${ROOT_PATH}wiki_727/"
WIKI727K_TRAIN_PATH="${ROOT_PATH}wiki_727/train/"
WIKI727K_TEST_PATH="${ROOT_PATH}wiki_727/test/"

WIKI50_PATH="../data/wiki_test_50/"

# AMI
AMI_FULL_PATH="${ROOT_PATH}ami-corpus/topics/"  # in this folder there is no division on folders train and test, only files
AMI_PATH="${ROOT_PATH}ami-corpus/splits/"  # in this folder there are two subfolders: train and test

AMI_TRAIN_PATH="../data/ami/train/"
AMI_VAL_PATH="../data/ami/dev/"
AMI_TEST_PATH="../data/ami/test/"

AMI_K_FOLDS_PATH="${ROOT_PATH}ami-corpus/ami_5_folds/"

# DialSeg
DIALSEG711_PATH="../data/dialseg711.txt"
DIALSEG711_VAL_PATH="../data/dialseg711-val.txt"
DIALSEG711_TEST_PATH="../data/dialseg711-test.txt"
DIALSEG711_SMALL_SAMPLE_PATH="../data/dialseg711_small_sample.txt"

# Doc2Dial
DOC2DIAL_PATH="../data/doc2dial_v1"

# TIAGE
TIAGE_TRAIN_PATH="../data/tiage/segmentation_file_train.txt"
TIAGE_VAL_PATH="../data/tiage/segmentation_file_validation.txt"
TIAGE_TEST_PATH="../data/tiage/segmentation_file_test.txt"

# SuperDialSeg
SUPERDIALSEG_TRAIN_PATH="../data/superseg/segmentation_file_train.txt"
SUPERDIALSEG_VAL_PATH="../data/superseg/segmentation_file_validation.txt"
SUPERDIALSEG_TEST_PATH="../data/superseg/segmentation_file_test.txt"

# QMSum
QMSUM_TRAIN_PATH="../data/qmsum/train.jsonl"
QMSUM_VAL_PATH="../data/qmsum/val.jsonl"
QMSUM_TEST_PATH="../data/qmsum/test.jsonl"

# WikiSection
WIKI_SECTION_TRAIN_PATH="../data/wikisection_dataset_json/train/wikisection_en_disease_train.json"
WIKI_SECTION_VAL_PATH="../data/wikisection_dataset_json/val/wikisection_en_disease_validation.json"
WIKI_SECTION_TEST_PATH="../data/wikisection_dataset_json/test/wikisection_en_disease_test.json"


if $RUN_NAIVE; then
    printf 'Running naive pipeline...\n'
    # nice bash ./run_naive.sh $WIKI50_PATH wiki

    # for i in {1..5}
    # do
    #     nice bash ./run_naive.sh ${AMI_K_FOLDS_PATH}fold${i}/test ami
    # done

    # nice bash ./run_naive.sh ${DIALSEG711_VAL_PATH} dialseg
    # nice bash ./run_naive.sh ${DIALSEG711_TEST_PATH} dialseg

    # nice bash ./run_naive.sh ${DOC2DIAL_PATH} doc2dial

    # printf 'TRAIN DATASETS\n'
    # nice bash ./run_naive.sh ${AMI_TRAIN_PATH} ami
    # nice bash ./run_naive.sh ${TIAGE_TRAIN_PATH} tiage
    # nice bash ./run_naive.sh ${SUPERDIALSEG_TRAIN_PATH} superdialseg
    # nice bash ./run_naive.sh ${QMSUM_TRAIN_PATH} qmsum
    
    # printf 'VAL DATASETS\n'
    # nice bash ./run_naive.sh ${AMI_VAL_PATH} ami
    # nice bash ./run_naive.sh ${TIAGE_VAL_PATH} tiage
    # nice bash ./run_naive.sh ${SUPERDIALSEG_VAL_PATH} superdialseg
    # nice bash ./run_naive.sh ${QMSUM_VAL_PATH} qmsum

    # printf 'TEST DATASETS\n'
    # nice bash ./run_naive.sh $AMI_TEST_PATH ami
    # nice bash ./run_naive.sh $TIAGE_TEST_PATH tiage
    # nice bash ./run_naive.sh $SUPERDIALSEG_TEST_PATH superdialseg
    # nice bash ./run_naive.sh $QMSUM_TEST_PATH qmsum
    nice bash ./run_naive.sh $WIKI_SECTION_VAL_PATH wikisection
    
    printf 'Naive pipeline is done.\n\n'
fi

if $RUN_SBERT; then
    printf 'Running sentence-bert pipeline...\n'
    # nice bash ./run_sbert.sh $WIKI50_PATH wiki
    CUDA_VISIBLE_DEVICES=6 nice bash ./run_sbert.sh $AMI_FULL_PATH ami
    printf 'Sentence-bert pipeline is done.\n\n'
fi

if $RUN_BTOPIC; then
    printf 'Running BERTopic pipeline...\n'
    # CUDA_VISIBLE_DEVICES=7 nice bash ./run_btopic.sh $AMI_TRAIN_PATH $AMI_TEST_PATH ami
    # CUDA_VISIBLE_DEVICES=7 nice bash ./run_btopic.sh ${AMI_K_FOLDS_PATH}fold1/train ${AMI_K_FOLDS_PATH}fold1/test ami

    # for i in {1..5}
    # do
    #     CUDA_VISIBLE_DEVICES=7 nice bash ./run_btopic.sh ${AMI_K_FOLDS_PATH}fold${i}/train ${AMI_K_FOLDS_PATH}fold${i}/test ami
    # done

    # CUDA_VISIBLE_DEVICES=7 nice bash ./run_btopic.sh $AMI_FULL_PATH $AMI_FULL_PATH ami
    # CUDA_VISIBLE_DEVICES=6 nice python main.py btopic --train_path /data/shared/datasets/nlp/wiki727train_embedded.hf/ --test_path /data/shared/datasets/nlp/wiki727test_embedded.hf/ -dt wiki

    CUDA_VISIBLE_DEVICES=6 nice bash run_btopic.sh $AMI_TRAIN_PATH $AMI_VAL_PATH $AMI_TEST_PATH ami > bt-ami.txt
    CUDA_VISIBLE_DEVICES=6 nice bash run_btopic.sh $TIAGE_TRAIN_PATH $TIAGE_VAL_PATH $TIAGE_TEST_PATH tiage > bt-tiage.txt
    CUDA_VISIBLE_DEVICES=6 nice bash run_btopic.sh $SUPERDIALSEG_TRAIN_PATH $SUPERDIALSEG_VAL_PATH $SUPERDIALSEG_TEST_PATH superdialseg > bt-superdialseg.txt
    CUDA_VISIBLE_DEVICES=6 nice bash run_btopic.sh $QMSUM_TRAIN_PATH $QMSUM_VAL_PATH $QMSUM_TEST_PATH qmsum > bt-qmsum.txt

    printf 'BERTopic pipeline is done.\n\n'
fi

if $RUN_ARTM; then
    printf 'Running BigARTM pipeline...\n'
    # nice bash ./run_artm.sh $AMI_PATH ami
    for i in {1..5}
    do
        nice bash ./run_artm.sh ${AMI_K_FOLDS_PATH}fold${i} ami
    done

    # nice bash ./run_artm.sh $WIKI727K_FULL_PATH wiki
    printf 'BigARTM pipeline is done.\n\n'
fi

if $RUN_TILING; then
    printf 'Running TopicTiling on sentence-bert embeddings pipeline...\n'
    # nice bash ./run_tiling.sh $AMI_FULL_PATH ami
    # for i in {1..5}
    # do
    #     CUDA_VISIBLE_DEVICES=7 nice bash ./run_tiling.sh ${AMI_K_FOLDS_PATH}fold${i}/test ami
    # done

    # nice python main.py sbert scores -i /data/shared/datasets/nlp/wiki727test_embedded.hf/ -dt wiki
    # CUDA_VISIBLE_DEVICES=7 nice bash ./run_tiling.sh $DIALSEG711_VAL_PATH $DIALSEG711_TEST_PATH dialseg > tt-dialseg.txt

    # CUDA_VISIBLE_DEVICES=6 nice bash run_tiling.sh $AMI_VAL_PATH $AMI_TEST_PATH ami > tt-ami.txt
    # CUDA_VISIBLE_DEVICES=6 nice bash run_tiling.sh $TIAGE_VAL_PATH $TIAGE_TEST_PATH tiage > tt-tiage.txt
    # CUDA_VISIBLE_DEVICES=6 nice bash run_tiling.sh $SUPERDIALSEG_VAL_PATH $SUPERDIALSEG_TEST_PATH superdialseg > tt-superdialseg.txt
    # CUDA_VISIBLE_DEVICES=6 nice bash run_tiling.sh $QMSUM_VAL_PATH $QMSUM_TEST_PATH qmsum > tt-qmsum.txt

    # CUDA_VISIBLE_DEVICES=0 nice bash ./run_tiling.sh $DIALSEG711_VAL_PATH $DIALSEG711_TEST_PATH dialseg

    nice bash ./run_tiling.sh $WIKI_SECTION_VAL_PATH $WIKI_SECTION_TEST_PATH wikisection

    printf 'TopicTiling pipeline is done.\n\n'
fi

if $RUN_SUM; then
    printf 'Running summarization pipeline...\n'
    CUDA_VISIBLE_DEVICES=2 nice bash run_sumseg.sh $WIKI_SECTION_VAL_PATH $WIKI_SECTION_TEST_PATH wikisection > ss-ws.txt

    # nice bash run_sumseg.sh $DIALSEG711_SMALL_SAMPLE_PATH $DIALSEG711_SMALL_SAMPLE_PATH dialseg
    # CUDA_VISIBLE_DEVICES=6 nice bash run_sumseg.sh $AMI_VAL_PATH $AMI_TEST_PATH ami > ss-ami.txt
    # CUDA_VISIBLE_DEVICES=6 nice bash run_sumseg.sh $TIAGE_VAL_PATH $TIAGE_TEST_PATH tiage #> ss-tiage.txt
    # CUDA_VISIBLE_DEVICES=6 nice bash run_sumseg.sh $SUPERDIALSEG_VAL_PATH $SUPERDIALSEG_TEST_PATH superdialseg > ss-superdialseg.txt
    # CUDA_VISIBLE_DEVICES=6 nice bash run_sumseg.sh $QMSUM_VAL_PATH $QMSUM_TEST_PATH qmsum # > ss-qmsum.txt
    # CUDA_VISIBLE_DEVICES=6 nice bash run_sumseg.sh $DIALSEG711_VAL_PATH $DIALSEG711_TEST_PATH dialseg > ss-dialseg.txt

    # python3 main.py sumseg summarize -i $AMI_VAL_PATH -o ami_val_sum -dt ami
    # python3 main.py sumseg summarize -i $AMI_TEST_PATH -o ami_test_sum -dt ami

    # CUDA_VISIBLE_DEVICES=6 python3 main.py sumseg scores -v runs/sumseg_run_20231013_085701/val_embedded_dataset -t runs/sumseg_run_20231013_085701/test_embedded_dataset -dt ami > ss-ami.txt
    # CUDA_VISIBLE_DEVICES=6 python3 main.py sumseg scores -v runs/sumseg_run_20231013_090133/val_embedded_dataset -t runs/sumseg_run_20231013_090133/test_embedded_dataset -dt tiage > ss-tiage.txt
    # CUDA_VISIBLE_DEVICES=6 python3 main.py sumseg scores -v runs/sumseg_run_20231013_090517/val_embedded_dataset -t runs/sumseg_run_20231013_090517/test_embedded_dataset -dt superdialseg > ss-superdialseg.txt
    # CUDA_VISIBLE_DEVICES=6 python3 main.py sumseg scores -v runs/sumseg_run_20231013_093416/val_embedded_dataset -t runs/sumseg_run_20231013_093416/test_embedded_dataset -dt qmsum > ss-qmsum.txt

    # CUDA_VISIBLE_DEVICES=6 python3 main.py sumseg scores -v runs/sumseg_run_20231015_160150/val_embedded_dataset -t runs/sumseg_run_20231015_160150/test_embedded_dataset -dt tiage > ss-tiage-prepared.txt

    printf 'Summarization pipeline is done.\n\n'
fi
