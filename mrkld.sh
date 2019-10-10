CUDA_VISIBLE_DEVICES=0 python crst_seg.py --random-mirror --random-scale --rm-prob --test-flipping --save results/mrkld --data-src-dir DATA_SRC_DIR --data-tgt-dir DATA_TGT_DIR --mr-weight-kld 0.1
