CUDA_VISIBLE_DEVICES=0 python crst_seg.py --random-mirror --random-scale --rm-prob --test-flipping --save results/lrent --data-src-dir DATA_SRC_DIR --data-tgt-dir DATA_TGT_DIR --lr-weight-ent 0.25
