python test.py \
--data_name LsSCD_MCI \
--list_path ../../LsSCD-Ex/ \
--network mobilenetv2 \
--gpu_id 3 \
--num_heads 8 \
--checkpoint ./WHU_dino_ckpt/heads-8/baseline_2026-07-01-03-29-12_train_goal_2_31/WHU_MCI_bts_16_mobilenetv2_epo_23_Sum_180574_MIou_88667_Bleu4_91907.pth \
--data_folder ../../LsSCD-Ex-dataset/images/ \
--token_folder ../../LsSCD-Ex/tokens/ \
# --result_path ./LsSCD-Ex_semantic \
# --save_mask \
# --save_caption
# python test.py \
# --data_name S2Looking_MCI \
# --list_path ../../S2Looking/ \
# --gpu_id 7 \
# --network mobilenetv2 \
# --checkpoint  ./S2Looking_ckpt/baseline_2026-08-04-10-36-40100/baseline_2026-08-04-10-36-40100/S2Looking_MCI_bts_16_mobilenetv2_epo_64_Sum_137986_MIou_67110_Bleu4_70876.pth \
# --data_folder ../../S2Looking_dataset/images/ \
# --token_folder ../../S2Looking/tokens/ \
# --result_path ./s2looking_semantic-cc \
# --save_mask \
# --save_caption
# python test.py \
# --data_name WHU_MCI \
# --list_path ../../WHU-BSCD/ \
# --network mobilenetv2 \
# --num_heads 4 \
# --gpu_id 3 \
# --checkpoint ./WHU_dino_ckpt/heads-4/baseline_2026-07-03-02-02-2026/WHU_MCI_bts_16_mobilenetv2_epo_24_Sum_178517_MIou_86529_Bleu4_91988.pth \
# --data_folder ../../WHU-BSCD-dataset/images/ \
# --token_folder ../../WHU-BSCD/tokens/ \
# --result_path ./whu_semantic-cc \
# --save_mask \
# --save_caption
# python test.py \
# --data_name WHU_MCI \
# --list_path ../../WHU-BSCD/ \
# --network mobilenetv2 \
# --num_heads 16 \
# --gpu_id 3 \
# --checkpoint ./WHU_dino_ckpt/heads-16/baseline_2026-07-03-10-29-3136/WHU_MCI_bts_16_mobilenetv2_epo_9_Sum_180211_MIou_88783_Bleu4_91427.pth \
# --data_folder ../../WHU-BSCD-dataset/images/ \
# --token_folder ../../WHU-BSCD/tokens/ 
# python test.py \
# --data_name WHU_MCI \
# --list_path ../../WHU-BSCD/ \
# --network mobilenetv2 \
# --num_heads 32 \
# --gpu_id 3 \
# --checkpoint  ./WHU_dino_ckpt/heads-32/baseline_2026-07-04-03-49-2560/WHU_MCI_bts_16_mobilenetv2_epo_13_Sum_179906_MIou_88807_Bleu4_91098.pth \
# --data_folder ../../WHU-BSCD-dataset/images/ \
# --token_folder ../../WHU-BSCD/tokens/ 
