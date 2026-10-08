# python train1.py \
# --train_goal 2 \
# --data_name LsSCD_MCI \
# --list_path ../../LsSCD-Ex/ \
# --gpu_ids '0' \
# --network mobilenetv2 \
# --train_batchsize 16 \
# --encoder_lr 5e-4 \
# --data_folder ../../LsSCD-Ex-dataset/images/ \
# --token_folder ../../LsSCD-Ex/tokens/ \
# --savepath ./LsSCD-Ex_ckpt/
python train.py \
--train_goal 2 \
--data_name LsSCD_MCI \
--list_path ../../LsSCD-Ex/ \
--gpu_ids '3' \
--train_batchsize 8 \
--encoder_lr 1.25e-5 \
--data_folder ../../LsSCD-Ex-dataset/images/ \
--token_folder ../../LsSCD-Ex/tokens/ \
--savepath ./LsSCD-Ex_ckpt/
