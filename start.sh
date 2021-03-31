# python main.py --fold 1 --epoch 50 --crop_size 4096 --train_img_size 1024 --w_path weights/4096_1024_outlier --iter 100
python main.py \
--fold 0 \
--epoch 30 \
--crop_size 4096 \
--train_img_size 1024 \
--w_path 4096_1024_pub_augs \
--iter 100 \
--batch_size 8 \
--encoder efficientnet-b3 \
--start_lr 0.001

python main.py \
--fold 0 \
--epoch 30 \
--crop_size 4096 \
--train_img_size 1024 \
--w_path 4096_1024_pub_augs_lr_1e-4 \
--iter 100 \
--start_lr 0.0001
