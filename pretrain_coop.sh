#!/bin/bash
post='Pretrain-CoOp-sdg-lr0.002-bs64'
backbone_name='RN101'
dataset='COCO'
train_data_dir='/data/public/coco2014/train2014'
train_list='/data/public/coco2014/annotations/instances_train2014.json'
test_data_dir='/data/public/coco2014/val2014'
test_list='/data/public/coco2014/annotations/instances_val2014.json'
train_label='./data/coco/train_label_vectors.npy'
test_label='./data/coco/val_label_vectors.npy'
graph_file='./data/coco/prob_train.npy'
word_file='./data/coco/vectors.npy'
category_file='./data/coco/category_name.json'

batch_size=64
epochs=30
learning_rate=0.002
weight_decay=0.0005
momentum=0.9
num_classes=80
#input parameter
crop_size=448
scale_size=512
#number of data loading workers
workers=5
#manual epoch number (useful on restarts)
start_epoch=0
#epoch number to decend lr
step_epoch=10
#print frequency (default: 10)
print_freq=100
n_ctx=16
class_token_position='end'

#path to latest checkpoint (default: none)
#resume="model_best_vgg_pretrain_bk.pth.tar"
#resume="backup/86.26.pth.tar"
#evaluate mode
cuda=0
CUDA_VISIBLE_DEVICES=${cuda} python pretrain_coop.py \
--dataset ${dataset} \
--train_data ${train_data_dir} \
--test_data ${test_data_dir} \
--train_list ${train_list} \
--test_list ${test_list} \
--batch_size ${batch_size} \
--train_label ${train_label} \
--test_label ${test_label} \
--graph_file ${graph_file} \
--word_file ${word_file} \
--workers ${workers} \
--epochs ${epochs} \
--start_epoch  ${start_epoch} \
--batch_size ${batch_size} \
--learning-rate ${learning_rate} \
--momentum ${momentum} \
--weight_decay ${weight_decay} \
--crop_size ${crop_size} \
--scale_size ${scale_size} \
--step_epoch ${step_epoch} \
--print_freq ${print_freq} \
--post ${post} \
--backbone_name $backbone_name \
--n_ctx ${n_ctx} \
--class_token_position ${class_token_position}