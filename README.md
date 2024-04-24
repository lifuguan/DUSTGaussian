export CUDA_VISIBLE_DEVICES=0,2,3,4
python -m torch.distributed.launch --nproc_per_node=4 \
       --master_port=$(( RANDOM % 1000 + 50000 )) \
       train.py --config configs/train_mvsplat.yaml \
       --ckpt_path model_zoo/mvsplat.pth --expname debug