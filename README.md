export CUDA_VISIBLE_DEVICES=2,3,5,6
python -m torch.distributed.launch --nproc_per_node=4 \
       --master_port=$(( RANDOM % 1000 + 50000 )) \
       train_dust2gs.py --config configs/train_dust2gs.yaml \
       --ckpt_path model_zoo/dust2gs.pth --expname dust2gs_4gpu