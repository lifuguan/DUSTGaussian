export CUDA_VISIBLE_DEVICES=2,3
python -m torch.distributed.launch --nproc_per_node=2 \
       --master_port=$(( RANDOM % 1000 + 50000 )) \
       train_dust2gs.py --config configs/train_dust2gs.yaml \
       --ckpt_path model_zoo/dust2gs_wo_gs.pth --expname dust2gs_costvolume



export CUDA_VISIBLE_DEVICES=4,5
python -m torch.distributed.launch --nproc_per_node=2 \
       --master_port=$(( RANDOM % 1000 + 50000 )) \
       train_dust2mvsplat.py --config configs/train_mvsplat.yaml \
       --ckpt_path model_zoo/dust2mvsplat.pth --expname dust2mvsplat_2gpu