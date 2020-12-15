python train.py --domain_name FetchPickAndPlace-v1 \
  --reward_type sparse --cameras 8 10 --frame_stack 1 --num_updates 1 \
  --observation_type pixel --encoder_type pixel --work_dir ./data/FetchPickAndPlace-v1 \
  --pre_transform_image_size 100 --image_size 84 --agent rad_sac \
  --seed -1 --critic_lr 0.001 --actor_lr 0.001 --eval_freq 1000 --batch_size 128 \
  --save_tb --save_video --demo_model_dir expert/FetchPickAndPlace-v1 \
  --demo_model_step 195000 --demo_samples 500 \
  --demo_special_reset grip --change_model --bc_only