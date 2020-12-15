python train.py --domain_name RealRobotEnv-v0 \
	--cameras 8 10 --frame_stack 1 --observation_type pixel --encoder_type pixel \
	--save_tb --save_buffer --save_video --save_sac \
	--work_dir real_robot_data/v0 \
  --num_eval_episodes 30 \
	--pre_transform_image_size 100 --image_size 84 --agent rad_sac --data_aug crop \
	--seed 15 \
	--batch_size 128  --init_steps 0 \
	--reward_type v0 --replay_buffer_load_dir real_robot_demo/v0 \
	--actor_log_std_max 0 --bc_only
