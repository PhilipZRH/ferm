import numpy as np
import torch
import argparse
import os
import math
import gym
import sys
import random
import time
import json
import copy

import utils
from logger import Logger
from video import VideoRecorder

from curl_sac import CurlSacAgent, RadSacAgent
from torchvision import transforms

import env_wrapper


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='cheetah')
    parser.add_argument('--task_name', default=None)
    parser.add_argument('--pre_transform_image_size', default=100, type=int)
    parser.add_argument('--cameras', nargs='+', default=[0], type=int)
    parser.add_argument('--observation_type', default='pixel')
    parser.add_argument('--reward_type', default='dense')
    parser.add_argument('--special_reset', default=None, type=str)
    parser.add_argument('--change_model', default=False, action='store_true')
    parser.add_argument('--synch_update', default=False, action='store_true')

    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--num_updates', default=1, type=int)
    parser.add_argument('--two_conv', default=False, action='store_true')
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    parser.add_argument('--replay_buffer_load_dir', default="None", type=str)
    parser.add_argument('--model_dir', default=None, type=str)
    parser.add_argument('--model_step', default=None, type=str)
    parser.add_argument('--demo_model_dir', default=None, type=str)
    parser.add_argument('--demo_model_step', default=0, type=int)
    parser.add_argument('--demo_samples', default=25000, type=int)
    parser.add_argument('--demo_special_reset', default=None, type=str)
    parser.add_argument('--success_demo_only', default=False, action='store_true')
    parser.add_argument('--bc_only', default=False, action='store_true')
    parser.add_argument('--log_networks_freq', default=1000000, type=int)
    # warmup options
    parser.add_argument('--warmup_cpc', default=0, type=int)
    parser.add_argument('--warmup_cpc_ema', default=False, action='store_true')
    parser.add_argument('--warmup_offline_sac', default=0, type=int)
    # train
    parser.add_argument('--agent', default='curl_sac', type=str)
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    # eval
    parser.add_argument('--eval_freq', default=5000, type=int)
    parser.add_argument('--num_eval_episodes', default=100, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)  # try 0.05 or 0.1
    parser.add_argument('--critic_target_update_freq', default=2,
                        type=int)  # try to change it to 1 and retain 0.01 above
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder
    parser.add_argument('--encoder_type', default='pixel', type=str)
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    parser.add_argument('--latent_dim', default=128, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--save_sac', default=False, action='store_true')
    parser.add_argument('--detach_encoder', default=False, action='store_true')

    parser.add_argument('--data_augs', default='crop', type=str)

    parser.add_argument('--log_interval', default=100, type=int)
    args = parser.parse_args()
    return args


def evaluate(env, agent, video, num_episodes, L, step, args):
    all_ep_rewards = []

    # if hasattr(env.env._env.unwrapped, 'callback'):
    #     callback_mem = env.env._env.unwrapped.callback
    #     env.env._env.unwrapped.callback = None

    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = 'stochastic_' if sample_stochastically else ''
        num_successes = 0
        for i in range(num_episodes):
            obs = env.reset()
            video.init(enabled=(i == 0))
            done = False
            episode_reward = 0
            episode_success = False
            while not done:
                # center crop image
                if (args.agent == 'curl_sac' and args.encoder_type == 'pixel') or\
                        (args.agent == 'rad_sac' and (args.encoder_type == 'pixel' or 'crop' in args.data_augs or 'translate' in args.data_augs)):
                    if isinstance(obs, list):
                        obs[0] = utils.center_crop_image(obs[0], args.image_size)
                    else:
                        obs = utils.center_crop_image(obs, args.image_size)
                with utils.eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs)
                    else:
                        action = agent.select_action(obs)
                obs, reward, done, info = env.step(action)
                if info.get('is_success'):
                    episode_success = True
                video.record(env)
                episode_reward += reward
            num_successes += episode_success

            video.save('%d.mp4' % step)
            L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
            all_ep_rewards.append(episode_reward)

        L.log('eval/' + prefix + 'eval_time', time.time() - start_time, step)
        if num_episodes > 0:
            mean_ep_reward = np.mean(all_ep_rewards)
            best_ep_reward = np.max(all_ep_rewards)
            std_ep_reward = np.std(all_ep_rewards)
            success_rate = num_successes / num_episodes
        else:
            mean_ep_reward = 0
            best_ep_reward = 0
            std_ep_reward = 0
            success_rate = 0
        L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
        L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)
        L.log('eval/' + prefix + 'success_rate', success_rate, step)

        filename = args.work_dir + '/eval_scores.npy'
        key = args.domain_name + '-' + str(args.task_name) + '-' + args.data_augs
        try:
            log_data = np.load(filename, allow_pickle=True)
            log_data = log_data.item()
        except FileNotFoundError:
            log_data = {}

        if key not in log_data:
            log_data[key] = {}

        log_data[key][step] = {}
        log_data[key][step]['step'] = step
        log_data[key][step]['mean_ep_reward'] = mean_ep_reward
        log_data[key][step]['max_ep_reward'] = best_ep_reward
        log_data[key][step]['success_rate'] = success_rate
        log_data[key][step]['std_ep_reward'] = std_ep_reward
        log_data[key][step]['env_step'] = step * args.action_repeat

        np.save(filename, log_data)

    run_eval_loop(sample_stochastically=False)
    L.dump(step)
    # if hasattr(env.env._env.unwrapped, 'callback'):
    #     env.env._env.unwrapped.callback = callback_mem


def make_agent(obs_shape, action_shape, args, device, hybrid_state_shape):
    if args.agent == 'curl_sac':
        return CurlSacAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            latent_dim=args.latent_dim,
            hybrid_state_shape=hybrid_state_shape,
            two_conv=args.two_conv
        )
    elif args.agent == 'rad_sac':
        return RadSacAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            latent_dim=args.latent_dim,
            data_augs=args.data_augs,
            hybrid_state_shape=hybrid_state_shape,
            two_conv=args.two_conv
        )
    else:
        assert 'agent is not supported: %s' % args.agent


def main():
    args = parse_args()
    if args.seed == -1:
        args.__dict__["seed"] = np.random.randint(1, 1000000)
    exp_id = str(int(np.random.random() * 100000))
    utils.set_seed_everywhere(args.seed)

    env = env_wrapper.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        visualize_reward=False,
        from_pixels=(args.observation_type == 'pixel' or args.observation_type == 'hybrid'),
        cameras=args.cameras,
        height=args.pre_transform_image_size,
        width=args.pre_transform_image_size,
        frame_skip=args.action_repeat,
        reward_type=args.reward_type,
        change_model=args.change_model
    )

    env.seed(args.seed)
    if args.special_reset is not None:
        env.set_special_reset(args.special_reset)
    if args.demo_special_reset is not None:
        env.set_special_reset(args.demo_special_reset)

    if args.observation_type == 'hybrid':
        env.set_hybrid_obs()

    # stack several consecutive frames together
    if args.encoder_type == 'pixel':
        env = utils.FrameStack(env, k=args.frame_stack)

    # make directory
    ts = time.gmtime()
    ts = time.strftime("%m-%d", ts)
    if args.task_name is None:
        env_name = args.domain_name
    else:
        env_name = args.domain_name + '-' + args.task_name
    exp_name = args.reward_type + '-' + args.agent + '-' + args.encoder_type + '-' + args.data_augs
    exp_name += '-' + ts + '-' + env_name + '-im' + str(args.image_size) + '-b' + str(args.batch_size) + '-nu' + str(args.num_updates)
    if args.observation_type == 'hybrid':
        exp_name += '-hybrid'
    if args.change_model:
        exp_name += '-change_model'
    if args.bc_only:
        exp_name += '-bc_only'

    exp_name += '-s' + str(args.seed)

    exp_name += '-id' + exp_id
    args.work_dir = args.work_dir + '/' + exp_name
    utils.make_dir(args.work_dir)
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    print("Working in directory:", args.work_dir)

    video = VideoRecorder(video_dir if args.save_video else None, camera_id=args.cameras[0])

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    action_shape = env.action_space.shape

    if args.encoder_type == 'pixel':
        cpf = 3 * len(args.cameras)
        obs_shape = (cpf * args.frame_stack, args.image_size, args.image_size)
        pre_aug_obs_shape = (cpf * args.frame_stack, args.pre_transform_image_size, args.pre_transform_image_size)
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    replay_buffer = utils.ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        image_size=args.image_size,
        hybrid_state_shape=env.hybrid_state_shape,
        load_dir=args.replay_buffer_load_dir
    )

    if args.demo_model_dir is not None:  # collect demonstrations using a state-trained expert
        episode_step, done = 0, True
        state_obs, obs = None, None
        episode_success = False
        original_encoder_type = args.encoder_type
        args.encoder_type = 'identity'

        if isinstance(env, utils.FrameStack):
            original_env = env.env
        else:
            original_env = env

        expert_agent = make_agent(
            obs_shape=original_env.observation_space.shape,
            action_shape=action_shape,
            args=args,
            device=device,
            hybrid_state_shape=env.hybrid_state_shape
        )
        args.encoder_type = original_encoder_type
        expert_agent.load(args.demo_model_dir, args.demo_model_step)
        print('Collecting expert trajectories...')
        t = 0
        while t < args.demo_samples:
            if done:
                episode_step = 0
                episode_success = False
                if args.demo_special_reset is not None:
                    env.reset(save_special_steps=True)
                    special_steps_dict = env.special_reset_save
                    obs_list = special_steps_dict['obs']
                    act_list = special_steps_dict['act']
                    reward_list = special_steps_dict['reward']
                    for i in range(len(act_list)):
                        replay_buffer.add(obs_list[i], act_list[i], reward_list[i], obs_list[i+1], False)
                    episode_step += len(act_list)
                    t += len(act_list)
                    obs = obs_list[-1]
                    state_obs = original_env._get_state_obs()
                else:
                    obs = env.reset()
                    state_obs = original_env._get_state_obs()

            action = expert_agent.sample_action(state_obs)
            next_obs, reward, done, info = env.step(action)
            if info.get('is_success'):
                episode_success = True
            state_obs = original_env._get_state_obs()

            # allow infinite bootstrap
            done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)

            replay_buffer.add(obs, action, reward, next_obs, done_bool)

            obs = next_obs
            episode_step += 1
            t += 1

            if args.success_demo_only and done and not episode_success:
                t -= episode_step
                replay_buffer.idx -= episode_step

        env.set_special_reset(args.special_reset)

    print('Starting with replay buffer filled to {}.'.format(replay_buffer.idx))

    # args.init_steps = max(0, args.init_steps - args.replay_buffer_load_pi_t)  # maybe tune this

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device,
        hybrid_state_shape=env.hybrid_state_shape
    )
    if args.model_dir is not None:
        agent.load(args.model_dir, args.model_step)
    L = Logger(args.work_dir, use_tb=args.save_tb)

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()

    def eval_and_save():
        if args.save_model:
            agent.save_curl(model_dir, step)
        if args.save_buffer:
            replay_buffer.save(buffer_dir)
        if args.save_sac:
            agent.save(model_dir, step)
        L.log('eval/episode', episode, step)
        print('evaluating')
        evaluate(env, agent, video, args.num_eval_episodes, L, step, args)

    if args.warmup_cpc:
        print("Warming up cpc for " + str(args.warmup_cpc) + ' steps.')
        for i in range(args.warmup_cpc):
            agent.update_cpc_only(replay_buffer, L, step=0, ema=args.warmup_cpc_ema)
        print('Warmed up cpc.')

    if args.warmup_offline_sac:
        for i in range(args.warmup_offline_sac):
            agent.update_sac_only(replay_buffer, L, step=0)

    if args.bc_only:
        step = 0
        for i in range(100):
            agent.train_bc(replay_buffer)
            step += 1
        eval_and_save()
        return

    time_computing = 0
    time_acting = 0
    callback_fn = None
    step = 0

    if args.synch_update:
        callback_fn = lambda: lambda: [agent.update(replay_buffer, L, step, log_networks=nu == 0 and step % args.log_networks_freq == 0) for nu in range(args.num_updates)] if step >= args.init_steps and not is_eval else 0  # pointers should all work properly, and execute in the proper frame

    if callback_fn is not None:
        env.env._env.env.set_callback(callback_fn)  # envwrapper (camera), framestack, timelimit

    # for step in range(args.num_train_steps):
    while step < args.num_train_steps:

        # evaluate agent periodically
        if step % args.eval_freq == 0:
            is_eval = True
            eval_and_save()
            is_eval = False

        if done:
            if step > 0:
                if step % args.log_interval == 0:
                    L.log('train/duration', time.time() - start_time, step)
                    L.dump(step)
                start_time = time.time()
            if step % args.log_interval == 0:
                L.log('train/episode_reward', episode_reward, step)

            time_tmp = time.time()
            obs = env.reset()
            time_acting += time.time() - time_tmp
            episode_reward = 0
            episode_step = 0
            episode += 1
            if step % args.log_interval == 0:
                L.log('train/episode', episode, step)

        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)
        if step == args.init_steps and args.demo_samples == 0:
            if args.warmup_cpc:
                for i in range(args.warmup_cpc):
                    print("Warming up cpc for " + str(args.warmup_cpc) + ' steps.')
                    agent.update_cpc_only(replay_buffer, L, step=0)
                    print('Warmed up cpc.')

        # run training update
        time_tmp = time.time()

        if step >= args.init_steps and not args.synch_update:
            for nu in range(args.num_updates):
                agent.update(replay_buffer, L, step, log_networks=nu == 0 and step % args.log_networks_freq == 0)

        time_computing += time.time() - time_tmp

        time_tmp = time.time()

        next_obs, reward, done, _ = env.step(action)
        time_acting += time.time() - time_tmp

        # allow infinite bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
        episode_reward += reward

        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_step += 1
        step += 1

    step = args.num_train_steps
    print("time spent computing:", time_computing)
    print("time spent acting:", time_acting)
    eval_and_save()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    main()
