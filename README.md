# Delayed TRPO and DQN

PuddleWorld can be installed following: https://github.com/EhsanEI/gym-puddle

To train the algortihms whose results are gathered in the paper, run the following commands: 

## TRPO on deterministic environment:
TRPO train with deterministic delays and Pendulum:
```bash
python run_trpo.py --env Pendulum --mode train --seed 0 --delay 3 --epochs 1000 --steps_per_epoch 5000 --max_ep_len 250 --delta 0.001 --v_hid 64 --v_l 1 --vf_lr 0.01 --v_iters 3 --pi_hid 64 --pi_l 2
```

Delayed TRPO train with deterministic delays and Pendulum:
```bash
python run_dtrpo.py --env Pendulum --mode train --seed 0 --delay 3 --epochs 1000 --steps_per_epoch 5000 --max_ep_len 250 --delta 0.001 --v_hid 64 --v_l 1 --vf_lr 0.01 --v_iters 3 --pi_hid 64 --pi_l 2 --pretrain_epochs 2 --pretrain_steps 10000 --size_pred_buf 100000 --batch_size_pred 10000 --train_enc_iters 1 --enc_lr 0.005 --enc_dim 64 --enc_pred_to_pi --enc_causal
```

Delayed TRPO train with stochastic delays and Pendulum:
```bash
python run_dtrpo.py --env Pendulum --mode train --seed 0 --stochastic_delays --max_delay 50 --delay_proba 0.55 --epochs 500 --steps_per_epoch 5000 --max_ep_len 250 --delta 0.001 --v_hid 64 --v_l 1 --vf_lr 0.01 --v_iters 3 --pi_hid 64 --pi_l 2 --pretrain_epochs 2 --pretrain_steps 10000 --size_pred_buf 100000 --batch_size_pred 10000 --enc_lr 0.005 --enc_dim 64 --enc_pred_to_pi --enc_causal
```

## DQN on stochastic environment:
DQN train with deterministic delays, PuddleWorld:
```bash
python run_dqn.py --env PuddleWorld --seed 0 --mode train --delay 3 --learning_rate 0.005 --batch_size 64 --frame_history_len 1 --learning_freq 1 --q_fun_neurons 64 --max_timesteps 2000000 --dueling_dqn --double_dqn
```

Delayed DQN train with deterministic delays, PuddleWorld:
```bash
python run_ddqn.py --env PuddleWorld --seed 0 --mode train --delay 3 --learning_rate 0.005 --batch_size 64 --frame_history_len 1 --dueling_dqn --double_dqn --learning_freq 1 --q_fun_neurons 64 --max_timesteps 2000000 --batch_size_pred 64 --enc_causal
```
