# CART POLE BALANCING

## AIM
To develop and fine tune the Monte Carlo algorithm to stabilize the Cart Pole.

## PROBLEM STATEMENT
The problem statement involves using a MC control algorithm on the discretized Cart pole environment. The environment consists of a Pole balancing on a cart, we know about the cart length, velocity, pole angle and pole angular velocity and aim to keep the pole upright by moving it either to the right or the left direction and keep it within specified bounds.

## MONTE CARLO CONTROL ALGORITHM FOR CART POLE BALANCING
Monte Carlo control for cart-pole balancing uses repeated simulated episodes to learn an optimal policy that keeps the pole upright by estimating action values through experience. In each episode, the agent interacts with the environment until termination, recording the sequence of states, actions, and rewards. After an episode ends, returns are computed for every state–action pair encountered, and these returns are averaged over many episodes to form empirical estimates of the action-value function Q(s,a)Q(s,a)Q(s,a). The policy is then improved using an ϵ\epsilonϵ-greedy strategy, selecting mostly the action with the highest estimated value while still exploring occasionally. Over many episodes, Monte Carlo control converges toward a policy that balances the pole by assigning higher value to actions that lead to longer survival and lower value to actions that cause the pole to fall, without requiring a model of the environment’s dynamics.

## MONTE CARLO CONTROL FUNCTION

```python
def mc_control (env,n_bins=g_bins, gamma = 1.0,
                init_alpha = 0.5,min_alpha = 0.01, alpha_decay_ratio = 0.5,
                init_epsilon = 1.0, min_epsilon = 0.1, epsilon_decay_ratio = 0.9,
                n_episodes = 3000, max_steps = 200, first_visit = True, init_Q=None):

    nA = env.action_space.n
    discounts = np.logspace(0, max_steps,
                            num = max_steps, base = gamma,
                            endpoint = False)
    alphas = decay_schedule(init_alpha, min_alpha,
                            0.9999, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon,
                            0.99, n_episodes)
    pi_track = []
    global Q_track
    global Q


    if init_Q is None:
        Q = np.zeros([n_bins]*env.observation_space.shape[0] + [env.action_space.n],dtype =np.float64)
    else:
        Q = init_Q

    n_elements = Q.size
    n_nonzero_elements = 0

    Q_track = np.zeros([n_episodes] + [n_bins]*env.observation_space.shape[0] + [env.action_space.n],dtype =np.float64)
    select_action = lambda state, Q, epsilon: np.argmax(Q[tuple(state)]) if np.random.random() > epsilon else np.random.randint(len(Q[tuple(state)]))

    progress_bar = tqdm(range(n_episodes), leave=False)
    steps_balanced_total = 1
    mean_steps_balanced = 0
    for e in progress_bar:
        trajectory = generate_trajectory(select_action, Q, epsilons[e],
                                    env, max_steps)

        steps_balanced_total = steps_balanced_total + len(trajectory)
        mean_steps_balanced = 0

        visited = np.zeros([n_bins]*env.observation_space.shape[0] + [env.action_space.n],dtype =np.float64)
        for t, (state, action, reward, _, _) in enumerate(trajectory):
            if visited[tuple(state)][action] and first_visit:
                continue
            visited[tuple(state)][action] = True
            n_steps = len(trajectory[t:])
            G = np.sum(discounts[:n_steps]*trajectory[t:, 2])
            Q[tuple(state)][action] = Q[tuple(state)][action]+alphas[e]*(G - Q[tuple(state)][action])
        Q_track[e] = Q
        n_nonzero_elements = np.count_nonzero(Q)
        pi_track.append(np.argmax(Q, axis=env.observation_space.shape[0]))
        if e != 0:
            mean_steps_balanced = steps_balanced_total/e
        progress_bar.set_postfix(episode=e, Epsilon=epsilons[e], StepsBalanced=f"{len(trajectory)}" ,MeanStepsBalanced=f"{mean_steps_balanced:.2f}")

    print("mean_steps_balanced={0},steps_balanced_total={1}".format(mean_steps_balanced,steps_balanced_total))
    V = np.max(Q, axis=env.observation_space.shape[0])
    pi = lambda s:{s:a for s, a in enumerate(np.argmax(Q, axis=env.observation_space.shape[0]))}[s]

    return Q, V, pi
```

## OUTPUT:
1. Specify the average number of steps achieved within two minutes when the Monte Carlo (MC) control algorithm is initiated with zero-initialized Q-values..
   <img width="630" height="35" alt="image" src="https://github.com/user-attachments/assets/4a5342e1-240e-4094-ad44-d22be3aeb80a" />
3. Mention the average number of steps maintained over a four-minute period when the Monte Carlo (MC) control algorithm is executed with pretrained Q-values.
   <img width="625" height="35" alt="image" src="https://github.com/user-attachments/assets/342ff3d2-b48f-45d9-a0c3-2869f834a04a" />
## RESULT:
Thus we have successfully simulated the Cart Pole environment by using MC control algorithm.
