# %%
import gymnasium as gym
import numpy as np

def run(policy, episodes=100, seed=0):
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed)
    scores = []

    for _ in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(1e9)))
        done, total = False, 0.0

        while not done:
            a = policy(obs, rng)
            obs, r, terminated, truncated, _ = env.step(a)
            total += r
            done = terminated or truncated

        scores.append(total)

    env.close()
    return np.array(scores)

# obs = [x, x_dot, theta, theta_dot]   action: 0 = left, 1 = right
def p_random(obs, rng):
    return int(rng.integers(2))

def p_angle(obs, rng):
    return 1 if obs[2] > 0 else 0

def p_both(obs, rng):
    return 1 if obs[2] + 0.5 * obs[3] > 0 else 0

# %%
for name, p in [("random", p_random),
                ("angle", p_angle),
                ("angle + 0.5*vel", p_both)]:
    s = run(p)
    print(f"{name:16s} mean {s.mean():6.1f}   min {s.min():5.0f}   "
          f"max {s.max():5.0f}   solved {(s >= 500).mean():.0%}")