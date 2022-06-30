import numpy as np
import matplotlib.pyplot as plt
from mab_implementation.static_environments.environments import Environment


class RandomAgent(object):

  def __init__(self, env, max_iterations=500):
    self.env = env
    self.iterations = max_iterations

    self.q_values = np.zeros(self.env.k_arms)
    self.arm_counts = np.zeros(self.env.k_arms)
    self.arm_rewards = np.zeros(self.env.k_arms)

    self.rewards = [0.0]
    self.cum_rewards = [0.0]

  def act(self):
    for i in range(self.iterations):
      arm = np.random.choice(self.env.k_arms)
      reward = self.env.choose_arm(arm)

      self.arm_counts[arm] = self.arm_counts[arm] + 1
      self.arm_rewards[arm] = self.arm_rewards[arm] + reward

      self.q_values[arm] = self.q_values[arm] + (1/self.arm_counts[arm]) * (reward - self.q_values[arm])
      self.rewards.append(reward)
      self.cum_rewards.append(sum(self.rewards) / len(self.rewards))

    return {"arm_counts": self.arm_counts, "rewards": self.rewards, "cum_rewards": self.cum_rewards}

if __name__ == '__main__':
  reward_probabilities, actual_rewards = [0.62, 0.05, 0.87, 0.49], [1.0, 1.0, 1.0, 1.0]
  test_env = Environment(reward_probabilities=reward_probabilities, actual_rewards=actual_rewards)
  random_agent = RandomAgent(test_env)
  random_agent_result = random_agent.act()
  cum_rewards = random_agent_result["cum_rewards"]
  arm_counts = random_agent_result["arm_counts"]

  fig = plt.figure(figsize=[30, 10])

  ax1 = fig.add_subplot(121)
  ax1.plot([1.0 for _ in range(random_agent.iterations)], "g--", label="target cummulative reward")
  ax1.plot(cum_rewards, label="cummulative rewards")
  ax1.set_xlabel("Time steps")
  ax1.set_ylabel("Cummulative rewards")

  ax2 = fig.add_subplot(122)
  ax2.bar([i for i in range(len(arm_counts))], arm_counts)
  fig.show()

  # we expect E(Rewards) --> average(rewards_ptrobs*reward)
  print("expected vs. real cum rewards")
  print(f"Average Reward from Machines: {cum_rewards[-1]}")
  print(f"Real Average Rewards: {np.mean([a*b for a,b in zip(reward_probabilities,actual_rewards)])}")

  # envirnment real rewards vs. agent asessment
  print("envirnment real rewards vs. agent asessment")
  print(f"Environment Reward Probabilities : {test_env.reward_probabilities}")
  print(f"Random Agent Action Values : {random_agent.q_values}")