import numpy as np
import matplotlib.pyplot as plt
from mab_implementation.static_environments.environment import Environment


class UCBAgent():

  def __init__(self, env, max_iterations=500, c=2):
    self.env = env
    self.iterations = max_iterations

    self.c = c
    self.q_values = np.zeros(self.env.k_arms)
    self.arm_counts = np.zeros(self.env.k_arms)
    self.arm_rewards = np.zeros(self.env.k_arms)

    self.rewards = [0.0]
    self.cum_rewards = [0.0]

  def act(self):
    for i in range(self.iterations):
      if i < len(self.arm_counts):
        arm = i
        reward = self.env.choose_arm(i)
      else :
        uncertainity = self.c * np.sqrt(np.log(i) / self.arm_counts)
        arm = np.argmax(self.q_values + uncertainity)
        reward = self.env.choose_arm(arm)
      self.arm_counts[arm] += 1
      self.arm_rewards[arm] += reward

      self.q_values[arm] = self.q_values[arm] + (1/self.arm_counts[arm]) * (reward - self.q_values[arm])
      self.rewards.append(reward)
      self.cum_rewards.append(sum(self.rewards) / len(self.rewards))

    return {"arm_counts": self.arm_counts, "rewards": self.rewards, "cum_rewards": self.cum_rewards}

if __name__ == '__main__':
  reward_probabilities, actual_rewards = [0.62, 0.05, 0.87, 0.49], [1.0, 1.0, 1.0, 1.0]
  c = 0.2

  test_env = Environment(reward_probabilities=reward_probabilities, actual_rewards=actual_rewards)
  ucb_agent = UCBAgent(test_env, c=c)
  ucb_agent_result = ucb_agent.act()
  cum_rewards = ucb_agent_result["cum_rewards"]
  arm_counts = ucb_agent_result["arm_counts"]

  fig = plt.figure(figsize=[7, 10])

  ax1 = fig.add_subplot(211)
  ax1.plot([1.0 for _ in range(ucb_agent.iterations)], "g--", label="target cummulative reward")
  ax1.plot(cum_rewards, label="cummulative rewards")
  ax1.set_xlabel("Time steps")
  ax1.set_ylabel("Cummulative rewards")
  ax1.title.set_text(f"Loss vs. Iterations (UCB, Static Env.)")

  ax2 = fig.add_subplot(212)
  ax2.bar([i for i in range(len(arm_counts))], arm_counts)
  ax2.title.set_text("Selected Arm Counts")
  fig.show()

  # we expect E(Rewards) --> average(rewards_ptrobs*reward)
  total_rewards = sum(ucb_agent_result['rewards'])
  regret = ucb_agent.iterations * max(actual_rewards) - total_rewards
  print(f"Total Reward: {total_rewards}")
  print(f"Regret: {regret}")
  print("expected vs. real cum rewards")
  print(f"Mean Reward from Machines: {cum_rewards[-1]}")
  print(f"Average Real Rewards: {np.mean([a*b for a,b in zip(reward_probabilities,actual_rewards)])}")

  # envirnment real rewards vs. agent asessment
  print("Envirnment real rewards vs. agent asessment")
  print(f"Environment Reward Probabilities : {test_env.reward_probabilities}")
  print(f"UCB Agent Action Values : {ucb_agent.q_values}")