import numpy as np
import matplotlib.pyplot as plt
from mab_implementation.static_environments.environment import Environment
import math

class ThompsonSamplingAgent():

  def __init__(self, env, max_iterations=500, epsilon=0.2):
    self.env = env
    self.epsilon = epsilon
    self.iterations = max_iterations
    self.means = np.zeros(self.env.k_arms)
    self.stds = np.ones(self.env.k_arms) * 100.0
    self.arm_counts = np.zeros(self.env.k_arms)
    self.arm_rewards = np.zeros(self.env.k_arms)
    self.rewards = [0.0]
    self.cum_rewards = [0.0]

  @staticmethod
  def get_gaussian_likelihood(x, mu, sigma):
    return 1/(sigma*np.sqrt(2*math.pi)) * np.exp(-0.5*((x-mu)/sigma)**2)

  def act(self):
    for i in range(self.iterations):
      # thompson sampling algorithm
      if np.random.rand() <= self.epsilon or i<30*len(self.means):
        rands = [np.random.rand() for _ in self.means]
        likelihoods = [self.get_gaussian_likelihood(x, mu, sigma) for x, mu, sigma in zip(rands, self.means, self.stds)]
        arm = np.argmax(likelihoods)
      else:
        arm = np.argmax(self.means)
      reward = self.env.choose_arm(arm)
      # update counts
      self.arm_counts[arm] += 1
      self.arm_rewards[arm] += reward

      # update posteriors
      self.stds[arm] = (1/100**2 + self.arm_counts[arm])**-0.5
      self.means[arm] = self.arm_rewards[arm] * self.stds[arm]**2

      # update rest of values
      self.rewards.append(reward)
      self.cum_rewards.append(sum(self.rewards) / len(self.rewards))

    return {"arm_counts": self.arm_counts, "rewards": self.rewards, "cum_rewards": self.cum_rewards}

if __name__ == '__main__':
  reward_probabilities, actual_rewards = [0.62, 0.05, 0.87, 0.49], [1.0, 1.0, 1.0, 1.0]

  test_env = Environment(reward_probabilities=reward_probabilities, actual_rewards=actual_rewards)
  thmopson_sampling_agent = ThompsonSamplingAgent(test_env)
  thmopson_sampling_agent_result = thmopson_sampling_agent.act()
  cum_rewards = thmopson_sampling_agent_result["cum_rewards"]
  arm_counts = thmopson_sampling_agent_result["arm_counts"]

  fig = plt.figure(figsize=[7, 10])

  ax1 = fig.add_subplot(211)
  ax1.plot([1.0 for _ in range(thmopson_sampling_agent.iterations)], "g--", label="target cummulative reward")
  ax1.plot(cum_rewards, label="cummulative rewards")
  ax1.set_xlabel("Time steps")
  ax1.set_ylabel("Cummulative rewards")
  ax1.title.set_text(f"Loss vs. Iterations (UCB, Static Env.)")

  ax2 = fig.add_subplot(212)
  ax2.bar([i for i in range(len(arm_counts))], arm_counts)
  ax2.title.set_text("Selected Arm Counts")
  fig.show()

  # we expect E(Rewards) --> average(rewards_ptrobs*reward)
  total_rewards = sum(thmopson_sampling_agent_result['rewards'])
  regret = thmopson_sampling_agent.iterations * max(actual_rewards) - total_rewards
  print(f"Total Reward: {total_rewards}")
  print(f"Regret: {regret}")
  print("expected vs. real cum rewards")
  print(f"Mean Reward from Machines: {cum_rewards[-1]}")
  print(f"Average Real Rewards: {np.mean([a*b for a,b in zip(reward_probabilities,actual_rewards)])}")

  # envirnment real rewards vs. agent asessment
  print("Envirnment real rewards vs. agent asessment")
  print(f"Environment Reward Probabilities : {test_env.reward_probabilities}")
  print(f"Thompson Sampling Agent Action Values : {thmopson_sampling_agent.means}")
  print(f"Posterior Assessed STDS : {thmopson_sampling_agent.stds}")
