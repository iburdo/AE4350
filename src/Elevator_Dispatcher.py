import numpy as np
import random
import matplotlib.pyplot as plt

class ElevatorEnv:
    def __init__(self, num_floors=20):
        self.num_floors = num_floors
        self.current_floor = 1
        self.destination_floor = None
        self.passengers_waiting = [False] * (num_floors + 1)
        
    def reset(self):
        self.current_floor = 1
        self.destination_floor = None
        self.passengers_waiting = [False] * (self.num_floors + 1)
        return self._get_state()
    
    def step(self, action):
        # Actions: 0 = stay, 1 = up, -1 = down
        if action == 1 and self.current_floor < self.num_floors:
            self.current_floor += 1
        elif action == -1 and self.current_floor > 1:
            self.current_floor -= 1
        
        reward = self._calculate_reward()
        done = (self.current_floor == self.destination_floor)
        
        if done:
            self.destination_floor = None
        
        return self._get_state(), reward, done
    
    def _get_state(self):
        return (self.current_floor, self.destination_floor)
    
    def _calculate_reward(self):
        if self.current_floor == self.destination_floor:
            return 10
        elif self.passengers_waiting[self.current_floor]:
            return 5
        else:
            return -1
    
    def set_destination(self, floor):
        self.destination_floor = floor
        self.passengers_waiting[floor] = True

class QLearningAgent:
    def __init__(self, num_floors, epsilon=0.1, alpha=0.1, gamma=0.9):
        self.num_floors = num_floors
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {}
        #epsilon: Probability of choosing a random action (exploration) instead of the best-known action (exploitation).
        #alpha: Learning rate.
        #gamma: Discount factor, which determines how much future rewards are valued.
        #q_table: A dictionary to store Q-values for each state-action pair.
    
    def get_action(self, state):  
        if random.uniform(0, 1) < self.epsilon:
            return random.choice([-1, 0, 1])
        else:
            return self._get_best_action(state)
        #Chooses an action based on the epsilon-greedy policy. 
        #It either picks a random action (exploration) or the best-known action (exploitation).
    
    def _get_best_action(self, state):
        if state not in self.q_table:
            return random.choice([-1, 0, 1])
        return max(self.q_table[state], key=self.q_table[state].get)
    
    def update(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {-1: 0, 0: 0, 1: 0}
        
        old_value = self.q_table[state][action]
        next_max = max(self.q_table.get(next_state, {-1: 0, 0: 0, 1: 0}).values())
        
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state][action] = new_value

def train(num_episodes=10000):
    env = ElevatorEnv()
    agent = QLearningAgent(env.num_floors)
    
    episode_rewards = []
    episode_steps = []
    
    for episode in range(num_episodes):
        state = env.reset()
        env.set_destination(random.randint(1, env.num_floors))
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        
        if episode % 1000 == 0:
            print(f"Episode {episode} completed")
    
    return agent, episode_rewards, episode_steps

def test(agent, num_tests=100):
    env = ElevatorEnv()
    total_steps = 0
    total_rewards = 0
    steps_list = []
    rewards_list = []
    
    for _ in range(num_tests):
        state = env.reset()
        env.set_destination(random.randint(1, env.num_floors))
        done = False
        steps = 0
        episode_reward = 0
        
        while not done:
            action = agent._get_best_action(state)
            state, reward, done = env.step(action)
            steps += 1
            episode_reward += reward
        
        total_steps += steps
        total_rewards += episode_reward
        steps_list.append(steps)
        rewards_list.append(episode_reward)
    
    avg_steps = total_steps / num_tests
    avg_reward = total_rewards / num_tests
    print(f"Average steps to reach destination: {avg_steps:.2f}")
    print(f"Average reward per episode: {avg_reward:.2f}")
    
    return steps_list, rewards_list

def plot_training_progress(episode_rewards, episode_steps):
    fig, (ax2) = plt.subplots(1, 1, figsize=(10, 5))
    
    # Plot episode rewards
    # ax1.plot(episode_rewards)
    # ax1.set_title('Total Reward per Episode')
    # ax1.set_xlabel('Episode')
    # ax1.set_ylabel('Total Reward')
    
    # Plot episode steps
    ax2.plot(episode_steps)
    ax2.set_title('Steps per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    
    plt.tight_layout()
    plt.show()

def plot_test_results(steps_list, rewards_list):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot histogram of steps
    ax1.hist(steps_list, bins=20)
    ax1.set_title('Distribution of Steps to Reach Destination')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Frequency')
    
    # Plot histogram of rewards
    ax2.hist(rewards_list, bins=20)
    ax2.set_title('Distribution of Rewards per Episode')
    ax2.set_xlabel('Reward')
    ax2.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

def analyze_q_table(agent):
    floor_diff = []
    q_values = []
    
    for state in agent.q_table:
        current_floor, destination_floor = state
        if destination_floor is not None:
            floor_difference = destination_floor - current_floor
            best_action = max(agent.q_table[state], key=agent.q_table[state].get)
            best_q_value = agent.q_table[state][best_action]
            
            floor_diff.append(floor_difference)
            q_values.append(best_q_value)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(floor_diff, q_values)
    plt.title('Q-values vs Floor Difference')
    plt.xlabel('Floor Difference (Destination - Current)')
    plt.ylabel('Best Q-value')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.show()

if __name__ == "__main__":
    trained_agent, episode_rewards, episode_steps = train()
    plot_training_progress(episode_rewards, episode_steps)
    
    test_steps, test_rewards = test(trained_agent)
    #plot_test_results(test_steps, test_rewards)
    
    analyze_q_table(trained_agent)
