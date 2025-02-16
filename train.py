from env import WordleEnvironment
from dqn import WorldeDQNAgent
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def train_agent(num_episode=10000, eval_frequencies=100):
    env = WordleEnvironment(word_list_path='words.txt')
    state_size = len(env._get_state())
    action_size = len(env.valid_words)

    agent = WorldeDQNAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_size=256
    )

    episode_rewards = []
    eval_rewards = []
    win_rates = []

    progress_bar = tqdm(range(num_episode), desc='Training')
    for episode in progress_bar:
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action_word = agent.select_action(state, env.valid_words)

            action_idx = env.valid_words.index(action_word)

            next_state, reward, done = env.step(action_word)

            agent.memory.push(
                state,
                action_idx,
                reward,
                next_state,
                done
                )

            state = next_state
            total_reward += reward

            if len(agent.memory) > agent.batch_size:
                agent.train(agent.batch_size)

        if episode % agent.target_update == 0:
            agent.update_target_network()

        episode_rewards.append(total_reward)

        if episode % eval_frequencies == 0:
            eval_reward, win_rate = evaluate_agent(env, agent, num_eval_episodes=100)
            eval_rewards.append(eval_reward)
            win_rates.append(win_rate)

            progress_bar.set_postfix({
                'Episode': episode,
                'Reward': f"{total_reward:.1f}",
                'Eval Reward': f"{eval_reward:.1f}",
                'Win Rate': f"{win_rate:.2%}",
                'Epsilon': f"{agent.epsilon:.2f}"
            })

            if len(win_rates) > 1 and win_rate > max(win_rates[:-1]):
                torch.save(agent.policy_net.state_dict(), 'best_model.pth')

    return episode_rewards, eval_rewards, win_rates

def evaluate_agent(env, agent, num_eval_episodes=100):
    eval_rewards = []
    wins = 0

    current_epsilon = agent.epsilon
    agent.epsilon = agent.epsilon_min

    for _ in range(num_eval_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action_word = agent.select_action(state, env.valid_words)
            next_state, reward, done = env.step(action_word)
            state = next_state
            total_reward += reward

            if reward > 100:
                wins += 1

        eval_rewards.append(total_reward)

    agent.epsilon = current_epsilon

    avg_reward = np.mean(eval_rewards)
    win_rates = wins / num_eval_episodes

    return avg_reward, win_rates

def plot_training_results(episode_rewards, eval_rewards, win_rates, eval_frequency):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    ax1.plot(episode_rewards)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')

    eval_episodes = range(0, len(episode_rewards), eval_frequency)
    ax2.plot(eval_episodes, eval_rewards)
    ax2.set_title('Evaluation Rewards')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Reward')

    ax3.plot(eval_episodes, win_rates)
    ax3.set_title('Win Rate')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Win Rate')
    ax3.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()

if __name__ == '__main__':
    set_seeds(97)

    NUM_EPISODES = 10000
    EVAL_FREQUENCY = 100

    print("Starting training...")
    episode_rewards, eval_rewards, win_rates = train_agent(NUM_EPISODES, EVAL_FREQUENCY)

    plot_training_results(episode_rewards, eval_rewards, win_rates, EVAL_FREQUENCY)
    print("Training complete.")