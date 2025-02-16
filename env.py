import numpy as np
import random
from typing import List, Tuple, Dict

class WordleEnvironment:
    def __init__(self, word_list_path: str = 'words.txt'):
        with open(word_list_path, 'r') as f:
            self.valid_words = [line.strip().upper() for line in f]
            
            self.max_attempts = 6
            self.word_length = 5

            self.target_word = None
            self.current_attempt = 0
            self.game_over = False
            self.board_state = None
            self.keyboard_state = None

            self.rewards = {
                'correct_position': 10,    #Green
                'correct_letter': 5,       #Yellow
                'invalid_guess': -5,       #Word not in dictionary
                'repeated_word': -3,       #Already Guessed
                'game_over': -50,          #All Attempts Used
                'game_win': 100            #Correct Guess
            }

            self.previous_guesses = set()
            self.reset()

    #Reset env for new game
    def reset(self):
        self.target_word = random.choice(self.valid_words)
        self.current_attempt = 0
        self.game_over = False
        self.previous_guesses.clear()

        self.board_state = np.zeros((self.max_attempts, self.word_length, 26))

        self.keyboard_state = np.ones(26)

        return self._get_state()
    
    #Move a step in the game
    def step(self, action: str) -> Tuple[np.ndarray, float, bool]:
        action = action.upper()
        reward = 0

        if self.game_over:
            return self._get_state(), 0, True
        
        if action not in self.valid_words:
            return self._get_state(), self.rewards['invalid_guess'], False
        
        if action in self.previous_guesses:
            return self._get_state(), self.rewards['repeated_word'], False
        
        self.previous_guesses.add(action)

        feedback = self._check_guess(action)
        reward += self._calculate_reward(feedback)

        self._update_board(action, feedback)
        self.current_attempt += 1

        #win or lose condition
        if action == self.target_word:
            reward += self.rewards['game_win']
            self.game_over = True
        elif self.current_attempt >= self.max_attempts:
            reward += self.rewards['game_over']
            self.game_over = True

        return self._get_state(), reward, self.game_over
    
    def _check_guess(self, guess: str) -> List[str]:
        feedback = []
        target_letters = list(self.target_word)

        for i, (guess_letter, target_letter) in enumerate(zip(guess, target_letters)):
            if guess_letter == target_letter:
                feedback.append(('correct', guess_letter))
                target_letters[i] = None
            else:
                feedback.append((None, guess_letter))

        for i, (status, guess_letter) in enumerate(feedback):
            if status is None:
                if guess_letter in target_letters:
                    feedback[i] = ('present', guess_letter)
                    target_letters[target_letters.index(guess_letter)] = None
                else:
                    feedback[i] = ('absent', guess_letter)
        
        return feedback
    
    def _calculate_reward(self, feedback: List[Tuple[str, str]]) -> float:
        reward = 0
        for status, _ in feedback:
            if status == 'correct':
                reward += self.rewards['correct_position']
            elif status == 'present':
                reward += self.rewards['correct_letter']
        return reward
    
    def _update_board(self, guess: str, feedback: List[Tuple[str, str]]):
        for i, letter in enumerate(guess):
            letter_idx = ord(letter) - ord('A')
            self.board_state[self.current_attempt, i, letter_idx] = 1

            for status, letter in feedback:
                letter_idx = ord(letter) - ord('A')
                if status == 'absent':
                    self.keyboard_state[letter_idx] = 0

    def _get_state(self) -> np.ndarray:
        board_flat = self.board_state.flatten()
        return np.concatenate([
            board_flat,
            self.keyboard_state,
            [self.current_attempt / self.max_attempts]
        ])
    
    def render(self):
        print(f"Target Word: {self.target_word}")
        print(f"Attempt {self.current_attempt + 1}/{self.max_attempts}")
        print("\nBoard:")
        for i in range(self.max_attempts):
            row = ""
            for j in range(self.word_length):
                if np.any(self.board_state[i, j]):
                    letter_idx = np.argmax(self.board_state[i, j])
                    row += chr(letter_idx + ord('A'))
                else:
                    row += "_"
            print(row)
        
        print("\nKeyboard:")
        for i in range(26):
            if self.keyboard_state[i]:
                print(chr(ord('A') + i), end=" ")
        print("\n")

if __name__ == '__main__':
    env = WordleEnvironment()
    state = env.reset()
    env.render()

    next_state, reward, done = env.step('MONEY')
    print(f"Reward: {reward}")
    env.render()