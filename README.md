# [PYTORCH] DEEP Q LEARNING(DQN) for Playing Snake

## Introduction

Here is my python source code for training an agent to play Tetris. It could be seen as a very basic example of Reinforcement Learning's application.

![Demo SNAKE DEEP Q LEARNING](output_video.gif)

## How to use my code
With my code, you can:
- Train your model by running `python train.py`
- Test your trained model by running `python test.py`

## IDEA
- Deep Q-Learning: Trains an AI agent to play Snake autonomously.
- Neural Network Architecture: Uses fully connected layers with Layer Normalization for stable training.
- State Representation: The game state is encoded with 13 input features, including snake direction, danger detection, and food position.
 
## Requirements
- `pygame==2.6.1`
- `numpy==1.26.3`
- `torch==2.4.1+cu118`

## Installation

First, install all dependencies using:

```bash
pip install -r requirements.txt