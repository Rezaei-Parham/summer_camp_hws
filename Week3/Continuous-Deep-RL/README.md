# DDPG Mountain Car Continuous Control

This is a reinforcement learning project that uses the DDPG (Deep Deterministic Policy Gradient) algorithm to solve the MountainCarContinuous-v0 environment from the OpenAI Gym.

## Dependencies

The following dependencies are required to run the project:

- gym
- math
- random
- numpy
- matplotlib
- torch
- torchvision

You can install the dependencies using pip:

```
pip install gym numpy matplotlib torch torchvision
```

## Usage

To run the project, execute the Python script. The script will train the DDPG agent on the MountainCarContinuous-v0 environment and display the total reward obtained in each episode.

```python
python ddpg_mountain_car.py
```

## Additional Notebooks

This project also includes Jupyter notebooks for two other reinforcement learning algorithms: REINFORCE and Actor-Critic. You can find the notebooks in the following files:

- REINFORCE: Reinforce.ipynb
- Actor-Critic: ACtorCritic.ipynb

These notebooks provide alternative implementations of reinforcement learning algorithms for the MountainCarContinuous-v0 environment.

## Description

The project consists of the following components:

### Actor and Critic Networks

The actor and critic networks are implemented as PyTorch modules. The actor network takes the state as input and outputs the action to be taken in that state. The critic network takes both the state and action as inputs and outputs the corresponding Q-value.

### Memory

The memory class is used to store the experiences of the agent. It implements a replay buffer using a deque data structure.

### DDPG Trainer

The DDPG trainer class is responsible for training the DDPG agent. It initializes the actor and critic networks, defines the loss functions and optimizers, and performs the necessary updates to the networks based on the collected experiences.

### Training Loop

The script contains a training loop that iterates over a fixed number of episodes. In each episode, the agent interacts with the environment, collects experiences, and updates its networks based on the collected experiences. The total reward obtained in each episode is recorded for visualization purposes.

### Visualization

The script includes functions to visualize the training progress. It can generate a video of the trained agent's performance in the environment and display it in the Jupyter Notebook.

## Video of Trained Agent

After training the agent for a few epochs, a video of the agent's performance in the environment is available. You can view the video by executing the following code:

```python
show_video('MountainCarContinuous-v0')
```

Here is a video of the agent's behavior in the MountainCarContinuous-v0 environment after a few epochs.

![trained agent video](https://github.com/cyberrosa/summer_camp_hws/assets/94908814/bee6a66d-0236-406a-940c-33562b38b370)


## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

This project is based on the DDPG algorithm and the MountainCarContinuous-v0 environment from the OpenAI Gym. The implementation is inspired by various online tutorials and resources.
