import numpy as np
import torch
import matplotlib.pyplot as plt
from time import time
from torch import nn, optim
from graph import Graph


g = Graph(2, [
    ('a', 'b'),
    ('a', 'b')
])

vtx_a = g.get_vertex_index('a')
vtx_b = g.get_vertex_index('b')
inp_size, out_size = g.get_mapping_shape()


hidden_size = 10

model = nn.Sequential(
    nn.Linear(inp_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, out_size),
    nn.Softmax(dim=1)
)

print(model)

def train(batch_size, batch_number):
    print('=====================')

    optimizer = optim.Adam(model.parameters(), lr=0.003)
    total_rewards = np.zeros((batch_number,))
    for b in range(batch_number):
        optimizer.zero_grad()

        states = [g.get_random_state() for _ in range(batch_size)]
        state_tensor = torch.FloatTensor(states)

        action_probs = model(state_tensor)
        action_probs_np = action_probs.detach().numpy()
        actions = [np.random.choice(out_size, p=action_probs_np[c]) for c in range(batch_size)]
        action_tensor = torch.LongTensor(actions)

        rewards = [g.reward(s,a) for s,a in zip(states, actions)]
        reward_tensor = torch.FloatTensor(rewards)


        total_rewards[b] = np.mean(rewards)
        print("\rbatch: {} Average reward: {:.2f}".format(
                    b+1, total_rewards[b]), end="")


        logprob = torch.log(action_probs)
        selected_logprobs = reward_tensor * logprob[np.arange(len(action_tensor)), action_tensor]
        loss = -selected_logprobs.mean()

        # print(logprob)
        # print(selected_logprobs)

        # Calculate gradients
        loss.backward()
        # Apply gradients
        optimizer.step()
    
    print('')
    return total_rewards

def test():
    s = [vtx_a, vtx_b,
        vtx_a, vtx_b]
    action_probs = model(torch.FloatTensor([s])).detach().numpy()
    action = np.random.choice(out_size, p=action_probs[0])

    g.print_action_summary(s, action)


test()
rewards = train(batch_size=200, batch_number=300)
test()

plt.figure(figsize=(12,8))
plt.plot(rewards)
plt.ylabel('Mean batch reward')
plt.xlabel('Batch')
plt.show()