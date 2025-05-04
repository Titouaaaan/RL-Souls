import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import copy


def build_net(layer_shape, activation, output_activation):
	'''Build networks with For loop'''
	layers = []
	for j in range(len(layer_shape)-1):
		act = activation if j < len(layer_shape)-2 else output_activation
		layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()]
	return nn.Sequential(*layers)

class Q_Net(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape):
		super(Q_Net, self).__init__()
		layers = [state_dim] + list(hid_shape) + [action_dim]
		self.Q = build_net(layers, nn.ReLU, nn.Identity)
	def forward(self, s):
		q = self.Q(s)
		return q


class Duel_Q_Net(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape):
		super(Duel_Q_Net, self).__init__()
		layers = [state_dim] + list(hid_shape)
		self.hidden = build_net(layers, nn.ReLU, nn.ReLU)
		self.V = nn.Linear(hid_shape[-1], 1)
		self.A = nn.Linear(hid_shape[-1], action_dim)

	def forward(self, s):
		s = self.hidden(s)
		Adv = self.A(s)
		V = self.V(s)
		Q = V + (Adv - torch.mean(Adv, dim=-1, keepdim=True))  # Q(s,a)=V(s)+A(s,a)-mean(A(s,a))
		return Q

class EnhancedDuel_Q_Net(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape):
        super(EnhancedDuel_Q_Net, self).__init__()
        layers = [state_dim] + list(hid_shape)

        hidden_layers = []
        for i in range(len(layers) - 1):
            hidden_layers.append(nn.Linear(layers[i], layers[i + 1]))
            hidden_layers.append(nn.ReLU())
            hidden_layers.append(nn.BatchNorm1d(layers[i + 1]))
            hidden_layers.append(nn.Dropout(0.5))

        self.hidden = nn.Sequential(*hidden_layers)

        # Value stream
        self.V = nn.Sequential(
            nn.Linear(layers[-1], hid_shape[-1]),
            nn.ReLU(),
            nn.Linear(hid_shape[-1], 1)
        )

        # Advantage stream
        self.A = nn.Sequential(
            nn.Linear(layers[-1], hid_shape[-1]),
            nn.ReLU(),
            nn.Linear(hid_shape[-1], action_dim)
        )

    def forward(self, s):
        s = self.hidden(s)
        V = self.V(s)
        A = self.A(s)
        Q = V + (A - torch.mean(A, dim=-1, keepdim=True))  # Q(s,a)=V(s)+A(s,a)-mean(A(s,a))
        return Q


class DQN_agent(object):
	def __init__(self, **kwargs):
		# Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
		self.__dict__.update(kwargs)
		self.tau = 0.005
		self.replay_buffer = PrioritizedReplayBuffer(self.state_dim, self.dvc, max_size=int(1e6))
		if self.Duel:
			self.q_net = Duel_Q_Net(self.state_dim, self.action_dim, (self.net_width,self.net_width)).to(self.dvc)
		elif self.Enhanced:
			self.q_net = EnhancedDuel_Q_Net(self.state_dim, self.action_dim, (self.net_width,self.net_width)).to(self.dvc)
		else:
			self.q_net = Q_Net(self.state_dim, self.action_dim, (self.net_width, self.net_width)).to(self.dvc)
		self.q_net_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
		self.q_target = copy.deepcopy(self.q_net)
		# Freeze target networks with respect to optimizers (only update via polyak averaging)
		for p in self.q_target.parameters(): p.requires_grad = False

		self.epsilon = kwargs.get('epsilon', 1.0)  
		self.epsilon_decay = kwargs.get('epsilon_decay', 0.995)
		self.epsilon_min = kwargs.get('epsilon_min', 0.01)
		self.debugging = kwargs.get('debugging')


	def select_action(self, state, deterministic):#only used when interact with the env
		with torch.no_grad():
			# state = flatten_observation(state)
			state = torch.FloatTensor(state.reshape(1, -1)).to(self.dvc)
			if deterministic:
				a = self.q_net(state).argmax().item()
			else:
				if np.random.rand() < self.epsilon:
					a = np.random.randint(0,self.action_dim)
				else:
					a = self.q_net(state).argmax().item()
		return a


	def train(self):
		if False:
			s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)

			'''Compute the target Q value'''
			with torch.no_grad():
				if self.Double:
					argmax_a = self.q_net(s_next).argmax(dim=1).unsqueeze(-1)
					max_q_next = self.q_target(s_next).gather(1,argmax_a)
				else:
					max_q_next = self.q_target(s_next).max(1)[0].unsqueeze(1)
				target_Q = r + (~dw) * self.gamma * max_q_next #dw: die or win

			# Get current Q estimates
			current_q = self.q_net(s)
			current_q_a = current_q.gather(1,a)

			# print('Q-Value: ', current_q_a)

			q_loss = F.mse_loss(current_q_a, target_Q)
			self.q_net_optimizer.zero_grad()
			q_loss.backward()

			""" if self.debugging:
				print(f"Initial Q-values mean: {current_q_a.mean().item()}, min: {current_q_a.min().item()}, max: {current_q_a.max().item()}")
				print(f"Target Q-values mean: {target_Q.mean().item()}, min: {target_Q.min().item()}, max: {target_Q.max().item()}")
				print(f"Gradient Mean: {sum(p.grad.abs().mean().item() for p in self.q_net.parameters() if p.grad is not None) / len(list(self.q_net.parameters()))}") """

			# Uncomment this if the gradient clipping between -1 and 1 doesn't work
			# torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
			
			# Clip gradients between -1 and 1
			for param in self.q_net.parameters():
				if param.grad is not None:
					param.grad.data.clamp_(-1, 1)

			self.q_net_optimizer.step()

			# Update the frozen target models 
			for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			
			return q_loss.item()
		
		elif True:
			s, a, r, s_next, dw, indices, weights = self.replay_buffer.sample(self.batch_size)

			'''Compute the target Q value'''
			with torch.no_grad():
				if self.Double:
					argmax_a = self.q_net(s_next).argmax(dim=1).unsqueeze(-1)
					max_q_next = self.q_target(s_next).gather(1, argmax_a)
				else:
					max_q_next = self.q_target(s_next).max(1)[0].unsqueeze(1)
				target_Q = r + (~dw) * self.gamma * max_q_next  # dw: die or win

			# Get current Q estimates
			current_q = self.q_net(s)
			current_q_a = current_q.gather(1, a)

			# Compute TD errors for updating priorities
			td_errors = torch.abs(target_Q - current_q_a).detach().squeeze().cpu().numpy()

			# Compute loss with importance-sampling weights
			q_loss = (weights * F.mse_loss(current_q_a, target_Q, reduction='none')).mean()
			self.q_net_optimizer.zero_grad()
			q_loss.backward()

			# Clip gradients between -1 and 1
			for param in self.q_net.parameters():
				if param.grad is not None:
					param.grad.data.clamp_(-1, 1)

			self.q_net_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			# Update priorities in the replay buffer
			self.replay_buffer.update_priorities(indices, td_errors + 1e-6)  # Adding a small constant to avoid zero priorities

			return q_loss.item()


	def save(self, algo, EnvName, steps, checkpoint=True):
		if checkpoint:
			save_dict = {
				'q_net_state_dict': self.q_net.state_dict(),
				'q_target_state_dict': self.q_target.state_dict(),
				'optimizer_state_dict': self.q_net_optimizer.state_dict(),
				'epsilon': self.epsilon,
				'steps': steps,
			}
			torch.save(save_dict, f"./model/{algo}_{EnvName}_checkpoint.pth")
		else:
			# Save final model (for deployment)
			torch.save(self.q_net.state_dict(), f"./model/{algo}_{EnvName}.pth")

	def load(self, algo, EnvName, checkpoint=True):
		if checkpoint:
			checkpoint = torch.load(f"./model/{algo}_{EnvName}_checkpoint.pth", map_location=self.dvc, weights_only=True)
			self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
			self.q_target.load_state_dict(checkpoint['q_target_state_dict'])
			self.q_net_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			self.epsilon = checkpoint['epsilon']
			return checkpoint['steps']  # Return 0 if 'steps' key is missing
		else:
			self.q_net.load_state_dict(torch.load(f"./model/{algo}_{EnvName}.pth", map_location=self.dvc))
			self.q_target.load_state_dict(torch.load(f"./model/{algo}_{EnvName}.pth", map_location=self.dvc))
			return 0


class ReplayBuffer(object):
	def __init__(self, state_dim, dvc, max_size=int(1e6)):
		self.max_size = max_size
		self.dvc = dvc
		self.ptr = 0
		self.size = 0

		self.s = torch.zeros((max_size, state_dim),dtype=torch.float,device=self.dvc)
		self.a = torch.zeros((max_size, 1),dtype=torch.long,device=self.dvc)
		self.r = torch.zeros((max_size, 1),dtype=torch.float,device=self.dvc)
		self.s_next = torch.zeros((max_size, state_dim),dtype=torch.float,device=self.dvc)
		self.dw = torch.zeros((max_size, 1),dtype=torch.bool,device=self.dvc)

	def add(self, s, a, r, s_next, dw):
		self.s[self.ptr] = torch.from_numpy(s).to(self.dvc)
		self.a[self.ptr] = a
		self.r[self.ptr] = r
		self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.dvc)
		self.dw[self.ptr] = dw

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))
		return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]
	
	def save(self, path):
		save_dict = {
			's': self.s,
            'a': self.a,
            'r': self.r,
            's_next': self.s_next,
            'dw': self.dw,
            'ptr': self.ptr,
            'size': self.size
		}
		torch.save(save_dict, path)

	def load(self, path):
		data = torch.load(path, map_location=self.dvc, weights_only=True)
		self.s = data['s']
		self.a = data['a']
		self.r = data['r']
		self.s_next = data['s_next']
		self.dw = data['dw']
		self.ptr = data['ptr']
		self.size = data['size']

class PrioritizedReplayBuffer(object):
    def __init__(self, state_dim, dvc, max_size=int(1e6), alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001):
        self.max_size = max_size
        self.dvc = dvc
        self.ptr = 0
        self.size = 0
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.max_priority = 1.0

        self.s = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.dvc)
        self.a = torch.zeros((max_size, 1), dtype=torch.long, device=self.dvc)
        self.r = torch.zeros((max_size, 1), dtype=torch.float, device=self.dvc)
        self.s_next = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.dvc)
        self.dw = torch.zeros((max_size, 1), dtype=torch.bool, device=self.dvc)
        self.priorities = np.zeros((max_size,), dtype=np.float32)

    def add(self, s, a, r, s_next, dw, priority=None):
        self.s[self.ptr] = torch.from_numpy(s).to(self.dvc)
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.dvc)
        self.dw[self.ptr] = dw

        if priority is None:
            priority = self.max_priority

        self.priorities[self.ptr] = priority
        self.max_priority = max(self.max_priority, priority)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        if self.size < batch_size:
            raise ValueError("Not enough samples in the buffer")

        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self.size, batch_size, p=probs)
        weights = (self.size * probs[indices]) ** -self.beta
        weights /= weights.max()

        self.beta = min(self.beta + self.beta_increment_per_sampling, 1.0)

        return (
            self.s[indices],
            self.a[indices],
            self.r[indices],
            self.s_next[indices],
            self.dw[indices],
            indices,
            torch.tensor(weights, dtype=torch.float, device=self.dvc)
        )

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def save(self, path):
        save_dict = {
            's': self.s.cpu(),
            'a': self.a.cpu(),
            'r': self.r.cpu(),
            's_next': self.s_next.cpu(),
            'dw': self.dw.cpu(),
            'priorities': self.priorities,
            'ptr': self.ptr,
            'size': self.size,
            'alpha': self.alpha,
            'beta': self.beta,
            'max_priority': self.max_priority
        }
        torch.save(save_dict, path)

    def load(self, path):
        data = torch.load(path, map_location=self.dvc)
        self.s = data['s'].to(self.dvc)
        self.a = data['a'].to(self.dvc)
        self.r = data['r'].to(self.dvc)
        self.s_next = data['s_next'].to(self.dvc)
        self.dw = data['dw'].to(self.dvc)
        self.priorities = data['priorities']
        self.ptr = data['ptr']
        self.size = data['size']
        self.alpha = data['alpha']
        self.beta = data['beta']
        self.max_priority = data['max_priority']