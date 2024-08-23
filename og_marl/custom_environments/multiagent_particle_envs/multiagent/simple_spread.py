import numpy as np
from .core import World, Agent, Landmark
from .scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, num_agents=3):
        world = World()
        # set any world properties first
        world.dim_c = 2
        # num_agents = 3
        num_landmarks = num_agents
        # print ('\033[1;32m[simple_spread] num_agents: {}, num_landmarks: {}\033[1;0m'.format(num_agents, num_landmarks))
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = "agent %d" % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        if len(world.agents) <= 3:
            boundary = 1
        elif len(world.agents) <= 10:
            boundary = 2
        else:
            return NotImplementedError
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-boundary, boundary, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-boundary, boundary, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    # def reset_world(self, world):
    #     # random properties for agents
    #     for i, agent in enumerate(world.agents):
    #         agent.color = np.array([0.35, 0.35, 0.85])
    #     # random properties for landmarks
    #     for i, landmark in enumerate(world.landmarks):
    #         landmark.color = np.array([0.25, 0.25, 0.25])
    #     # set random initial states
    #     for agent in world.agents:
    #         agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
    #         agent.state.p_vel = np.zeros(world.dim_p)
    #         agent.state.c = np.zeros(world.dim_c)
    #     for i, landmark in enumerate(world.landmarks):
    #         landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
    #         landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        num_agents_on_landmarks = []  # modified by ling

        for l in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents
            ]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1

            # modified by ling
            num_agents_on_l = 0
            for dist in dists:
                if dist < 0.1:
                    num_agents_on_l += 1
            num_agents_on_landmarks.append(num_agents_on_l)

        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent) and a.name != agent.name:  # modified by ling
                    rew -= 1
                    collisions += 1

        return (rew, collisions, min_dists, occupied_landmarks, num_agents_on_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for l in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents
            ]
            # rew -= min(dists)
            rew += min(1.0 / min(dists), 10.0)  # modified by ling
            # print ('landmark [{}]: {}'.format(l.name, 1. / min(dists), 10))
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent) and a.name != agent.name:  # modified by ling
                    rew -= 5  # modified by ling
        return rew

    # def reward(self, agent, world):
    #     # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
    #     rew = 0
    #     for l in world.landmarks:
    #         dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
    #         if min(dists) < 0.1:
    #             rew += 10
    #         else:
    #             # rew -= 0.1 * min(dists)
    #             rew -= 0.01 * min(dists)
    #             # rew -= min(dists)
    #         # rew += min(1. / min(dists), 10.) # modified by ling
    #         # print ('landmark [{}]: {}'.format(l.name, 1. / min(dists), 10))
    #     if agent.collide:
    #         for a in world.agents:
    #             if self.is_collision(a, agent) and a.name != agent.name: # modified by ling
    #                 rew -= 1
    #     return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)

        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        # print ('p_vel: {}, p_pos: {}, entity_pos: {}, other_pos: {}, comm: {}'.format(agent.state.p_vel, agent.state.p_pos, entity_pos, other_pos, comm))
        # p_vel: [1.54798273 1.80932997]
        # p_pos: [1.64310331 2.93676854]
        # entity_pos: [array([-1.69713533, -2.61021445]), array([-1.0316461 , -3.43080753]), array([-2.48395644, -2.47124733])]
        # other_pos: [array([-0.5826037 , -2.28894684]), array([-1.20824731, -1.75810608])]
        # comm: [array([0., 0.]), array([0., 0.])]

        return np.concatenate(
            [agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm
        )
