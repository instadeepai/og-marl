import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, num_adversaries=4, num_good_agents=2, num_landmarks=1):
        world = World()
        # set any world properties first
        world.dim_c = 4

        num_adversaries, num_good_agents = 3, 1

        world.scenario_name = "simple_world"
        world.num_adversaries, world.num_good_agents = num_adversaries, num_good_agents
        world.num_landmarks = num_landmarks
        num_agents = num_adversaries + num_good_agents
        world.num_agents = num_agents

        num_food = 2
        num_forests = 2
        world.num_food, world.num_forests = num_food, num_forests

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = "agent %d" % i
            agent.collide = True
            agent.leader = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.075 if agent.adversary else 0.045
            agent.accel = 3.0 if agent.adversary else 4.0
            agent.max_speed = 1.0 if agent.adversary else 1.3

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False

        world.food = [Landmark() for i in range(num_food)]
        for i, landmark in enumerate(world.food):
            landmark.name = "food %d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.03
            landmark.boundary = False

        world.forests = [Landmark() for i in range(num_forests)]
        for i, landmark in enumerate(world.forests):
            landmark.name = "forest %d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.3
            landmark.boundary = False
        world.landmarks += world.food
        world.landmarks += world.forests

        # make initial conditions
        self.reset_world(world)

        return world

    def set_boundaries(self, world):
        boundary_list = []
        landmark_size = 1
        edge = 1 + landmark_size
        num_landmarks = int(edge * 2 / landmark_size)
        for x_pos in [-edge, edge]:
            for i in range(num_landmarks):
                l = Landmark()
                l.state.p_pos = np.array([x_pos, -1 + i * landmark_size])
                boundary_list.append(l)

        for y_pos in [-edge, edge]:
            for i in range(num_landmarks):
                l = Landmark()
                l.state.p_pos = np.array([-1 + i * landmark_size, y_pos])
                boundary_list.append(l)

        for i, l in enumerate(boundary_list):
            l.name = "boundary %d" % i
            l.collide == True
            l.movable = False
            l.boundary = True
            l.color = np.array([0.75, 0.75, 0.75])
            l.size = landmark_size
            l.state.p_vel = np.zeros(world.dim_p)

        return boundary_list

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = (
                np.array([0.45, 0.95, 0.45])
                if not agent.adversary
                else np.array([0.95, 0.45, 0.45])
            )
            agent.color -= np.array([0.3, 0.3, 0.3]) if agent.leader else np.array([0, 0, 0])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        for i, landmark in enumerate(world.food):
            landmark.color = np.array([0.15, 0.15, 0.65])
        for i, landmark in enumerate(world.forests):
            landmark.color = np.array([0.6, 0.9, 0.6])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, landmark in enumerate(world.food):
            landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, landmark in enumerate(world.forests):
            landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def manual_reset_specific_world(self, world, specific_init_state):
        # set color for agents
        for i, agent in enumerate(world.agents):
            agent.color = (
                np.array([0.45, 0.95, 0.45])
                if not agent.adversary
                else np.array([0.95, 0.45, 0.45])
            )
            agent.color -= np.array([0.3, 0.3, 0.3]) if agent.leader else np.array([0, 0, 0])
        # set color for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        for i, landmark in enumerate(world.food):
            landmark.color = np.array([0.15, 0.15, 0.65])
        for i, landmark in enumerate(world.forests):
            landmark.color = np.array([0.6, 0.9, 0.6])

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, landmark in enumerate(world.food):
            landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, landmark in enumerate(world.forests):
            landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        # set random initial states
        for agent_idx, agent in enumerate(world.agents):
            curr_agent_init_state = specific_init_state[agent_idx * 8 : (agent_idx + 1) * 8]
            agent.state.p_pos = curr_agent_init_state[:2]
            agent.state.p_vel = curr_agent_init_state[2:4]
            agent.state.c = curr_agent_init_state[4:]
        for i, landmark in enumerate(world.landmarks):
            curr_landmark_init_state = specific_init_state[
                len(world.agents) * 8 + i * 4 : len(world.agents) * 8 + (i + 1) * 4
            ]
            landmark.state.p_pos = curr_landmark_init_state[:2]
            landmark.state.p_vel = curr_landmark_init_state[2:]

    def benchmark_data(self, agent, world):
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = (
            self.adversary_reward(agent, world)
            if agent.adversary
            else self.agent_reward(agent, world)
        )
        return main_reward

    def outside_boundary(self, agent):
        if (
            agent.state.p_pos[0] > 1
            or agent.state.p_pos[0] < -1
            or agent.state.p_pos[1] > 1
            or agent.state.p_pos[1] < -1
        ):
            return True
        else:
            return False

    def agent_reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        rew = 0
        shape = True
        adversaries = self.adversaries(world)
        if shape:
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 5

        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)  # 1 + (x - 1) * (x - 1)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= 2 * bound(x)

        for food in world.food:
            if self.is_collision(agent, food):
                rew += 2

        rew += 0.05 * min(
            [
                np.sqrt(np.sum(np.square(food.state.p_pos - agent.state.p_pos)))
                for food in world.food
            ]
        )

        return rew

    def adversary_reward(self, agent, world):
        rew = 0
        shape = True
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:
            for adv in adversaries:
                rew -= 0.1 * min(
                    [np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents]
                )
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 5
        return rew

    def observation2(self, agent, world):
        return None
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        food_pos = []
        for entity in world.food:
            if not entity.boundary:
                food_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)
        return np.concatenate(
            [agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel
        )

    def observation(self, agent, world):
        # get relative positions from the agent to the landmarks (in this agent's reference frame)
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # whether the agent is inside one of the forests
        in_forest = [np.array([-1]), np.array([-1])]
        inf1 = False
        inf2 = False
        if self.is_collision(agent, world.forests[0]):
            in_forest[0] = np.array([1])
            inf1 = True
        if self.is_collision(agent, world.forests[1]):
            in_forest[1] = np.array([1])
            inf2 = True

        # get relative positions from the agent to the foods (in this agent's reference frame)
        food_pos = []
        for entity in world.food:
            if not entity.boundary:
                food_pos.append(entity.state.p_pos - agent.state.p_pos)

        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent:
                continue

            comm.append(other.state.c)

            oth_f1 = self.is_collision(other, world.forests[0])
            oth_f2 = self.is_collision(other, world.forests[1])
            if (
                (inf1 and oth_f1)
                or (inf2 and oth_f2)
                or (not inf1 and not oth_f1 and not inf2 and not oth_f2)
                or agent.leader
            ):  # without forest vis
                other_pos.append(other.state.p_pos - agent.state.p_pos)
                if not other.adversary:
                    other_vel.append(other.state.p_vel)
            else:
                other_pos.append([0, 0])
                if not other.adversary:
                    other_vel.append([0, 0])

        # to tell the predators when the preys are in the forest
        prey_forest = []
        ga = self.good_agents(world)
        for a in ga:
            if any([self.is_collision(a, f) for f in world.forests]):
                prey_forest.append(np.array([1]))
            else:
                prey_forest.append(np.array([-1]))

        # to tell leader when pred are in forest
        prey_forest_lead = []
        for f in world.forests:
            if any([self.is_collision(a, f) for a in ga]):
                prey_forest_lead.append(np.array([1]))
            else:
                prey_forest_lead.append(np.array([-1]))

        comm = [world.agents[0].state.c]

        return np.concatenate(
            [agent.state.p_vel]
            + [agent.state.p_pos]
            + entity_pos
            + other_pos
            + other_vel
            + in_forest
        )
