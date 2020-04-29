from typing import Any, Dict, List, Tuple
import random

class Edge:
    def __init__(self, index: int):
        self.index = index
        self.counter = 0
    
    def flush_counter(self):
        self.counter = 0

    def increment_counter(self):
        self.counter += 1

class Vertex:
    def __init__(self, identifier, index: int):
        self.identifier = identifier
        self.index = index
        self.edges: List[Edge] = []

    def add_edge(self, edge: Edge):
        self.edges.append(edge)

    def degree(self):
        return len(self.edges)
    
    def random_neighbor(self):
        return random.choice(self.edges)

class Graph:
    def __init__(self, agent_number, edge_data: List[Tuple[Any, Any]]):
        self.agent_number = agent_number
        self.vertex_list: List[Vertex] = []
        self.edge_list: List[Edge] = []

        for ind_a, ind_b in edge_data:
            vtx_a = self.get_or_create_vertex(ind_a)
            vtx_b = self.get_or_create_vertex(ind_b)

            new_edge = Edge(len(self.edge_list))
            self.edge_list.append(new_edge)

            vtx_a.add_edge(new_edge)
            vtx_b.add_edge(new_edge)
        
        self.action_size = 1 + max(vtx.degree() for vtx in self.vertex_list)

    def get_vertex_index(self, identifier):
        for vtx in self.vertex_list:
            if vtx.identifier == identifier:
                return vtx.index
        
        raise Exception('Vertex not found')

    def get_mapping_shape(self):
        return 2*self.agent_number, self.action_size**self.agent_number
            
    def get_or_create_vertex(self, identifier):

        alredy_existing = False
        for vtx in self.vertex_list:
            if vtx.identifier == identifier:
                return vtx

        new_vtx = Vertex(identifier, len(self.vertex_list))
        self.vertex_list.append(new_vtx)
        return new_vtx

    def get_random_state(self):
        state: List[int] = []
        for _ in range(self.agent_number):
            rnd_pos = random.choice(self.vertex_list)
            rnd_dst = rnd_pos.random_neighbor()

            state.append(rnd_pos.index)
            state.append(rnd_dst.index)
        return state

    def flush_counters(self):
        for edge in self.edge_list:
            edge.flush_counter()

    def decode_action(self, action: int) -> List[int]:
        actions = []
        for _ in range(self.agent_number):
            actions.append(action % self.action_size)
            action = action // self.action_size
        return actions

    def print_action_summary(self, state, action: int) -> None:
        print('=====================')
        actions = self.decode_action(action)

        for a_idx in range(self.agent_number):
            pos = self.vertex_list[state[2*a_idx  ]]
            dst = self.vertex_list[state[2*a_idx+1]]

            current_action = actions[a_idx]
            if current_action == self.action_size-1:
                if pos == dst:
                    continue
                else: # If agent is falsely waiting
                    print(f'Agent {a_idx} is falsely wainting')
            elif current_action < pos.degree():
                edge = pos.edges[current_action]
                print(f'Agent {a_idx} --> {edge.index}')
                edge.increment_counter()
                invalid_edge = edge not in dst.edges
                if invalid_edge: # If agent has chosen a edge pointing to a wrong vertex
                    print(f'Agent {a_idx} did not selected an edge pointing to the destination')

            else: # If edge index out of range (invalid action)
                print(f'Agent {a_idx} selected an invalid action')
        
        max_c = max(edge.counter for edge in self.edge_list)
        print(f'Maximal number of agents on one road: {max_c}')
        self.flush_counters()


    def reward(self, state, action: int) -> float:
        actions = self.decode_action(action)
        # print(actions)

        for a_idx in range(self.agent_number):
            pos = self.vertex_list[state[2*a_idx  ]]
            dst = self.vertex_list[state[2*a_idx+1]]

            current_action = actions[a_idx]
            if current_action == self.action_size-1:
                if pos == dst:
                    continue
                else: # If agant is falsly waiting
                    self.flush_counters()
                    return 0
            elif current_action < pos.degree():
                edge = pos.edges[current_action]
                edge.increment_counter()
                invalid_edge = edge not in dst.edges
                if invalid_edge: # If agent has chosen a edge pointing to a wrong vertex
                    self.flush_counters()
                    return 0
            else: # If edge index out of range (invalid action)
                self.flush_counters()
                return 0
        
        max_c = max(edge.counter for edge in self.edge_list)
        reward = 1 if max_c == 0 else 1/max_c
        self.flush_counters()
        return reward

