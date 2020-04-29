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

    def reward(self, state, action: int):
        actions = []
        for _ in range(self.agent_number):
            actions.append(action % self.action_size)
            action = action // self.action_size

        for a_idx in range(self.agent_number):
            vtx = self.vertex_list[state[2*a_idx]]

            current_action = actions[a_idx]
            if current_action == self.action_size-1:
                continue
            elif current_action < vtx.degree:
                vtx.edges[current_action].increment_counter()
            else:
                self.flush_counters()
                return 0
        
        reward = 1/max(edge.counter for edge in self.edge_list)
        self.flush_counters
        return reward


g = Graph(2, [
    ('a', 'b'),
    ('a', 'b')
])