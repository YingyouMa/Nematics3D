from collections import defaultdict

class Graph:

    def __init__(self):
        self.graph = defaultdict(set)
    
    def add_edge(self, u, v):
        self.graph[u].add(v)
        self.graph[v].add(u)
    
    def remove_edge(self, u, v):
        self.graph[u].discard(v)
        self.graph[v].discard(u)
    
    def find_start_node(self):
        start_node = next((node for node in self.graph if len(self.graph[node]) % 2 == 1), None)
        if start_node is None:
            start_node = next((node for node in self.graph if len(self.graph[node]) > 0), None)       
        return start_node
    
    def hierholzer_algorithm(self, start_node):

        path = [start_node]  
        
        while True:
            u = path[-1]
            if self.graph[u]:
                v = list(self.graph[u])[0]
                path.append(v)
                self.remove_edge(u, v)
            else:
                break

        return path
    
    def find_path(self):

        paths = []

        while any(self.graph.values()):

            start_node = self.find_start_node()
            if start_node is None:
                return paths
            
            path = self.hierholzer_algorithm(start_node)
            paths.append(path)

        return paths