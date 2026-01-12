import networkx as nx
import matplotlib.pyplot as plt
from pymcm.core.base import BaseModel


class GraphModel(BaseModel):
    """
    Graph Theory Utilities.

    Includes:
    1. Shortest Path (Dijkstra).
    2. Minimum Spanning Tree (MST).
    """

    def __init__(self):
        super().__init__(name="Graph Model")
        self.G = nx.Graph()  # Undirected graph by default
        self.pos = None  # For plotting layout

    def add_edge(self, u, v, weight=1):
        """
        Add an edge between node u and node v with weight.
        """
        self.G.add_edge(u, v, weight=weight)

    def add_edges_from_matrix(self, matrix):
        """
        Add edges from an Adjacency Matrix.
        matrix[i][j] = weight. (0 or inf means no edge)
        """
        rows = len(matrix)
        cols = len(matrix[0])
        for i in range(rows):
            for j in range(i + 1, cols):  # Only upper triangle for undirected
                w = matrix[i][j]
                if w > 0 and w < float('inf'):
                    self.add_edge(i, j, weight=w)

    def shortest_path(self, source, target):
        """
        Find shortest path using Dijkstra.
        """
        try:
            path = nx.dijkstra_path(self.G, source, target, weight='weight')
            length = nx.dijkstra_path_length(self.G, source, target, weight='weight')
            print(f"[{self.name}] Shortest Path ({source}->{target}): {path}, Length: {length}")

            self._plot_graph(highlight_path=path, title=f"Shortest Path: {path}")
            return path, length
        except nx.NetworkXNoPath:
            print(f"[{self.name}] No path between {source} and {target}.")
            return None, float('inf')

    def mst(self):
        """
        Minimum Spanning Tree (Kruskal/Prim).
        Connects all nodes with minimum total weight.
        """
        T = nx.minimum_spanning_tree(self.G, weight='weight')
        edges = list(T.edges(data=True))
        total_weight = sum([d['weight'] for u, v, d in edges])

        print(f"[{self.name}] MST Found. Total Weight: {total_weight}")

        self._plot_graph(highlight_edges=T.edges(), title=f"MST (Weight={total_weight})")
        return edges

    def _plot_graph(self, highlight_path=None, highlight_edges=None, title="Graph"):
        plt.figure(figsize=(8, 6))

        if self.pos is None:
            self.pos = nx.spring_layout(self.G, seed=42)  # Force-directed layout

        # Draw all nodes and edges
        nx.draw_networkx_nodes(self.G, self.pos, node_color='lightblue', node_size=500)
        nx.draw_networkx_edges(self.G, self.pos, edge_color='gray', alpha=0.5)
        nx.draw_networkx_labels(self.G, self.pos)

        # Draw weights
        labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=labels)

        # Highlight Path (Nodes + Edges)
        if highlight_path:
            # Edges in path
            path_edges = list(zip(highlight_path, highlight_path[1:]))
            nx.draw_networkx_nodes(self.G, self.pos, nodelist=highlight_path, node_color='orange')
            nx.draw_networkx_edges(self.G, self.pos, edgelist=path_edges, edge_color='red', width=2)

        # Highlight Edges (MST)
        if highlight_edges:
            nx.draw_networkx_edges(self.G, self.pos, edgelist=highlight_edges, edge_color='green', width=2)

        plt.title(title)
        plt.axis('off')
        plt.show()

    def fit(self):
        pass

    def _predict_single(self):
        return 0

    def plot(self):
        self._plot_graph()