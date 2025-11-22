import sys
import csv
import os

# Get the directory *one level up* from this file (since this script is inside Libraries)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

csv_file = os.path.join(base_dir, "Weighted_Test_Data.csv")


# Add the Libraries folder relative to this file
libraries_path = os.path.join(base_dir, "Libraries")
sys.path.append(libraries_path)

# Import custom graph and algorithm
from adjacency_list_graph import AdjacencyListGraph
from dijkstra import dijkstra

# Load CSV relative to this script
csv_file = os.path.join(base_dir, "Weighted_Test_Data.csv")

edges = []   # (station1, station2, weight)
stations = set()

with open(csv_file, 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        # Expected columns: LineName, Station1, Station2, Weight
        line = row[0].strip() if len(row) > 0 else ''
        s1 = row[1].strip() if len(row) > 1 and row[1] else ''
        s2 = row[2].strip() if len(row) > 2 and row[2] else ''
        weight = row[3].strip() if len(row) > 3 and row[3] else None

        if s1:
            stations.add(s1)
        if s2:
            stations.add(s2)

        if s1 and s2 and weight is not None:
            try:
                weight = float(weight)
                edges.append((s1, s2, weight))
            except ValueError:
                pass

# Build Graph
vertex_to_index = {v: i for i, v in enumerate(sorted(stations))}
index_to_vertex = {i: v for v, i in vertex_to_index.items()}

G = AdjacencyListGraph(len(stations), weighted=True)

added_edges = set()

for u, v, w in edges:
    u_idx = vertex_to_index[u]
    v_idx = vertex_to_index[v]
    edge_key = frozenset([u_idx, v_idx])
    if edge_key not in added_edges:
        try:
            G.insert_edge(u_idx, v_idx, w)
            added_edges.add(edge_key)
        except RuntimeError:
            pass

# Dijkstra Shortest Path
source = 'LineOne_One'
target = 'LineThree_Five'

source_idx = vertex_to_index[source]
target_idx = vertex_to_index[target]

dist, parent = dijkstra(G, source_idx)

# Reconstruct path
def get_path(parent, target_idx, index_to_vertex):
    path = []
    current = target_idx
    visited = set()
    while current is not None and current not in visited:
        path.insert(0, index_to_vertex[current])
        visited.add(current)
        if parent[current] == current or parent[current] == -1:
            break
        current = parent[current]
    return path

path = get_path(parent, target_idx, index_to_vertex)

print("=== Task 2a: Journey Planner ===\n")
print(f"Shortest-time path from {source} to {target}: {' → '.join(path)}")
print(f"Total travel time: {dist[target_idx]} minutes")