import networkx as nx
import matplotlib.pyplot as plt

# NetworkXを使用してグラフを作成する

# ノードの座標を適当に定義する
# この座標はグラフの構造を反映するためのものであり、ユーザーから提供されたものではない。
positions = {
    1: (0, 1),
    2: (1, 2),
    3: (2, 1),
    4: (1, 0),
    5: (3, 1),
    6: (2, 0),
    7: (3, 0)
}

# エッジを定義する
edges = [(1, 2), (1, 4), (2, 3), (3, 5), (4, 6), (6, 7), (5, 7)]

# グラフオブジェクトを作成し、ノードとエッジを追加する
G = nx.Graph()
G.add_nodes_from(positions.keys())
G.add_edges_from(edges)

# グラフを描画する
plt.figure(figsize=(8, 6))
nx.draw(G, pos=positions, with_labels=True, node_size=700, node_color="lightblue", linewidths=2)
plt.title("Graph Representation with NetworkX")
plt.show()

# エッジ上に新しいノードを追加する関数を定義する
def add_node_on_edge(graph, edge, positions, node_label):
    """
    graph: NetworkXのグラフオブジェクト
    edge: エッジ上にノードを追加するための元となるノードのタプル (node1, node2)
    positions: 各ノードの位置が格納された辞書
    node_label: 新しいノードに割り当てるラベル
    """
    # エッジを構成するノードの座標を取得する
    (x1, y1), (x2, y2) = positions[edge[0]], positions[edge[1]]
    
    # 二点の中間点を計算する
    new_position = ((x1 + x2) / 2, (y1 + y2) / 2)
    
    # 新しいノードとその位置を追加する
    graph.add_node(node_label)
    positions[node_label] = new_position
    
    # 元のエッジを削除し、新しいエッジを追加する
    graph.remove_edge(*edge)
    graph.add_edge(edge[0], node_label)
    graph.add_edge(node_label, edge[1])

# ノード8をエッジ(2, 3)上に追加する
add_node_on_edge(G, (2, 3), positions, 8)

# ノード9をエッジ(5, 7)上に追加する
add_node_on_edge(G, (5, 7), positions, 9)

# グラフを再描画する
plt.figure(figsize=(8, 6))
nx.draw(G, pos=positions, with_labels=True, node_size=700, node_color="lightblue", linewidths=2)
plt.title("Graph Representation with Additional Nodes on Edges")
plt.show()

# 各ノードに属性を追加するための関数を定義する
def set_node_attributes(graph, original_nodes, new_nodes):
    """
    graph: NetworkXのグラフオブジェクト
    original_nodes: 元々あったノードのリスト
    new_nodes: 新しく追加されたノードのリスト
    """
    # 元々あったノードには'type': 'original'という属性を設定する
    original_attributes = {node: {'type': 'original'} for node in original_nodes}
    nx.set_node_attributes(graph, original_attributes)

    # 新しく追加されたノードには'type': 'new'という属性を設定する
    new_attributes = {node: {'type': 'new'} for node in new_nodes}
    nx.set_node_attributes(graph, new_attributes)

# 元のノードのリスト
original_nodes = list(range(1, 8))

# 新しく追加したノードのリスト
new_nodes = [8, 9]

# ノードに属性を設定する
set_node_attributes(G, original_nodes, new_nodes)

# 動点をモデル化するために、動くオブジェクトのクラスを定義する
class MovingPoint:
    def __init__(self, graph, start_node):
        self.graph = graph
        self.current_node = start_node
        self.path = [start_node]

    def move(self, target_node):
        """
        動点を新しいノードに移動させるメソッド
        target_node: 移動先のノード
        """
        # 移動可能か確認する
        if target_node in nx.neighbors(self.graph, self.current_node):
            self.current_node = target_node
            self.path.append(target_node)
        else:
            raise ValueError(f"Can't move to node {target_node}, it's not a neighbor of node {self.current_node}")

# 新しい動点オブジェクトを作成する
moving_point = MovingPoint(G, start_node=1)

# 動点をノード2へ移動させる
moving_point.move(2)

# 動点の現在位置と経路を表示する
print("Current location of the moving point:", moving_point.current_node)
print("Path of the moving point:", moving_point.path)

# 現在のグラフの状態を表示する
plt.figure(figsize=(8, 6))
node_colors = ['lightblue' if G.nodes[n]['type'] == 'original' else 'lightgreen' for n in G.nodes]
nx.draw(G, pos=positions, with_labels=True, node_size=700, node_color=node_colors, linewidths=2)
plt.title("Graph with Node Attributes and Moving Point")
plt.show()
