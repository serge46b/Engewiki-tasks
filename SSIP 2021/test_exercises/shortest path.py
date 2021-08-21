import networkx as nx

graph = nx.tutte_graph()
req_node = 7
queue = [0]
done = []
found_node = None


while len(done) < len(graph.nodes):
    for i in queue:
        if i == req_node:
            found_node = i
            break
        else:
            for a in graph.adj[i]:
                if a not in queue and a not in done:
                    queue.append(a)
            queue.remove(i)
            done.append(i)
    if found_node is not None:
        break

print(found_node)
