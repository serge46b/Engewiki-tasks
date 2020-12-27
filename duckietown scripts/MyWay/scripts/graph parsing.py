import json
jsonGraph = json.load(open("duckietown map graph exemple.json"))
graph = []
for i in jsonGraph["graph"]:
    graph.append(i)
x = 1
y = 1
index = None
for i in range(len(graph)):
    print(graph[i])
    if graph[i]["cord_x"] == x and graph[i]["cord_y"] == y:
        index = i
id = graph[index]["id"]
print(id)
index = None
for i in range(len(graph)):
    print(graph[i])
    if graph[i]["id"] == id:
        index = i
ids = []
for i in range(len(graph[index]["neighbors_data"])):
    ids.append(graph[index]["neighbors_data"][i]["id"])
print(ids)
