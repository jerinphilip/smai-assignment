import sys

inp, hidden, out = list(map(lambda x: list(range(1, int(x)+1)), sys.argv[1:]))
# layer-i-i


def draw_between(xs, ys, n):
    for i in xs:
        for j in ys:
            print("layer%d%d -> layer%d%d [arrowsize=.2, weight=1.]"%(n, i, n+1, j))


def cluster(xs, i, color, ltype):
    outstr = "color=white;\nnode [style=solid,color=%s, shape=circle];\nlabel=\"%s\";\n"%(color, ltype)
    node_id = lambda x: "layer%d%d"%(i, x)
    nodes = ' '.join(map(node_id, xs))
    final = "subgraph cluster_%d {\n %s; \n}"%(i, outstr+nodes)
    print(final)
    
print("digraph G{")
props = ["rankdir=LR", "splines=line", "nodesep=.1;" , "ranksep=1", "node [label=\"\"];"]
print('\n'.join(props))
cluster(inp, 1, "red2", "Input Layer")
cluster(hidden[2:], 2, "blue4", "Hidden Layer")
cluster(out, 3, "green2", "Output Layer")
draw_between(inp, hidden[1:], 1)
draw_between(hidden, out, 2)
print("}")

