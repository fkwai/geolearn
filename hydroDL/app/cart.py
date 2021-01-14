import numpy as np


def TraverseTree(regTree, Xin, Fields=None):
    nnode = regTree.node_count
    if Fields is not None:
        featurename = [Fields[i] for i in regTree.feature]
    else:
        featurename = ["Fields%i" % i for i in regTree.feature]
    string = ['']*nnode
    nodeind = [None]*nnode
    nind = Xin.shape[0]
    leaf = []
    label = np.zeros([nind])

    def recurse(tempstr, node, Xtemp, indtemp):
        string[node] = "node#%i: " % node+tempstr
        nodeind[node] = indtemp
        if (regTree.threshold[node] != -2):
            if regTree.children_left[node] != -1:
                tempstr = tempstr + \
                    " ( " + featurename[node] + " <= " + \
                    "%.3f" % (regTree.threshold[node]) + " ) ->\n "
                indlocal = np.where(
                    Xtemp[:, regTree.feature[node]] <= regTree.threshold[node])[0]
                indleft = [indtemp[i] for i in indlocal]
                Xleft = Xtemp[indlocal, :]
                recurse(tempstr, regTree.children_left[node], Xleft, indleft)
            if regTree.children_right[node] != -1:
                tempstr = " ( " + featurename[node] + " > " + \
                    "%.3f" % (regTree.threshold[node]) + " ) ->\n "
                indlocal = np.where(
                    Xtemp[:, regTree.feature[node]] > regTree.threshold[node])[0]
                indright = [indtemp[i] for i in indlocal]
                Xright = Xtemp[indlocal, :]
                recurse(
                    tempstr, regTree.children_right[node], Xright, indright)
        else:
            leaf.append(node)
            label[indtemp] = node
    recurse('', 0, Xin, range(0, nind))
    return string, nodeind, leaf, label
