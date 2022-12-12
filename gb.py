import numpy as np
import dt
import plotly.figure_factory as ff
import plotly.express as px


class GB():

    def __init__(self, M, alfa):
        self.M = M
        self.y = []
        self.node = []
        self.al = []
        self.alfa = alfa

    def decision_stump(self, dn, cl, ver):
        d = dt.DT(1, 0.05, 5, 2)
        root = dt.Node()
        d.build_tree(dn, cl, root, 0,ver)
        return d, root

    def sigm(self, x):
        return 1 / (1 + np.exp(-x))

    def gradBoost(self, dn, cl):
        self.Y = [sum(cl) / len(cl)]
        self.y = np.array([sum(cl) / len(cl) for i in range(len(cl))])
        for i in range(self.M):
            pob = self.sigm(self.y)
            r = cl - pob
            d, root = self.decision_stump(dn, r, pob)
            self.y += self.alfa * np.array(d.pass_tree_all(root, dn))
            self.Y.append(d)
            self.node.append(root)

    def pass_ad(self, dn):
        s = self.Y[0] + self.alfa * np.sum(
            np.array([self.Y[i].pass_tree(self.node[i - 1], dn) for i in range(1, len(self.Y))]))
        if s >= 0.5:
            return 1
        else:
            return 0

    def pass_ad_all(self, dn):
        s = []
        for i in range(len(dn)):
            s.append(self.pass_ad(dn[i]))
        return np.array(s)

    def MSE(self, dn, cl):
        s = np.sum((cl - self.pass_ad_all(dn))**2) / len(dn)
        return s

    def confusion_matrix(self, dn, cl):
        TP, TN, FP, FN = 0, 0, 0, 0
        for i in range(len(dn)):
            if self.pass_ad(dn[i]) == cl[i]:
                if cl[i] == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if cl[i] == 1:
                    FN += 1
                else:
                    FP += 1
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * precision * recall / (precision + recall)
        z = [[TP, FP], [FN, TN]]
        fig = px.imshow(z, text_auto=True, title="precision = " + str(precision) + "; recall = " + str(recall) +
                                                 "; f1_score = " + str(f1_score))
        fig.show()
        fig.write_html("asdad.html")