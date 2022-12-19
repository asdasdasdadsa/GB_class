import data
import gb

d = data.WineDataset("C:/Users/lul-0/PycharmProjects/GB_class/wine-quality-white-and-red 2 (2).csv")
dannn = d()
x_train, t_train, x_test, t_test = dannn['train_input'], dannn['train_target'], dannn['test_input'], dannn['test_target']
aadd = gb.GB(20, 0.05)
aadd.gradBoost(x_train, t_train)
aadd.confusion_matrix(x_test, t_test)