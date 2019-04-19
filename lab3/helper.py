from IPython.display import Image
import pydotplus
from sklearn import tree


def create_graph_png(classifier, file_path):
    dot_data = tree.export_graphviz(classifier)
    graph = pydotplus.graph_from_dot_data(dot_data)
    Image(graph.create_png())
    graph.write_png(file_path)
