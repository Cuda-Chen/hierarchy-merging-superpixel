from skimage import data, io, segmentation, color
from skimage.future import graph
import numpy as np
import argparse
import os

def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.node[dst]['mean color'] - graph.node[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}


def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.node[dst]['total color'] += graph.node[src]['total color']
    graph.node[dst]['pixel count'] += graph.node[src]['pixel count']
    graph.node[dst]['mean color'] = (graph.node[dst]['total color'] /
                                     graph.node[dst]['pixel count'])

parser = argparse.ArgumentParser()
parser.add_argument('input_image', type=str, help='input image path')
parser.add_argument('num_superpixel', type=int, help='number of segments')
parser.add_argument('compactness', type=int, help='compactness param of SLIC')
parser.add_argument('thresh', type=float, help='threshold of combining edge')
args = parser.parse_args()

img = io.imread(args.input_image)
outputfile = os.path.splitext(args.input_image)[0] \
+ '_' + str(args.num_superpixel) \
+ '_' + str(args.compactness) \
+ '_' + str(args.thresh) + '.bmp'

labels = segmentation.slic(img, n_segments=args.num_superpixel, compactness=args.compactness)
g = graph.rag_mean_color(img, labels)

labels2 = graph.merge_hierarchical(labels, g, thresh=args.thresh, rag_copy=False,
                                   in_place_merge=True,
                                   merge_func=merge_mean_color,
                                   weight_func=_weight_mean_color)

out = color.label2rgb(labels2, img, kind='avg')
#out = segmentation.mark_boundaries(out, labels2, (0, 0, 0))
io.imshow(out)
io.show()
