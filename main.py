from skimage import data, io, segmentation, color, measure
from skimage.future import graph
from skimage.color import rgb2gray
import numpy as np
import argparse
import os
import cv2 as cv

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

def drawShape(img, coordinates, color):
    coordinates = coordinates.astype(int)
    img[coordinates[:, 0], coordinates[:, 1]] = color
    return img

parser = argparse.ArgumentParser()
parser.add_argument('input_image', type=str, help='input image path')
parser.add_argument('num_superpixel', type=int, help='number of segments')
parser.add_argument('compactness', type=int, help='compactness param of SLIC')
parser.add_argument('thresh', type=float, help='threshold of combining edge')
args = parser.parse_args()

img = io.imread(args.input_image)
#img = data.coffee()
outputfile = os.path.splitext(args.input_image)[0] \
+ '_' + str(args.num_superpixel) \
+ '_' + str(args.compactness) \
+ '_' + str(args.thresh) + '.bmp'

labels = segmentation.slic(img, n_segments=args.num_superpixel, compactness=args.compactness, sigma=2)
g = graph.rag_mean_color(img, labels)

labels2 = graph.merge_hierarchical(labels, g, thresh=args.thresh, rag_copy=False,
                                   in_place_merge=True,
                                   merge_func=merge_mean_color,
                                   weight_func=_weight_mean_color)

out = color.label2rgb(labels2, img, kind='avg')
#out = segmentation.mark_boundaries(out, labels2, (0, 0, 0))

out_gray = cv.cvtColor(out, cv.COLOR_RGB2GRAY)
io.imshow(out_gray)
io.show()
#out_gray = out_gray.astype(np.uint8) * 255
out_gray = cv.GaussianBlur(out_gray, (5, 5), 0)
io.imshow(out_gray)
io.show()
canny = cv.Canny(out_gray, 20, 160)
io.imshow(canny)
io.show()
contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
if len(contours) != 0:
    cv.drawContours(out, contours, -1, (255, 0, 0), 2)
    c = max(contours, key = cv.contourArea)
    x,y,w,h = cv.boundingRect(c)
    cv.rectangle(out, (x,y), (x+w,y+h),(0,255,0),3)

io.imshow(out)
io.show()
#io.imsave(outputfile, out)
