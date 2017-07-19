# coding: utf-8
# Author: Laurent Jégou <jegou@univ-tlse2.fr>
# With code from:
#          Robert Layton <robertlayton@gmail.com>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Mathieu Blondel <mathieu@mblondel.org>
#
# License: BSD 3 clause

import os
from flask import Flask,render_template,request
import math
import numpy as np
import operator
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from time import time
from skimage import color

app = Flask(__name__)
progress = 0

# Default root route (pun intended)
@app.route('/')
def hello():
    return '<a href="/tool">Color relations and proportions of an image</a>'

# Route to tool HTML template
@app.route('/tool')
def toolpage():
    return render_template('tool.html')

# Route to progress request (threaded)
@app.route('/progress', methods=['GET'])
def ajax_index():
    global progress
    return str(progress)

# Main analysis route
@app.route('/analysis', methods=['POST'])
def analyse_couleurs():
    import StringIO
    import urllib
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    global progress
    progress = 10
    n_colors = int(request.form['nbbins'])
    nbsamples = int(request.form['nbsamples'])
    opacity = float(request.form['opacity'])
    satlum = request.form['satlum']
    ct = float(request.form['sizecoef'])
    srcimage = request.form['srcimage']
    if (srcimage[:4] == "http"):
        srcimage = StringIO.StringIO(urllib.urlopen(srcimage).read())

    # Opens image to an array of RGB colors
    img = Image.open(srcimage)

    # Convert to floats instead of the default 8 bits integer coding. Dividing by
    # 255 is important so that plt.imshow behaves works well on float data (need to
    # be in the range [0-1])
    img = np.array(img, dtype=np.uint8)

    # Convert to HSV color space
    imgHSV = color.rgb2hsv(img)

    # Load Image and transform to a 2D numpy array.
    w, h, d = original_shape = tuple(imgHSV.shape)
    assert d == 3
    image_array = np.reshape(imgHSV, (w * h, d))

    # print("Fitting model on a small sub-sample of the data")
    t0 = time()
    image_array_sample = shuffle(image_array, random_state=0, n_samples=nbsamples)[:1000]
    kmeans = KMeans(n_clusters=n_colors, random_state=0, n_jobs=1, max_iter=200, init='k-means++', algorithm='elkan',
                    precompute_distances=True).fit(image_array_sample)
    #print("done in %0.3fs." % (time() - t0))
    progress = 50

    # Get labels for all points
    #print("Predicting color indices on the full image (k-means)")
    t0 = time()
    labels = kmeans.predict(image_array)
    couleurs = kmeans.cluster_centers_
    print("done in %0.3fs." % (time() - t0))
    # Counting pixels in color bins, for sorting and circles dimensioning
    unique, counts = np.unique(labels, return_counts=True)
    progress = 60

    ro = math.pi / 2
    maxc = max(counts)
    rcmc = math.sqrt(maxc)
    fig = plt.figure()
    ax = plt.subplot(111, projection='polar')
    ax.set_theta_zero_location("N")
    ax.set_rmax(1)
    ax.set_rticks([0.25, 0.5, 0.75, 1])  # less radial ticks
    ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
    ax.grid(True)

    cp = dict(zip(range(0, n_colors), counts))
    co = sorted(cp.items(), key=operator.itemgetter(1), reverse=True)

    # Figure-drawing loop
    for i in range(0, n_colors):
        ht = -((math.pi * 2) * couleurs[co[i][0]][0]) + ro # Color hue coefficient
        x = couleurs[co[i][0]][1] * math.cos(ht) # Conversion of ht to X pos
        y = couleurs[co[i][0]][1] * math.sin(ht) # Conversion of ht to Y pos
        va = counts[co[i][0]] # Raw size of color bin
        ra = ((math.sqrt(va) / rcmc) / 2) * ct # Adapted size for display, relative to the max size
        at = (np.array([[couleurs[co[i][0]]]]) * 255).astype(np.uint8)
        c = color.hsv2rgb(at) # Retro-conversion to RGB for plotting
        tc = plt.Circle((x, y), ra, transform=ax.transProjectionAffine + ax.transAxes, color=c[0][0].astype(np.float),
                        alpha=opacity, clip_on=False) # Circle drawing
        ax.add_artist(tc)

    # Figure to PNG export
    progress = 70
    canvas = FigureCanvas(fig)
    png_output = StringIO.StringIO()
    canvas.print_png(png_output)
    progress = 80
    data = png_output.getvalue().encode('base64')
    data_url = 'data:image/png;base64,{}'.format(urllib.quote(data.rstrip('\n')))
    progress = 100
    return data_url

if __name__=="__main__":
    app.run(threaded=True)