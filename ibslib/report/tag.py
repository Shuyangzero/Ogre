# -*- coding: utf-8 -*-


import os
from datetime import datetime
import matplotlib.image as mpimg
from matplotlib.figure import figaspect


file_dir = os.path.split(__file__)[0]
tag_img = mpimg.imread('{}/ibslib_10.png'.format(file_dir))
tag_aspect = figaspect(tag_img)


def plot_tag(ax):
    
    ax.imshow(tag_img, aspect="auto")


def plot_time(ax):
    today = str(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    ax.text(
        0.5,
        0.25,
        today,
        **{
            "horizontalalignment": "center",
            "verticalalignment": "center",
            "fontweight": "bold",
            "fontsize": 12,
            "color": "k",
        })
    