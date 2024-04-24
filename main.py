from PIL import Image
import glob
import cv2, numpy as np
from sklearn.cluster import KMeans
import os
import math

dir = "/run/media/Vld088/fun/desktop/Рисунки/сохран/"+"*.jpg"
# print(files)

def get_colors(file):
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    reshape = image.reshape((image.shape[0] * image.shape[1], 3))

    cluster = KMeans(n_clusters=5).fit(reshape)
    cluster, centroids = cluster, cluster.cluster_centers_
    out = []
    # Get the number of different clusters, create histogram, and normalize
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins = labels)
    hist = hist.astype("float")
    hist /= hist.sum()
    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])

    for (percent, color) in colors:
        out.append((color, "{:0.2f}%".format(percent * 100)))
    return out
# Load image and convert to a list of pixels
def get_max_percent(clusters):
    maxP=0
    for i in clusters:
        if float(i[1][0:-1]) > maxP:
            maxP = float(i[1][0:-1])
            color = i[0]
            if (maxP > 50):
                break
    color[0] = int(color[0])
    color[1] = int(color[1])
    color[2] = int(color[2])
    return (tuple(color),maxP)


def rgb2hex(r,g,b):
    return "#{:02x}{:02x}{:02x}".format(r,g,b)

def hex2rgb(hexcode):
    return tuple(map(ord,hexcode[1:].decode('hex')))

def reload_files():
    for file in files:
        perc = get_max_percent(get_colors(file))
        print(file)
        print(perc)
        print(rgb2hex(int(perc[0][0]),int(perc[0][1]),int(perc[0][2])))
        tohex = rgb2hex(int(perc[0][0]),int(perc[0][1]),int(perc[0][2]))
        if (os.path.basename(file).split("_")[0] != tohex):
            os.rename(file, file.replace(os.path.basename(file), tohex+"_"+str(perc[1])+".jpg"))
            print(file.replace(os.path.basename(file), tohex+"_"+str(perc[1])+".jpg"))

def initialise(dir):
    files = glob.glob(dir)
    arr = []
    for file in files:
        arr.append((os.path.basename(file).split("_")[0], os.path.basename(file).split("_")[1]))
    return arr


def compare_by_color(input_hex, arr) -> str:
    colors = arr
    color_distance_list = []

    input_color = tuple(int(input_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

    for i in range (len(colors)):
        use_color = colors[i][0]
        my_color = tuple(int(use_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        get_distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(my_color, input_color)]))
        color_distance_list.append(get_distance)

    sorted_color_distance_list = min(color_distance_list)
    closest_hex = color_distance_list.index(sorted_color_distance_list)

    return closest_hex

def get_compared_images_with(color, arr, count):
    outArr = []
    print(color)
    while len(outArr) < count:
        print(len(outArr))
        comp = compare_by_color(color,arr)
        if arr[comp][0] != color:
            #print(arr[comp][0])
            outArr.append(comp)
        arr.pop(comp)
    return outArr


arr = initialise(dir)
outArr = get_compared_images_with(arr[4][0], arr, 10)
print(outArr)
