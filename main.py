from PIL import Image
import glob
import cv2, numpy as np
from sklearn.cluster import KMeans
import os
import math
from datetime import datetime
import subprocess
import random

print("Enter path with photos(only jpg)")
dir = input()+"*.jpg"
if dir == "*.jpg":
    dir = "/mnt/fun/desktop/Рисунки/сохран/"+"*.jpg"#you path


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
    files = glob.glob(dir)
    print(len(files))
    for file in files:
        if (os.path.basename(file).split("_")[0][0] != "#"):
            perc = get_max_percent(get_colors(file))
            tohex = rgb2hex(int(perc[0][0]),int(perc[0][1]),int(perc[0][2]))
            os.rename(file, file.replace(os.path.basename(file), tohex+"_"+str(perc[1])+".jpg"))

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

def get_arr_paths(arr):
    files = glob.glob(dir)
    newArr = []
    for i in arr:
        newArr.append(files[i])
    return newArr

def connect_rigth(img1, img2):
    # Select the image files to be merged
    img_file_1 = img1
    img_file_2 = img2

    # Open the image files
    try:
        im_1 = Image.open(img_file_1)
    except:
        im_1 = img1
    try:
        im_2 = Image.open(img_file_2)
    except:
        im_2 = img2

    # ИЗМЕНИТЬ ЦВЕТ С 250 на средний
    new_image = Image.new('RGB', ((im_2.size[0] + im_1.size[0])+100, min(im_1.size[1], im_2.size[1])), (0, 0, 0))

    # Paste the images onto the new image
    new_image.paste(im_1, (0, 0))
    new_image.paste(im_2, (im_1.size[0]+100, 0))

    return new_image

def connect_up(img1, img2):
    # Select the image files to be merged
    img_file_1 = img1
    img_file_2 = img2

    # Open the image files
    try:
        im_1 = Image.open(img_file_1)
    except:
        im_1 = img1
    try:
        im_2 = Image.open(img_file_2)
    except:
        im_2 = img2

    new_image = Image.new('RGB', (min(im_1.size[0], im_2.size[0]), (im_2.size[1] + im_1.size[1])+100), (0, 0, 0))


    new_image.paste(im_1, (0, 0))
    new_image.paste(im_2, (0, im_1.size[1]+100))

    return new_image

def create_img_matrix(arr, x, y):
  img_matrix = [ [None] * x for _ in range(y) ]
  for i in range(y):
    for j in range(x):
      img = Image.open(arr[i*x+j])
      img_matrix[i][j] = (arr[i*x+j], (img.size[0], img.size[1]))
  return img_matrix

def get_size(img):
    img_file_1 = img
    try:
        im_1 = Image.open(img_file_1)
    except:
        im_1 = img

    return im_1.size

reload_files()#init files in folder
arr = initialise(dir)# get arr of all imgs in dir
x = int(input("width:"))
y = int(input("height:"))
if(x == 0 or y == 0):
    x = 32
    y = 18

img_matrix = create_img_matrix(get_arr_paths(get_compared_images_with(arr[random.randint(0, 100)][0], arr, x*y)), x, y)
img_arr_lines = []
for i in range(0,y):
    img_arr_lines.append(img_matrix[i][0][0])



for i in range(y):
  for j in range(1,x):
    img_arr_lines[i] = connect_rigth(img_arr_lines[i], img_matrix[i][j][0])


out_img = img_arr_lines[0]
for i in range(1,y):
    out_img = connect_up(out_img, img_arr_lines[i])



try:
    os.remove("si1.jpg")
    img_file = "si2.jpg"
except:
    try:
        os.remove("si2.jpg")
    except:
        pass
    img_file = "si1.jpg"


out_img.save(img_file, "JPEG")

img_path = os.path.join("/mnt/Work/autoWallpaper", img_file)

command = f'qdbus org.kde.plasmashell /PlasmaShell org.kde.PlasmaShell.evaluateScript \'var allDesktops = desktops();for (i=0;i<allDesktops.length;i++) {{d = allDesktops[i];d.wallpaperPlugin = "org.kde.image";d.currentConfigGroup = Array("Wallpaper", "org.kde.image", "General");d.writeConfig("Image", "file://{img_path}")}}\''

subprocess.run(command, shell=True)
os.system(command)
