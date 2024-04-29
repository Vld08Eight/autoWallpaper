from PIL import Image
import glob
import cv2, numpy as np
from sklearn.cluster import KMeans
import os
import math
from datetime import datetime
import subprocess
import random

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
    files = glob.glob(dir)
    print(len(files))
    for file in files:
        if (os.path.basename(file).split("_")[0][0] != "#"):
            perc = get_max_percent(get_colors(file))
        #print(file)
        #print(perc)
        #print(rgb2hex(int(perc[0][0]),int(perc[0][1]),int(perc[0][2])))
            tohex = rgb2hex(int(perc[0][0]),int(perc[0][1]),int(perc[0][2]))
            os.rename(file, file.replace(os.path.basename(file), tohex+"_"+str(perc[1])+".jpg"))
            #print(file.replace(os.path.basename(file), tohex+"_"+str(perc[1])+".jpg"))

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
    # Resize the images to the same size
    #im_1 = im_1.resize((400, 400))
    #im_2 = im_2.resize((400, 400))

    # ИЗМЕНИТЬ ЦВЕТ С 250 на средний
    new_image = Image.new('RGB', ((im_2.size[0] + im_1.size[0])+100, min(im_1.size[1], im_2.size[1])), (0, 0, 0))

    # Paste the images onto the new image
    new_image.paste(im_1, (0, 0))
    new_image.paste(im_2, (im_1.size[0]+100, 0))

    # Save the merged image in the desired format
    #new_image.save("some_image3.jpg", "JPEG")
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
    # Resize the images to the same size
    #im_1 = im_1.resize((400, 400))
    #im_2 = im_2.resize((400, 400))

    # ИЗМЕНИТЬ ЦВЕТ С 250 на средний
    new_image = Image.new('RGB', (min(im_1.size[0], im_2.size[0]), (im_2.size[1] + im_1.size[1])+100), (0, 0, 0))

    # Paste the images onto the new image
    new_image.paste(im_1, (0, 0))
    new_image.paste(im_2, (0, im_1.size[1]+100))

    # Save the merged image in the desired format
    #new_image.save("some_image3.jpg", "JPEG")
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

    # Open the image files
    try:
        im_1 = Image.open(img_file_1)
    except:
        im_1 = img

    return im_1.size

reload_files()#init files in folder
arr = initialise(dir)# get arr of all imgs in dir
x = 16
y = 9

img_matrix = create_img_matrix(get_arr_paths(get_compared_images_with(arr[random.randint(0, 100)][0], arr, x*y)), x, y)
img_arr_lines = []
for i in range(0,y):
    img_arr_lines.append(img_matrix[i][0][0])



for i in range(y):
  for j in range(1,x):
    #if (get_size(img_arr_lines[i])[0] < 1920):
    img_arr_lines[i] = connect_rigth(img_arr_lines[i], img_matrix[i][j][0])


out_img = img_arr_lines[0]
for i in range(1,y):
    #if (get_size(out_img)[1] < 1080):
    out_img = connect_up(out_img, img_arr_lines[i])



i = 2

name = "si"+str(i)+".jpg"
out_img.save("si"+str(i)+".jpg", "JPEG")

#os.system("xwallpaper --center some_image.jpg")
#os.system("feh")
command1 = 'qdbus org.kde.plasmashell /PlasmaShell org.kde.PlasmaShell.evaluateScript \'var allDesktops = desktops();for (i=0;i<allDesktops.length;i++) {d = allDesktops[i];d.wallpaperPlugin = "org.kde.image";d.currentConfigGroup = Array("Wallpaper", "org.kde.image", "General");d.writeConfig("Image", "file:///mnt/Work/autoWallpaper/si1.jpg")}\''
subprocess.run(command1, shell=True)
os.system(command1)
command2 = 'qdbus org.kde.plasmashell /PlasmaShell org.kde.PlasmaShell.evaluateScript \'var allDesktops = desktops();for (i=0;i<allDesktops.length;i++) {d = allDesktops[i];d.wallpaperPlugin = "org.kde.image";d.currentConfigGroup = Array("Wallpaper", "org.kde.image", "General");d.writeConfig("Image", "file:///mnt/Work/autoWallpaper/si2.jpg")}\''
subprocess.run(command2, shell=True)
os.system(command2)