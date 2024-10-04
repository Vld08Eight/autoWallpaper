
# Image Mosaic Generator and Wallpaper Setter

## Overview
This project is designed to generate a mosaic image from a collection of input images and set it as the desktop wallpaper. It uses KMeans clustering to determine the dominant colors in each image, organizes them based on color similarity, and then merges these images into a larger mosaic.

## Dependencies
- `PIL` (Python Imaging Library)
- `OpenCV`
- `NumPy`
- `Scikit-learn`
- `glob`
- `os`
- `math`
- `datetime`
- `subprocess`

## Usage

### Input Path
The script prompts for a directory path containing JPEG files. If no path is provided, it defaults to `/mnt/fun/desktop/Рисунки/сохран/`.

### Running the Script
1. **Enter Directory Path**: When prompted, enter the directory path in the format `/path/*.jpg`.
2. **Specify Dimensions**: Enter the desired width and height of the output mosaic image.
3. **Generate Mosaic**: The script will reload files, initialize colors, compare colors, and generate the mosaic image.
4. **Set Wallpaper**: The generated mosaic image will be set as the desktop wallpaper using KDE PlasmaShell commands.

### Example Command Flow
```bash
$ python script_name.py
Enter path in format /path/*.jpg 
/mnt/fun/desktop/Рисунки/сохран/*.jpg
width: 32
height: 18
```

## Functions

### get_colors(file)
Extracts dominant colors from an image using KMeans clustering.

### get_max_percent(clusters)
Finds the color with the maximum percentage in an image.

### rgb2hex(r,g,b) and hex2rgb(hexcode)
Converts RGB values to hexadecimal and vice versa.

### reload_files()
Renames files based on their dominant color and percentage.

### initialise(dir)
Initializes an array of file names and their corresponding colors.

### compare_by_color(input_hex, arr)
Compares input color with colors in an array to find closest matches.

### get_compared_images_with(color, arr, count)
Returns indices of images with closest matching colors.

### create_img_matrix(arr, x, y)
Creates a matrix of images for merging into a mosaic.

### connect_rigth(img1, img2) and connect_up(img1, img2)
Merges two images horizontally or vertically.

### generateImg(dir, x, y)
Generates a mosaic image from input images and sets it as wallpaper.

## Code Structure

The script is structured into several functions that handle different aspects such as:
- Color extraction
- File renaming
- Image comparison
- Merging images
- Setting wallpaper

## Notes

- Ensure that you have all dependencies installed before running this script.
- This script assumes you are using KDE Plasma for setting wallpapers; adjust commands accordingly if using another desktop environment.
- The default directory can be changed within `initialise` function if needed.

## Contributing
Contributions are welcome Feel free to fork this repository and submit pull requests for improvements or new features.

## License
This project is licensed under [MIT License](https://opensource.org/licenses/MIT).
