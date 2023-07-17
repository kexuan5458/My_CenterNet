from PIL import Image 
import os
# import argparse 
# parser = argparse.ArgumentParser()
# parser.add_argument("--path", help="Path of the folder",
#                     default='/data/RADIATE/city_1_1/Navtech_Cartesian_png',
#                     type=str)
# args = parser.parse_args()
# path = args.path

def convert_png_to_jpg(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over the PNG files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            # Open the PNG image
            png_path = os.path.join(input_folder, filename)
            img = Image.open(png_path)

            # Convert the image to JPEG format
            jpg_filename = os.path.splitext(filename)[0] + ".jpg"
            jpg_path = os.path.join(output_folder, jpg_filename)
            img.save(jpg_path, "JPEG")

            # Close the image
            img.close()

# Provide the paths of the input and output folders
input_folder = "/data/RADIATE/city_3_7/Navtech_Cartesian_png"
output_folder = "/data/RADIATE/city_3_7/Navtech_Cartesian"

# Convert PNG files to JPEG
convert_png_to_jpg(input_folder, output_folder)
