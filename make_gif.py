from PIL import Image
from glob import glob

def make_gif(path, output_gif_name='output.gif', duration=500):
    # Get all png files in the current directory
    png_files = sorted(glob(f'{path}/*.png'))

    # Create an image list
    images = []

    # Open each image and append to the list
    for png in png_files:
        img = Image.open(png)
        images.append(img)

    # Save the images as a gif
    if images:
        images[0].save(output_gif_name, save_all=True, append_images=images[1:], duration=duration, loop=0)
        print(f'GIF created successfully: {output_gif_name}')
    else:
        print('No PNG files found in the directory.')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Create a gif from a sequence of PNG images.')
    parser.add_argument('--output', type=str, default='output.gif', help='Output gif name')
    parser.add_argument('--duration', type=int, default=200, help='Duration between frames in milliseconds')
    parser.add_argument('--path', type=str, default='tmp', help='Path to the directory containing PNG images')
    args = parser.parse_args()
    make_gif(args.path, args.output, args.duration)
