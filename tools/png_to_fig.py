import glob
from PIL import Image

def make_gif(frame_folder):
    images = glob.glob(f"{frame_folder}/*.png")
    images.sort()
    frames = [Image.open(image) for image in images]
    frame_one = frames[0]
    frame_one.save("1-22_2nd_chase_16217-16244.gif", format="GIF", append_images=frames,
                   save_all=True, duration=66, loop=0) #  duration: The time to display the current frame of the GIF, in milliseconds.
    
if __name__ == "__main__":
    # frame_folder = "D:/Research/FMTResult_image/1-14_display_23465-23515/"
    # frame_folder = "D:/Research/FMTResult_image/1-14_bite_12923-12944/"
    # frame_folder = "D:/Research/FMTResult_image/1-22_2nd_normal_11894-11942/"
    frame_folder = "D:/Research/FMTResult_image/1-22_2nd_chase_16217-16244/"

    make_gif(frame_folder)