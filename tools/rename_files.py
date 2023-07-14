import os
os.getcwd()
video_type = "normal"
folder = "D:/Research/FMTResult_image/" + video_type+ "/"
for i, filename in enumerate(os.listdir(folder)):
    os.rename(folder + filename, folder + video_type + "-" + str(i) + ".png")