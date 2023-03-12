import os

path = "/home/jawabreh/Desktop/rav_sc2/"

def rename_images(path):
    i = 1001
    for filename in os.listdir(path):
        src = path + filename
        dst = path + str(i) + ".jpeg"
        os.rename(src, dst)
        i += 1

rename_images(path)
print("\n\n\tSUCCESS\n\n")