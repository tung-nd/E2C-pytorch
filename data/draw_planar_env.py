import numpy as np
from PIL import Image, ImageDraw

img_arr = np.zeros(shape=(40,40))
obstacles = np.array([[19,5], [19,11], [19,27], [19,34], [10,19], [29,19]])
r = 2

img = Image.fromarray(img_arr)
draw = ImageDraw.Draw(img)
for y, x in obstacles:
    draw.ellipse((x-r, y-r, x+r, y+r), fill=255)
img = img.convert('L')
img.save('env.png')

img_arr = np.array(img) / 255.
# print (len(np.where(img_arr == 1.)[0]))
np.save('./env.npy', img_arr)