from PIL import Image, ImageDraw
import numpy as np
width, height = 40, 40
r = 2.5
obstacles_center = np.array([[20.5, 5.5], [20.5, 12.5], [20.5, 27.5], [20.5, 35.5], [10.5, 20.5], [30.5, 20.5]])

def generate_env():
    print ('Making the environment...')
    img_arr = np.zeros(shape=(width,height))

    img_env = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img_env)
    for y, x in obstacles_center:
        draw.ellipse((int(x)-int(r), int(y)-int(r), int(x)+int(r), int(y)+int(r)), fill=255)
    img_env = img_env.convert('L')
    img_env.save('env.png')

    img_arr = np.array(img_env) / 255.
    np.save('./env.npy', img_arr)
    return img_arr

env = generate_env()