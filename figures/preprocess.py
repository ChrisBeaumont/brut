"""
Show what preprocessing looks like
"""
from gallery import *

def collage(bubbles):
    ex = RGBExtractor()
    ex.shp = (40, 40)
    images = [ex.extract(*p) for p in bubble_params(bubbles)]

    if len(images) == 3:
        return np.vstack(images)

    r, g, b = tuple(montage2d(np.array([a[:, :, i] for a in images]))
                    for i in range(3))
    return np.dstack((r, g, b)).astype(np.uint8)


def main():
    bubbles = [13, 14, 17, 7]

    im = collage(bubbles)

    plt.figure(figsize=(3, 3), dpi=200)
    plt.gca().axis('off')
    plt.imshow(im)

    w = im.shape[1]

    h, w = im.shape[0:2]
    kwargs = dict(color='#222222', fontsize=20)

    hide_axes()
    plt.savefig('preprocess.eps')

if __name__ == "__main__":
    main()
