import imageio
import os

def make_gif(figures_path):
    """A simple function to collate figures into gif."""
    if not os.path.exists(figures_path + '/gif'):
        os.mkdir(figures_path + '/gif')

    for folder in os.listdir(figures_path):
        if folder == 'gif':
            continue
        images = []
        figures = os.listdir(figures_path + "/" + folder)
        figures.sort(key=lambda x: int(x.split('-')[0]))
        for file in figures:
            images.append(imageio.imread(figures_path + "/" + folder + "/" + file))
        imageio.mimsave(figures_path + "/gif/" + folder + ".gif", images, duration=0.2)
