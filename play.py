import pygame
from pygame.locals import *
import numpy as np
import matplotlib.pyplot as plt
import math
from osuparser.beatmapparser import *
from osuparser.curve import *
from osuparser.slidercalc import *
from posu.emulator import *
from posu.env_utils import *


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

def main():
    parser = BeatmapParser()
    parser.parseFile('leaf.osu')
    beatmap_d = parser.build_beatmap()
    
    hitObjects = FormatParsedBeatmap(beatmap_d)
    
    pygame.quit()
    pygame.init()

    size = width, height = 192 + 80, 144 + 80
    clock = pygame.time.Clock()

    screen = pygame.display.set_mode(size, DOUBLEBUF, 16)
    screen.fill(BLACK)

    ms = pygame.Surface((10, 10))
    draw_cursor(ms, (5, 5), 5)
    cursor = pygame.cursors.Cursor((5, 5), ms)
    pygame.mouse.set_cursor(cursor)

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                run = False

        screen.fill(BLACK)

        objs = []
        for obj in hitObjects:
            if obj.appear < pygame.time.get_ticks() < obj.disappear:
                objs.append(obj)

        frame = Frame(objs)
        frame.draw(screen, pygame.time.get_ticks())

        pygame.display.flip()
        pygame.display.update()

        if pygame.time.get_ticks() > hitObjects[-1].disappear + 2000:
            run = False
        clock.tick(60)
        
if __name__ == '__main__':
    main()