import datetime
import os
from random import shuffle

import beatmapparser

# Created by Awlex

if __name__ == "__main__":

    # get Songs folder
    osu_songs_directory = os.path.join(os.getenv('LOCALAPPDATA'), 'osu!', 'Songs')

    # List Songs and shuffle the list
    maps = os.listdir(osu_songs_directory)
    shuffle(maps)

    # Pick random map
    map_path = os.path.join(osu_songs_directory, maps[0])

    # Pick first .osu file
    file = [x for x in os.listdir(map_path) if x.endswith(".osu")][0]
    osu_path = os.path.join(map_path, file)
    print(osu_path)

    # init parser
    parser = beatmapparser.BeatmapParser()

    # Parse File
    time = datetime.datetime.now()
    parser.parseFile(osu_path)
    print("Parsing done. Time: ", (datetime.datetime.now() - time).microseconds / 1000, 'ms')

    #Build Beatmap
    time = datetime.datetime.now()
    parser.build_beatmap()
    print("Building done. Time: ", (datetime.datetime.now() - time).microseconds / 1000, 'ms')

    quit()