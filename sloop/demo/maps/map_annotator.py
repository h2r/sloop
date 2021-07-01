import cv2
import time
import os
import pygame
import copy
import json
import shutil
import numpy as np
from pprint import pprint
from spatial_foref.oopomdp.env.visual import MosViz
from spatial_foref.utils import random_unique_color, hex_to_rgb

MAP_IMG = "./neighborhoods_topdown.png"


#### Interactive tool to create landmarks ####
class InteractiveMapper:

    def __init__(self, map_img_path=MAP_IMG, res=25, map_dim=(41,41)):
        self.map_img_path = map_img_path
        self.res = res
        self.map_dim = map_dim

        self.on_init()

    @property
    def img_width(self):
        return self.map_dim[0] * self.res

    @property
    def img_height(self):
        return self.map_dim[0] * self.res

    def _make_gridworld_image(self):
        w, l = self.map_dim
        img = MosViz.make_gridworld_image(w, l, self.res, bg_path=self.map_img_path)
        img = cv2.flip(img, 1)  # flip horizontally
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # rotate 90deg clockwise
        return img

    def on_init(self):
        """pygame init"""
        pygame.init()  # calls pygame.font.init()
        # init main screen and background
        self._display_surf = pygame.display.set_mode((self.img_width,
                                                      self.img_height),
                                                     pygame.HWSURFACE)
        self._background = pygame.Surface(self._display_surf.get_size()).convert()
        self._clock = pygame.time.Clock()

        # Font
        self._myfont = pygame.font.SysFont('Comic Sans MS', 30)
        self._running = True

    def show_img(self, img):
        """
        Internally, the img origin (0,0) is top-left (that is the opencv image),
        so +x is right, +z is down.
        But when displaying, to match the THOR unity's orientation, the image
        is flipped, so that in the displayed image, +x is right, +z is up.
        """
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        img = cv2.flip(img, 1)  # flip horizontally
        pygame.surfarray.blit_array(self._display_surf, img)
        pygame.display.flip()

    @staticmethod
    def get_clicked_pos(r, w, l):
        pos = None
        pygame.event.clear()
        while pos is None:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = event.pos
                    break
            time.sleep(0.001)
        # y coordinate is flipped
        return pos[0] // r, l - pos[1] // r - 1

    def mark_cell(self, img, pos, r, linewidth=1, text="", color=(242, 227, 15)):
        x, y = pos
        font                   = cv2.FONT_HERSHEY_COMPLEX_SMALL
        fontScale              = 0.72
        fontColor              = (43, 13, 4)
        lineType               = 1
        imgtxt = np.full((r, r, 3), color, dtype=np.uint8)
        text_loc = (int(round(r/4)), int(round(r/1.5)))
        cv2.putText(imgtxt, text, text_loc, #(y*r+r//4, x*r+r//2),
                    font, fontScale, fontColor, lineType)
        imgtxt = cv2.rotate(imgtxt, cv2.ROTATE_90_CLOCKWISE)
        img[x*r:x*r+r, y*r:y*r+r] = imgtxt
        return img


    def start(self, res=25):
        """Starts the interactive mapper"""
        def print_options(options):
            for num, txt in sorted(options.items()):
                print(num, ":", txt)

        w, l = self.map_dim

        img = self._make_gridworld_image()
        done_pos = (0, l-1)
        img = self.mark_cell(img, done_pos, res, linewidth=1, color=(230, 2, 55), text="X")
        self.show_img(img)

        OPTIONS = {
            0: "add_landmark",
            1: "save"
        }

        _landmarks = {}
        _current_landmark_cells = set()
        _colors = [(242, 227, 15)]

        last_opt = None
        while True:
            print_options(OPTIONS)


            if last_opt != 0:  # if not add landmark
                opt = input("Action [{}-{}]: ".format(min(OPTIONS), max(OPTIONS)))
                if len(opt) == 0:
                    opt = last_opt
                    if last_opt is None:
                        opt = 0  # add_landmark, by default

                try:
                    opt = int(opt)
                    action = OPTIONS[opt]
                    last_opt = opt
                except Exception:
                    print("Invalid option {}".format(opt))
                    continue

            if action == "add_landmark":
                print("Click on window to select cell")
                x, y = self.get_clicked_pos(self.res, w, l)

                if (x, y) != done_pos:
                    # mark clicked grid cell as part of current landmark
                    _current_landmark_cells.add((x,y))
                    img = self.mark_cell(img, (x,y), res, linewidth=1, color=_colors[-1])
                    self.show_img(img)
                else:
                    # Done adding landmark cells. Ask for symbol and name
                    last_opt = None
                    while True:
                        landmark_spec = input("Landmark Symbol, Name (e.g. House1, House 1): ")
                        if len(landmark_spec.split(",")) != 2:
                            print("Invalid input. Format: \"LandmarkSymbol, Landmark Name\"")
                        else:
                            break
                    landmark_symbol, landmark_name = landmark_spec.split(",")
                    _landmarks[landmark_symbol.strip()] = {
                        "name": landmark_name.strip(),
                        "footprint": list(sorted(copy.deepcopy(_current_landmark_cells)))
                    }
                    _current_landmark_cells = set()
                    _colors.append(hex_to_rgb(random_unique_color(_colors)))

            elif action == "save":
                map_name = input("Map name (e.g. austin): ")
                os.makedirs(map_name, exist_ok=True)

                # save a bunch of things ... only those we could save; Others, we will do empty
                print("Saving JSON files")
                with open(os.path.join(map_name, "cardinal_to_limit.json"), "w") as f:
                    f.write("{}")
                with open(os.path.join(map_name, "excluded_symbols.json"), "w") as f:
                    f.write("[]")
                with open(os.path.join(map_name, "idx_to_cell.json"), "w") as f:
                    f.write("{}")
                with open(os.path.join(map_name, "name_to_feats.json"), "w") as f:
                    f.write("{}")
                with open(os.path.join(map_name, "name_to_symbols.json"), "w") as f:
                    name_to_symbols = {}
                    for landmark_symbol in _landmarks:
                        name = _landmarks[landmark_symbol]["name"]
                        name_to_symbols[name] = landmark_symbol
                    json.dump(name_to_symbols, f)
                with open(os.path.join(map_name, "pomdp_to_idx.json"), "w") as f:
                    pomdp_to_idx = {}
                    idx = 0
                    for x in range(w):
                        for y in range(l):
                            pomdp_coord = (y,x)
                            pomdp_to_idx[str(pomdp_coord)] = idx
                            idx += 1
                    json.dump(pomdp_to_idx, f)
                with open(os.path.join(map_name, "name_to_idx.json"), "w") as f:
                    name_to_idx = {}
                    for landmark_symbol in _landmarks:
                        name = _landmarks[landmark_symbol]["name"]
                        footprint = _landmarks[landmark_symbol]["footprint"]
                        name_to_idx[name] = [pomdp_to_idx[str(loc)]
                                             for loc in footprint]
                    json.dump(name_to_idx, f)
                with open(os.path.join(map_name, "streets.json"), "w") as f:
                    f.write("[]")
                with open(os.path.join(map_name, "symbol_to_name.json"), "w") as f:
                    symbol_to_name = {landmark_symbol: _landmarks[landmark_symbol]["name"]
                                      for landmark_symbol in _landmarks}
                    json.dump(symbol_to_name, f)
                with open(os.path.join(map_name, "symbol_to_synonym.json"), "w") as f:
                    symbol_to_synonym = {landmark_symbol: [_landmarks[landmark_symbol]["name"]]
                                      for landmark_symbol in _landmarks}
                    json.dump(symbol_to_synonym, f)

                print("Copying map image")
                shutil.copyfile(self.map_img_path, os.path.join(map_name, "{}_diam_100m.PNG".format(map_name)))

if __name__ == "__main__":
    mapper = InteractiveMapper()
    mapper.start()
