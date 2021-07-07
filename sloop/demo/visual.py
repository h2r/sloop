import cv2
from sloop.oopomdp.env.visual import MosViz
from sloop.demo.utils import cv2shape

class AirSimSearchViz(MosViz):

    def __init__(self, topo_map, *args, **kwargs):
        self._topo_map = topo_map
        super().__init__(*args, **kwargs)

    @classmethod
    def draw_topo(cls, img, topo_spec, res):
        radius = int(round(res / 2))
        for nid1, nid2 in topo_spec["edges"]:
            pos1 = topo_spec["nodes"][nid1]["x"], topo_spec["nodes"][nid1]["y"]
            pos2 = topo_spec["nodes"][nid2]["x"], topo_spec["nodes"][nid2]["y"]
            img = cls.draw_edge(img, pos1, pos2, res)

        for nid in topo_spec["nodes"]:
            pos = topo_spec["nodes"][nid]["x"], topo_spec["nodes"][nid]["y"]
            img = cv2shape(img, cv2.circle,
                           (pos[1]*res+radius,
                            pos[0]*res+radius), radius,
                           (200, 200, 36), thickness=-1, alpha=0.75)
        return img

    @classmethod
    def draw_edge(cls, img, pos1, pos2, r, thickness=2):
        x1, y1 = pos1
        x2, y2 = pos2
        cv2.line(img, (y1*r+r//2, x1*r+r//2), (y2*r+r//2, x2*r+r//2),
                 (0, 0, 0, 255), thickness=thickness)
        return img

    def _make_gridworld_image(self, r, bg_path=None):
        return AirSimSearchViz.make_gridworld_image(self._env.width,
                                                    self._env.length,
                                                    self._topo_map, r,
                                                    bg_path=bg_path,
                                                    state=self._env.state,
                                                    target_colors=self._target_colors)

    @classmethod
    def make_gridworld_image(cls, width, length, topo_map, res,
                             state=None, bg_path=None,
                             target_colors={}):
        img = MosViz.make_gridworld_image(width, length, res, state=state,
                                          bg_path=bg_path, target_colors=target_colors)
        img = cls.draw_topo(img, topo_map.to_json(), res)
        return img
