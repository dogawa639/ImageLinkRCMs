if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    import json
    import configparser
    from preprocessing.network import *
    from preprocessing.geo_util import *
    from utility import *

    print(os.getcwd())

    CONFIG = "../../config/config_test.ini"
    config = configparser.ConfigParser()
    config.read(CONFIG, encoding="utf-8")

    read_geo = config["GEOGRAPHIC"]
    mask_path = read_geo["mask_path"]

    bb_coords = get_vertex_from_polygon(mask_path)
    print(bb_coords)

    bb_coords = json.loads(read_geo["bb_coords"])
    print(bb_coords)

    read_data = config["DATA"]

    node_path = read_data["node_path"]
    link_path = read_data["link_path"]
    link_prop_path = read_data["link_prop_path"]

    NetworkBase.get_from_osm(bb_coords, 4,
                             node_path, link_path, link_prop_path)

    nw_data = NetworkBase(node_path, link_path, mode="car")
    ax = nw_data.plot_link(nw_data.lids)
    plt.show()