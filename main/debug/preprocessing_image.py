if __name__ == "__main__":
    import configparser
    import os
    import json
    from preprocessing.image import *
    from preprocessing.network import *
    from preprocessing.mesh import *
    from models.deeplabv3 import resnet50, ResNet50_Weights
    import numpy as np

    import pyproj
    import shapely
    from shapely.ops import transform

    CONFIG = "../../config/config_test.ini"
    config = configparser.ConfigParser()
    config.read(CONFIG, encoding="utf-8")

    read_general = config["GENERAL"]
    device = read_general["device"]

    read_geographic = config["GEOGRAPHIC"]
    bb_coords = np.array(json.loads(read_geographic["bb_coords"]))
    utm_num = int(read_geographic["utm_num"])

    read_data = config["DATA"]
    node_path = read_data["node_path"]
    link_path = read_data["link_path"]
    link_prop_path = read_data["link_prop_path"]
    image_data_path = read_data["image_data_path"]
    image_data_dir = read_data["satellite_image_datadir"]
    onehot_data_path = read_data["onehot_data_path"]
    onehot_data_dir = read_data["onehot_image_datadir"]

    mesh_image_dir = read_data["mesh_image_dir"]

    MESH = True
    SATELLITE = False
    ONEHOT = True
    SETFOLDER = True
    COMPRESS = False
    if not MESH:
        nw_data = NetworkCNN(node_path, link_path, link_prop_path=link_prop_path)
    else:
        bb = shapely.geometry.Polygon(bb_coords)
        wgs84 = pyproj.CRS("EPSG:4326")
        utm = pyproj.CRS(f"EPSG:{2442 + utm_num}")
        project = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform
        bb_utm = transform(project, bb)
        bb_coords_utm = bb_utm.bounds
        mnw_data = MeshNetwork(bb_coords_utm, 10, 10, 0)

    if SATELLITE:
        if SETFOLDER:
            if not MESH:
                image_data = SatelliteImageData(image_data_path, resolution=0.5,
                                                output_data_file=os.path.join(image_data_dir,
                                                                              "satellite_image_processed.json"))
                image_data.set_voronoi(nw_data)
                image_data.set_datafolder(image_data_dir)
            else:
                image_data = SatelliteImageData(image_data_path, resolution=0.5,
                                                output_data_file=os.path.join(mesh_image_dir,
                                                                              "satellite_image_processed.json"))
                image_data.split_by_mesh(mnw_data, mesh_image_dir)
        if COMPRESS:
            encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
            link_image_data = LinkImageData(os.path.join(image_data_dir, "satellite_image_processed.json"), nw_data)
            link_image_data.compress_images(encoder, device)
    if ONEHOT:
        if SETFOLDER:
            if not MESH:
                image_data = OneHotImageData(onehot_data_path, resolution=0.5,
                                             output_data_file=os.path.join(onehot_data_dir,
                                                                           "onehot_image_processed.json"))
                image_data.set_voronoi(nw_data)
                image_data.set_datafolder(onehot_data_dir)
                image_data.write_link_prop(onehot_data_dir, os.path.join(onehot_data_dir, "link_prop.csv"))
            else:
                image_data = OneHotImageData(onehot_data_path, resolution=0.5,
                                             output_data_file=os.path.join(mesh_image_dir,
                                                                           "onehot_image_processed.json"))
                image_data.split_by_mesh(mnw_data, mesh_image_dir, aggregate=True)
        if COMPRESS:
            encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
            link_image_data = LinkImageData(os.path.join(onehot_data_dir, "onehot_image_processed.json"), nw_data)
            link_image_data.compress_images(encoder, device)




