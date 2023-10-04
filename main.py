import configparser
import os

from preprocessing import *
from models import *
from learning import *

import pandas as pd
import numpy as np

CONFIG = "./config/config.ini"

if __name__ == "__main__":
    print(os.getcwd())
    config_ini = configparser.ConfigParser()
    config_ini.read(CONFIG, encoding="utf-8")  # general, preprocessing, models, learning, trained_models

    # # general setting
    read_general = config_ini["GENERAL"]
    device = read_general["device"]  # cpu, mps
    cross_valid = read_general["cross_valid"]  # 0->no validation, 2-5->k-hold cross validation
    train_model = read_general["train_model"]  # bool
    output_dir = read_general["output_dir"]  # path

    # # preprocessing
    read_preproc = config_ini["PREPROCESSING"]
    data_path = read_preproc["data_path"]

    # # models
    read_models = config_ini["MODELS"]
    gen_name = read_models["generator"]
    disc_name = read_models["discriminator"]

    # # learning
    read_learning = config_ini["LEARNING"]
    lr = read_learning["lr"]  # 0.01
    batch_size = read_learning["batch_size"]
    loss_type = read_learning["loss_type"]  # gan, airl
    max_itr = read_learning["max_itr"]
    early_stop_itr = read_learning["early_stop_itr"]  # the iteration num from which early stopping is applied
    diffaugment_num = read_learning["diffaugment_num"]

    # # trained_models
    read_trained = config_ini["TRAINED"]
    gen_trained = read_trained["gen_trained"]  # path
    disc_trained = read_trained["disc_trained"]  # path


    ######  main  ######

    # # load and process data
    # image data
    image_data = ImageData(data_path["image_data"])
    # network data
    nw_data = NWData(data_path["nw_data"])
    # pp data
    pp_data = PPData(data_path["pp_data"])

    # # learning
    if cross_valid > 0:
        image_data_list = image_data.split_into(cross_valid)  # list of ImageData object
        nw_data_list = nw_data.split_into(cross_valid)  # list of NWData object
        pp_data_list = pp_data.split_into(cross_valid)  # list of PPData object
        loss_data = pd.DataFrame([[None for i in range(2*cross_valid+1)] for j in range(max_itr)], columns=["Itr"]+np.array([[f"Gen_loss({i})", f"Disc_loss({i})"] for i in range(cross_valid)]).reshape(-1).tolist()).astype(float)
        loss_data["Itr"] = np.arange(max_itr)
        for i in range(cross_valid):
            # create model instance
            generator = get_model(gen_name)
            discriminator = get_model(disc_name)
            # dataloader
            dataloader_train, dataloader_val, dataloader_test = get_dataloader(image_data_list[i], nw_data_list[i], pp_data_list[i])
            gan_learning = GANLearning(generator, discriminator, dataloader_train, dataloader_val, dataloader_test, loss_type)
            loss_gen, loss_disc = gan_learning.train(max_itr,
                                       early_stop_itr, diffaugment_num, get_loss=True)  # loss_gen, loss_disc: (iter)
            loss_data.loc[f"Gen_loss({i})", 0:len(loss_gen)] = loss_gen
            loss_data.loc[f"Disc_loss({i})", 0:len(loss_disc)] = loss_disc
        loss_data.to_csv(os.path.join(output_dir, "loss_crossvalid.csv"), index=False)

    if train_model:
        # create model instance
        generator = get_model(gen_name)
        discriminator = get_model(disc_name)
        # dataloader
        dataloader_train, dataloader_val, dataloader_test = get_dataloader(image_data, nw_data,
                                                                           pp_data)
        gan_learning = GANLearning(generator, discriminator, dataloader_train, dataloader_val, dataloader_test,
                                   loss_type, (gen_trained, disc_trained))
        loss_gen, loss_disc = gan_learning.train(max_itr,
                                                 early_stop_itr, diffaugment_num,
                                                 get_loss=True)  # loss_gen, loss_disc: (iter)
        loss_data = pd.DataFrame({"Itr": np.arange(max_itr), "Gen_loss": loss_gen, "Disc_loss": loss_disc})
        loss_data.to_csv(os.path.join(output_dir, "loss_train.csv"), index=False)