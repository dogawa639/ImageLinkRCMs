if __name__ == "__main__":
    import configparser
    import os
    import time
    from logger import Logger

    CONFIG = "../../config/config_test.ini"
    config = configparser.ConfigParser()
    config.read(CONFIG, encoding="utf-8")
    read_save = config["SAVE"]
    log_dir = read_save["log_dir"]

    logger = Logger(os.path.join(log_dir, "log_test.json"), CONFIG)
    for i in range(10):
        logger.add_log("test", i)
        time.sleep(1)
    logger.close()
    logger.save_fig(os.path.join(log_dir, "log_test.png"))