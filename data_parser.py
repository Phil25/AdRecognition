from os import path

def get_data(cwd):
    data_path = path.join(cwd, "data", "ad.data")

    with open(data_path, "r") as f: