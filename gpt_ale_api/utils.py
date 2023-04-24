import os

default_local_path_root = "./download"

# TODO: enable user to specify local path root
def get_local_path_from_cloud_path(cloud_path, local_path_root=default_local_path_root):
    return os.path.join(default_local_path_root,cloud_path)