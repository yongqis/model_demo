"""General utility functions"""
import os
import json
import logging


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def get_ab_path(root_path):
    # 得到当前文件夹内的所有子文件
    image_paths = []
    # c_folder当前文件夹完全路径，subfolder当前文件夹的子文件夹，files当前文件夹的子文件
    for c_folder, subfolder, files in os.walk(root_path):
        for file in files:
            if file.endswith('.jpg'):
                image = os.path.join(c_folder, file)
                # print(image)
                image_paths.append(image)
    return image_paths


def get_dict(root_path):
    # 当前root_path的 子文件夹名 作为key，子文件夹内的 子文件完全路径 作为value
    truth_dict = {}
    for c_folder, subfolder, files in os.walk(root_path):
        image_list = []
        for file in files:
            if file.endswith('.jpg'):
                image = os.path.join(c_folder, file)
                image_list.append(image)
        label = c_folder.split('\\')[-1]
        truth_dict[label] = image_list
    return truth_dict


def compute_topk(score, gallery_images, truth_images, query_image, top_k):
    true_num = 0
    error_num = 0
    success = False

    for score_id in range(top_k):
        retrieve_image = gallery_images[score[score_id]]
        find_self = True

        while find_self:
            if retrieve_image in truth_images:
                # 文件名不同，不是同一张图片
                if retrieve_image.split('\\')[-1] != query_image.split('\\')[-1]:
                    true_num += 1
                    find_self = False
                # 文件名相同，找到自己，忽略，计算top2
                else:
                    retrieve_image = gallery_images[score[score_id+1]]

            else:
                # 展示错误的检索结果

                # truth_im = plt.imread(query_image)
                # error_im = plt.imread(a)
                # plt.subplot(1, 2, 1)
                # plt.imshow(truth_im)
                # plt.title('true label: %s' % query_image.split('\\')[-2])
                # plt.subplot(1, 2, 2)
                # plt.imshow(error_im)
                # plt.title('error label: %s' % a.split('\\')[-2])
                # plt.show()
                print('true label:', query_image.split('\\')[-2])
                print('error label', retrieve_image.split('\\')[-2])
                print('---------')

                find_self = False
    # top-k中正确的个数大于0.5 认为该张query图片检索正确
    precision = true_num / top_k
    if precision >= 0.5:
        success = True

    return success
