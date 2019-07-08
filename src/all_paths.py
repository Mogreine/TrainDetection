class Paths(object):
    def __init__(self, root_path='../'):
        self.WEIGHTS_PATH = root_path + 'weights/'
        self.COCO_WEIGHTS_PATH = root_path + self.WEIGHTS_PATH + 'pretrained/mask_rcnn_coco.h5'
        self.NOMEROFF_NET_WEIGHTS_PATH = root_path + self.WEIGHTS_PATH + 'pretrained/nomeroff_net.h5'

        self.WEIGHT_LOGS_PATH = root_path + self.WEIGHTS_PATH + 'logs/'

        self.IMAGES_PATH = root_path + 'data/images/'
        self.VIDEOS_PATH = root_path + 'data/videos/'
        self.ANNOTATIONS_PATH = root_path + 'data/via/'
        self.ANNOTATION_LOGS_PATH = root_path + self.ANNOTATIONS_PATH + 'logs'

        self.PLOTS_PATH = root_path + 'data/plots/'
