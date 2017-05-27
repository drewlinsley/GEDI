from os.path import join as pjoin


# Main configuration file for the GEDI project.
class GEDIconfig(object):  # at some point use pjoin throughout
    def __init__(self):

        self.which_dataset = 'gfp'  # gedi gfp masked_gfp or ratio

        # Parameters for extracting image patches from supplied TIFF files
        self.home_dir = '/media/data/GEDI/drew_images/'
        self.project_stem = 'project_files'
        # The suffix for the original images.
        # Connected to the prefixes below with a
        # '2_14_17' # 'middle' # 'human' # 'human_bs'
        self.experiment_image_set = 'all_rh_analysis_rat'
        # The prefix for the original images/object categories

        ##############
        blinded = False
        test_set = True
        # Typically ['Dead', 'Live']. Use [''] for blinded.
        if blinded:
            self.image_prefixes = ['']
            self.image_prefixes = [self.experiment_image_set]
        else:
            self.image_prefixes = ['Dead', 'Live']
            # Append the experiment name to the images
            self.image_prefixes = [
                x + '_' + self.experiment_image_set
                for x in self.image_prefixes]
        self.validation_type = 'separate'  # separate validation/training or sampled
        ##############

        self.original_image_dir = 'original_images'
        self.processed_image_patch_dir = pjoin(
            self.home_dir, self.project_stem, 'image_patches')
        self.raw_im_dirs = [pjoin(
            self.original_image_dir, x) for x in self.image_prefixes]
        self.output_dirs = [pjoin(
            self.processed_image_patch_dir,
            self.which_dataset + '_' + x) for x in self.image_prefixes]
        self.raw_im_ext = '.tif'
        self.im_ext = '.png'  # preserve floating point
        self.channel = 0  # 0-5 timepoints
        if self.which_dataset == 'gfp':
            self.panel = 0
            self.divide_panel = None
        elif self.which_dataset == 'masked_gfp':
            self.panel = 1
            self.divide_panel = None
        elif self.which_dataset == 'gedi':
            self.panel = 2
            self.divide_panel = None
        elif self.which_dataset == 'ratio':
            self.panel = 2
            self.divide_panel = 0
        self.max_gedi = 16117  # 16383.5  # 16383.5 is for background subtraction/None
        self.min_gedi = 0  # 0  # 0 is for background subtraction/None

        # Paths for creating tfrecords.
        self.GEDI_path = pjoin(self.home_dir, self.project_stem)
        self.label_directories = [
            self.which_dataset + '_' + x for x in self.image_prefixes]
        self.train_directory = pjoin(
            self.GEDI_path, 'train',
            self.experiment_image_set + '_' + self.which_dataset + '/')
        self.validation_directory = pjoin(
            self.GEDI_path, 'validation',
            self.experiment_image_set + '_' + self.which_dataset + '/')
        self.tfrecord_dir = pjoin(
            self.GEDI_path, 'tfrecords',
            self.experiment_image_set + '_' + self.which_dataset + '/')
        # Which sets to produce in seperate tfrecords
        if blinded:
            self.tvt_flags = 'val'
        elif blinded == False and test_set == True:
            self.tvt_flags = ['train', 'val', 'test']
            self.test_directory = pjoin(
                self.GEDI_path, 'test',
                self.experiment_image_set + '_' + self.which_dataset + '/')
        else:
            self.tvt_flags = ['train', 'val']  # ['train','val','test']
        self.max_file = 'maximum_value.npz'

        # Data parameters for tfrecords
        self.train_proportion = 0.9  # validation with 10% of data
        self.num_threads = 4
        self.train_shards = 1  # 024 #number of training images per record
        self.validation_shards = 1  # 024 #number of training images per record
        self.train_batch = 64  # number of training images per record
        self.validation_batch = 32  # 64
        # Normalize GEDIs in uint8 to 0-1 float. May be redundant.
        self.normalize = False

        # Model training
        self.src_dir = '/home/drew/Documents/tf_experiments/'
        self.epochs = 200  # Increase since we are augmenting
        self.keep_checkpoints = self.epochs  # keep checkpoints at every epoch
        self.train_checkpoint = pjoin(self.GEDI_path, 'train_checkpoint/')
        self.train_summaries = pjoin(self.GEDI_path, 'train_summaries/')
        self.vgg16_weight_path = pjoin(
            self.src_dir, 'pretrained_weights', 'vgg16.npy')
        self.model_image_size = [224, 224, 3]
        self.gedi_image_size = [300, 300, 3]
        self.output_shape = 2  # how many categories for classification
        # choose from ['conv5_1', 'fc6', 'conv5_3', 'fc7', 'fc8', 'conv5_2',
        # 'conv4_1', 'conv4_2', 'conv4_3', 'conv3_3', 'conv3_2',
        # 'conv3_1', 'conv1_1', 'conv1_2', 'conv2_2', 'conv2_1']
        self.fine_tune_layers = [
            'conv5_1',
            'conv5_2',
            'conv5_3',
            'fc6',
            'fc7',
            'fc8'
        ]
        self.batchnorm_layers = ['fc6', 'fc7', 'fc8']
        self.optimizer = 'sgd'  # 'adam'
        self.hold_lr = 1e-8
        self.new_lr = 1e-4
        # choose from: left_right, up_down, random_crop, random_brightness,
        # random_contrast, rotate
        self.data_augmentations = [
            'left_right', 'up_down', 'random_crop',
            'rotate', 'random_brightness', 'random_contrast']
        self.balance_cost = True  # True  # True

        # Model testing
        self.results = pjoin(self.GEDI_path, 'results/')

        ######
        # Visualization settings
        ######

        # Directory with images for heatmaps
        self.heatmap_source_images = pjoin(
            self.GEDI_path, 'images_for_heatmaps')
        self.heatmap_dataset_images = pjoin(
            self.heatmap_source_images, self.which_dataset)
        self.heatmap_image_labels = pjoin(
            self.GEDI_path, 'list_of_' + self.which_dataset + '_labels.txt')

        # Images for visualization parameters
        # > 0 = number of images, < 0 = proportion of images
        self.heatmap_image_amount = 90
        self.heatmap_batch = 10

        # Bubbles parameters
        self.visualization_output = pjoin(
            self.GEDI_path, 'visualizations', self.which_dataset)
        self.generate_plots = True
        self.use_true_label = False
        self.bubbles_ckpt = pjoin(
            self.train_checkpoint, 'bs_and_no_bs_model', 'model_14000.ckpt-14000')
        self.block_size = 10
        self.block_stride = 1

    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

