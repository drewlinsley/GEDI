from os.path import join as pjoin


# Main configuration file for the GEDI project.
class GEDIconfig(object):  # at some point use pjoin throughout
    def __init__(self):

        ##########################
        # Basic project settings #
        #########################
        #  Directory pointers for CNN and GEDI files
        self.home_dir = '/media/data/GEDI/drew_images/'  # Arbitrary home directory
        self.project_stem = 'project_files'  # Folder within home where analysis files will go
        self.original_image_dir = 'original_images'  # Folder within home where TIFF image files exist
        self.src_dir = '/home/drew/Documents/GEDI_update/'  # Project source directory

        # Choose which image panel you want to run your analyses on:
        # -- gedi gfp masked_gfp or ratio
        self.which_dataset = 'gfp'

        # The suffix for tiff image folders that are input for the CNN.
        # This exists in original_image_dir.
        # Expectation is that there is a seperate folder per image category (e.g. Live/Dead)
        # Naming convention is XXX_{self.experiment_image_set}
        # Some recent examples: 'all_rh_analysis_rat', 'human', 'human_bs'
        self.experiment_image_set = 'all_rh_analysis_rat'

        # Parameters for preparing CNN image sets
        blinded = False  # Pass a dataset through a trained CNN without supplying labels
        test_set = True  # If you have a {Training/Validation} and seperate {Test} image sets
        self.raw_im_ext = '.tif'  # Extension of neuron images
        self.im_ext = '.png'  # If you are going from tiff -> image -> CNN image format
        self.channel = 0  # Which image timepoint do we extract: 0-5 timepoints
        self.easy_analysis = False  # Set to True if you are simply trying to pass a new image set through a trained model.

        # CNN settings you should feel free to tweak
        self.vgg16_weight_path = pjoin(  # Location of pretrained CNN weights.
            self.src_dir, 'pretrained_weights', 'vgg16.npy')
        self.gedi_image_size = [300, 300, 3]  # Size of GEDI TIFF files.
        self.output_shape = 2  # How many categories for classification? If Dead/Live this is 2.

        ###########################
        # Advanced image settings #
        ###########################
        if blinded:
            self.image_prefixes = ['']
            self.image_prefixes = [self.experiment_image_set]
        else:
            self.image_prefixes = ['Dead', 'Live']  # Image directory prefixes (see self.experiment_image_set)
            # Append the experiment name to the images
            self.image_prefixes = [
                x + '_' + self.experiment_image_set
                for x in self.image_prefixes]

        self.processed_image_patch_dir = pjoin(
            self.home_dir, self.project_stem, 'image_patches')
        self.raw_im_dirs = [pjoin(
            self.original_image_dir, x) for x in self.image_prefixes]
        self.output_dirs = [pjoin(
            self.processed_image_patch_dir,
            self.which_dataset + '_' + x) for x in self.image_prefixes]
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
        self.max_gedi = 16117  # Max value of imaging equiptment
        self.min_gedi = 0  # 0  # Min value of imaging equiptment

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
        if self.easy_analysis:
            self.tvt_flags = ['train']
        else:
            if blinded:
                self.tvt_flags = 'val'
            elif blinded == False and test_set == True:
                self.tvt_flags = ['train', 'val', 'test']
                self.test_directory = pjoin(
                    self.GEDI_path, 'test',
                    self.experiment_image_set + '_' + self.which_dataset + '/')
            else:
                self.tvt_flags = ['train', 'val']  # ['train','val','test']
        self.max_file = 'maximum_value.npz'  # File to save with empirical max values. Goes into CNN image data directory.

        # Data parameters for tfrecords
        self.train_proportion = 0.9  # If training/validation, proportion of training images. 1 - this is validation.
        self.num_threads = 4  # Number of CPU threads to use to create the CNN data.
        self.train_shards = 1
        self.validation_shards = 1  
        self.train_batch = 64  # Number of training images per iteration of training.
        self.validation_batch = 32  # Number of validation images to evaluate after every N iterations. 
        # Normalize GEDIs in uint8 to 0-1 float. May be redundant.
        self.normalize = False  # Normalize each image to [0, 1] during preprocessing.


        #####################
        # Modeling settings #
        #####################
        self.model_image_size = [224, 224, 3]
        self.epochs = 200  # Number of training epochs
        self.keep_checkpoints = self.epochs  # keep checkpoints at every epoch
        self.train_checkpoint = pjoin(self.GEDI_path, 'train_checkpoint/')
        self.train_summaries = pjoin(self.GEDI_path, 'train_summaries/')
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

        ##########################
        # Visualization settings #
        #########################

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

