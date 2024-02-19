from easydict import EasyDict as eDict


def getArg():
	arg = eDict()

	arg.batch = 16
	arg.epoch = 1
	arg.lr = 1e-4
	arg.seed = 21
	arg.save_capacity = 5
	

	# arg.image_root = "D:/NT data/Merge2D3D/image_total_merge"
	arg.train_img_label = "D:/NT data/Merge2D3D/label_total_merge"
	arg.train_img_path = "D:/NT data/Merge2D3D/image_total_merge"

	# arg.train_json = "train_0.json"
	# arg.val_json = "valid_0.json"
	# arg.test_json = "test.json"

	arg.output_path = "D:/NT data/"

	# arg.train_worker = 4
	# arg.valid_worker = 4
	# arg.test_worker = 4

	# arg.wandb = True
	# arg.wandb_project = "segmentation"
	# arg.wandb_entity = "cv4"

	arg.custom_name = "test"
	
	arg.TTA = True
	arg.test_batch = 4
	arg.csv_size = 256

	return arg