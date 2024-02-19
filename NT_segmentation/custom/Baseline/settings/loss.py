from torch import nn
from monai.losses.dice import DiceLoss, one_hot
from monai.metrics import DiceMetric

def getLoss():
	# criterion = nn.CrossEntropyLoss()
	criterion = DiceLoss()

	return criterion