
import segmentation_models_pytorch as smp #smp라는 라이브러리
def getModel():
	
	model = smp.Unet(
			encoder_name="resnet18",
			encoder_weights="imagenet",
			in_channels=1,
			classes=3
		)

		
	return model
