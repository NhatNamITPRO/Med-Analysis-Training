from dataset import ImgDateSetForSemeticSegmentation
class SegformerDataset(ImgDateSetForSemeticSegmentation):
    def __init__(self, *args, image_processor=None,augment=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_processor = image_processor  
        self.augment = augment
    def __getitem__(self, index):
        X, Y = super().__getitem__(index)
        Y[Y==255] = 1
        Y = Y.permute(1, 2, 0).squeeze().numpy()
        X = X.permute(1, 2, 0).squeeze().numpy()
        augmented = self.augment(image=X, mask=Y)
        X = augmented['image']
        Y = augmented['mask']
        encoded_inputs = self.image_processor(X.int(), Y, return_tensors="pt")
        for k,v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()
        return encoded_inputs
