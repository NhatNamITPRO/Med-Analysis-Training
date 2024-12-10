from dataset import ImgDateSetForSemeticSegmentation
class UnetDataset(ImgDateSetForSemeticSegmentation):
    def __init__(self, *args, augment=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.augment = augment
    def __getitem__(self, index):
        X, Y = super().__getitem__(index)
        Y[Y==255] = 1
        Y = Y.permute(1, 2, 0).squeeze().numpy()
        X = X.permute(1, 2, 0).squeeze().numpy()
        augmented = self.augment(image=X, mask=Y)
        X = augmented['image']
        Y = augmented['mask']
        return X, Y
