from torchvision import transforms

_MEAN, _STD=[0.6257231324993875, 0.4934742948338769, 0.42569583700621416],[0.25667137400692847, 0.2345312511218496, 0.2305881956020596]

val_transforms = transforms.Compose([
    transforms.Resize(400),
    transforms.CenterCrop(380),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD)
])
