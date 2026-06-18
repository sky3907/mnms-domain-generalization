from dataset import build_dataset
from mnms_dataset import MNMSDataset

samples = build_dataset()

dataset = MNMSDataset(samples)

print("Total slice samples:", len(dataset))

image, mask = dataset[0]

print("Image shape:", image.shape)
print("Mask shape:", mask.shape)
print("Mask labels:", mask.unique())
for i in [0, 50, 100, 500, 1000]:

    image, mask = dataset[i]

    print(
        i,
        mask.unique()
    )