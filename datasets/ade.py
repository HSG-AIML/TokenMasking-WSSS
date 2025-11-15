"""ADE20K Semantic Segmentation Dataset."""

import os

import albumentations as A
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, RandomSampler


class ADE20KSegmentation:
    """ADE20K Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to ADE20K folder. Default is './datasets/ade'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    >>> ])
    >>> # Create Dataset
    >>> trainset = ADE20KSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """

    BASE_DIR = "ADEChallengeData2016"
    NUM_CLASS = 151

    def __init__(self, root="../ADEChallengeData2016", split="train", **kwargs):
        assert os.path.exists(root), (
            "Please setup the dataset using ../datasets/ade20k.py"
        )
        self.images, self.masks = _get_ade20k_pairs(root, split)
        assert len(self.images) == len(self.masks)
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")
        print("Found {} images in the folder {}".format(len(self.images), root))

        self.augmentation = A.Compose(
            [
                A.Resize(
                    height=208,
                    width=320,
                    p=1,
                    interpolation=cv2.INTER_NEAREST,
                ),
                ToTensorV2(),
            ]
        )

    def __getitem__(self, index):
        img = cv2.imread(self.images[index])
        mask = cv2.imread(self.masks[index])

        label = np.zeros(self.NUM_CLASS)
        unique = list(np.unique(mask.copy()))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / img.max()

        if -1 in unique:
            unique.remove(-1)
        label[np.array(unique)] = 1

        augmented = self.augmentation(image=img, mask=mask)

        img = augmented["image"].float()
        mask = augmented["mask"]

        output_tensor = {
            "img": img,
            "imgfile": os.path.basename(self.images[index]),
            "mask": mask,
            "y": torch.from_numpy(label.copy()),
            "y_ohe": torch.from_numpy(label.copy()),
        }

        return output_tensor

    def _mask_transform(self, mask):
        return torch.LongTensor(np.array(mask).astype("int32") - 1)

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 1

    @property
    def colors(self):
        return (
            "#000000",
            "#787878",
            "#B47878",
            "#06E6E6",
            "#503232",
            "#04C803",
            "#787850",
            "#8C8C8C",
            "#CC05FF",
            "#E6E6E6",
            "#04FA07",
            "#E005FF",
            "#EBFF07",
            "#96053D",
            "#787846",
            "#08FF33",
            "#FF0652",
            "#8FFF8C",
            "#CCFF04",
            "#FF3307",
            "#CC4603",
            "#0066C8",
            "#3DE6FA",
            "#FF0633",
            "#0B66FF",
            "#FF0747",
            "#FF09E0",
            "#0907E6",
            "#DCDCDC",
            "#FF095C",
            "#7009FF",
            "#08FFD6",
            "#07FFE0",
            "#FFB806",
            "#0AFF47",
            "#FF290A",
            "#07FFFF",
            "#E0FF08",
            "#6608FF",
            "#FF3D06",
            "#FFC207",
            "#FF7A08",
            "#00FF14",
            "#FF0829",
            "#FF0599",
            "#0633FF",
            "#EB0CFF",
            "#A09614",
            "#00A3FF",
            "#8C8C8C",
            "#FA0A0F",
            "#14FF00",
            "#1FFF00",
            "#FF1F00",
            "#FFE000",
            "#99FF00",
            "#0000FF",
            "#FF4700",
            "#00EBFF",
            "#00ADFF",
            "#1F00FF",
            "#0BC8C8",
            "#FF5200",
            "#00FFF5",
            "#003DFF",
            "#00FF70",
            "#00FF85",
            "#FF0000",
            "#FFA300",
            "#FF6600",
            "#C2FF00",
            "#008FFF",
            "#33FF00",
            "#0052FF",
            "#00FF29",
            "#00FFAD",
            "#0A00FF",
            "#ADFF00",
            "#00FF99",
            "#FF5C00",
            "#FF00FF",
            "#FF00F5",
            "#FF0066",
            "#FFAD00",
            "#FF0014",
            "#FFB8B8",
            "#001FFF",
            "#00FF3D",
            "#0047FF",
            "#FF00CC",
            "#00FFC2",
            "#00FF52",
            "#000AFF",
            "#0070FF",
            "#3300FF",
            "#00C2FF",
            "#007AFF",
            "#00FFA3",
            "#FF9900",
            "#00FF0A",
            "#FF7000",
            "#8FFF00",
            "#5200FF",
            "#A3FF00",
            "#FFEB00",
            "#08B8AA",
            "#8500FF",
            "#00FF5C",
            "#B800FF",
            "#FF001F",
            "#00B8FF",
            "#00D6FF",
            "#FF0070",
            "#5CFF00",
            "#00E0FF",
            "#70E0FF",
            "#46B8A0",
            "#A300FF",
            "#9900FF",
            "#47FF00",
            "#FF00A3",
            "#FFCC00",
            "#FF008F",
            "#00FFEB",
            "#85FF00",
            "#FF00EB",
            "#F500FF",
            "#FF007A",
            "#FFF500",
            "#0ABED4",
            "#D6FF00",
            "#00CCFF",
            "#1400FF",
            "#FFFF00",
            "#0099FF",
            "#0029FF",
            "#00FFCC",
            "#2900FF",
            "#29FF00",
            "#AD00FF",
            "#00F5FF",
            "#4700FF",
            "#7A00FF",
            "#00FFB8",
            "#005CFF",
            "#B8FF00",
            "#0085FF",
            "#FFD600",
            "#19C2C2",
            "#66FF00",
            "#5C00FF",
        )

    @property
    def classes(self):
        """Category names."""
        return (
            "background",
            "wall",
            "building, edifice",
            "sky",
            "floor, flooring",
            "tree",
            "ceiling",
            "road, route",
            "bed",
            "windowpane, window",
            "grass",
            "cabinet",
            "sidewalk, pavement",
            "person, individual, someone, somebody, mortal, soul",
            "earth, ground",
            "door, double door",
            "table",
            "mountain, mount",
            "plant, flora, plant life",
            "curtain, drape, drapery, mantle, pall",
            "chair",
            "car, auto, automobile, machine, motorcar",
            "water",
            "painting, picture",
            "sofa, couch, lounge",
            "shelf",
            "house",
            "sea",
            "mirror",
            "rug, carpet, carpeting",
            "field",
            "armchair",
            "seat",
            "fence, fencing",
            "desk",
            "rock, stone",
            "wardrobe, closet, press",
            "lamp",
            "bathtub, bathing tub, bath, tub",
            "railing, rail",
            "cushion",
            "base, pedestal, stand",
            "box",
            "column, pillar",
            "signboard, sign",
            "chest of drawers, chest, bureau, dresser",
            "counter",
            "sand",
            "sink",
            "skyscraper",
            "fireplace, hearth, open fireplace",
            "refrigerator, icebox",
            "grandstand, covered stand",
            "path",
            "stairs, steps",
            "runway",
            "case, display case, showcase, vitrine",
            "pool table, billiard table, snooker table",
            "pillow",
            "screen door, screen",
            "stairway, staircase",
            "river",
            "bridge, span",
            "bookcase",
            "blind, screen",
            "coffee table, cocktail table",
            "toilet, can, commode, crapper, pot, potty, stool, throne",
            "flower",
            "book",
            "hill",
            "bench",
            "countertop",
            "stove, kitchen stove, range, kitchen range, cooking stove",
            "palm, palm tree",
            "kitchen island",
            "computer, computing machine, computing device, data processor, "
            "electronic computer, information processing system",
            "swivel chair",
            "boat",
            "bar",
            "arcade machine",
            "hovel, hut, hutch, shack, shanty",
            "bus, autobus, coach, charabanc, double-decker, jitney, motorbus, "
            "motorcoach, omnibus, passenger vehicle",
            "towel",
            "light, light source",
            "truck, motortruck",
            "tower",
            "chandelier, pendant, pendent",
            "awning, sunshade, sunblind",
            "streetlight, street lamp",
            "booth, cubicle, stall, kiosk",
            "television receiver, television, television set, tv, tv set, idiot "
            "box, boob tube, telly, goggle box",
            "airplane, aeroplane, plane",
            "dirt track",
            "apparel, wearing apparel, dress, clothes",
            "pole",
            "land, ground, soil",
            "bannister, banister, balustrade, balusters, handrail",
            "escalator, moving staircase, moving stairway",
            "ottoman, pouf, pouffe, puff, hassock",
            "bottle",
            "buffet, counter, sideboard",
            "poster, posting, placard, notice, bill, card",
            "stage",
            "van",
            "ship",
            "fountain",
            "conveyer belt, conveyor belt, conveyer, conveyor, transporter",
            "canopy",
            "washer, automatic washer, washing machine",
            "plaything, toy",
            "swimming pool, swimming bath, natatorium",
            "stool",
            "barrel, cask",
            "basket, handbasket",
            "waterfall, falls",
            "tent, collapsible shelter",
            "bag",
            "minibike, motorbike",
            "cradle",
            "oven",
            "ball",
            "food, solid food",
            "step, stair",
            "tank, storage tank",
            "trade name, brand name, brand, marque",
            "microwave, microwave oven",
            "pot, flowerpot",
            "animal, animate being, beast, brute, creature, fauna",
            "bicycle, bike, wheel, cycle",
            "lake",
            "dishwasher, dish washer, dishwashing machine",
            "screen, silver screen, projection screen",
            "blanket, cover",
            "sculpture",
            "hood, exhaust hood",
            "sconce",
            "vase",
            "traffic light, traffic signal, stoplight",
            "tray",
            "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, "
            "dustbin, trash barrel, trash bin",
            "fan",
            "pier, wharf, wharfage, dock",
            "crt screen",
            "plate",
            "monitor, monitoring device",
            "bulletin board, notice board",
            "shower",
            "radiator",
            "glass, drinking glass",
            "clock",
            "flag",
        )


def _get_ade20k_pairs(folder, mode="train"):
    img_paths = []
    mask_paths = []
    if mode == "train":
        img_folder = os.path.join(folder, "images/training")
        mask_folder = os.path.join(folder, "annotations/training")
    else:
        img_folder = os.path.join(folder, "images/validation")
        mask_folder = os.path.join(folder, "annotations/validation")
    count = 0
    for filename in os.listdir(img_folder):
        basename, _ = os.path.splitext(filename)
        if filename.endswith(".jpg"):
            imgpath = os.path.join(img_folder, filename)
            maskname = basename + ".png"
            maskpath = os.path.join(mask_folder, maskname)
            if os.path.isfile(maskpath):
                img_paths.append(imgpath)
                mask_paths.append(maskpath)
            else:
                print("cannot find the mask:", maskpath)

    return img_paths, mask_paths


class Ade20KDataModule(pl.LightningDataModule):
    def __init__(
        self,
    ):
        super(Ade20KDataModule).__init__()

    def prepare_data(self):
        pass

    def setup(
        self,
        train_batch_size,
        eval_batch_size,
        datadir="/ds2/computer_vision/ADEChallengeData2016",
    ):
        self.trainset = ADE20KSegmentation(root=datadir, split="train", transform=True)

        self.valset = ADE20KSegmentation(
            root=datadir,
            split="val",
            transform=True,
        )

        print(len(self.trainset), len(self.valset))

        self.train_bs = train_batch_size
        self.eval_bs = eval_batch_size

    def train_dataloader(self):
        return DataLoader(
            self.trainset, batch_size=self.train_bs, num_workers=4, pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset, batch_size=self.eval_bs, num_workers=4, pin_memory=True
        )


if __name__ == "__main__":
    data_module = Ade20KDataModule()

    train_batch_size = 1
    eval_batch_size = 1
    data_module.setup(train_batch_size, eval_batch_size)

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    count = 0
    nb_labels = 0
    for batch in train_loader:
        count_of_ones = torch.sum(batch["y_ohe"] == 1.0).item()
        nb_labels += count_of_ones
        count += 1
    print("Average label for train_loader: ", nb_labels / count, nb_labels, count)

    for batch in val_loader:
        count_of_ones = torch.sum(batch["y_ohe"] == 1.0).item()
        nb_labels += count_of_ones
        count += 1
    print("Average label for val_loader: ", nb_labels / count, nb_labels, count)
