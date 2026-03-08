# Package imports
import os
import string
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import xml.etree.ElementTree as ET

dir_path = os.path.dirname(os.path.realpath(__file__))

lines_root = dir_path + "/lines"
xml_root = dir_path + "/xml"



# =========================
# DATASET
# =========================
class IAMLinesDataset(Dataset):
    def __init__(self, transform=None):
        self.lines_root = lines_root
        self.xml_root = xml_root
        self.transform = transform
        self.samples = []

        # Parse all xml files
        for xml_file in os.listdir(xml_root):
            if not xml_file.endswith(".xml"):
                continue

            xml_path = os.path.join(xml_root, xml_file)

            tree = ET.parse(xml_path)
            root = tree.getroot()

            for line in root.iter("line"):
                line_id = line.attrib["id"]
                text = line.attrib["text"]

                folder1 = line_id.split("-")[0]
                folder2 = "-".join(line_id.split("-")[:2])

                img_path = os.path.join(
                    lines_root, folder1, folder2, line_id + ".png"
                )

                if os.path.exists(img_path):
                    self.samples.append((img_path, text))

        print("Loaded", len(self.samples), "line samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, label

# =========================
# COLLATE FUNCTION
# =========================
def collate_fn(batch):
    images, labels = zip(*batch)
    widths = [img.shape[2] for img in images]
    max_width = max(widths)
    padded_images = [F.pad(img, (0, max_width - img.shape[2])) for img in images]
    images = torch.stack(padded_images)
    return images, labels

# =========================
# TRANSFORM
# =========================
class ResizeHeight:
    def __init__(self, height):
        self.height = height

    def __call__(self, img):
        w, h = img.size
        new_w = int(w * self.height / h)
        return img.resize((new_w, self.height), Image.BILINEAR)

transform = transforms.Compose([
    ResizeHeight(64), # builtin resize makes shorter side 64, may cause issues if dataset contains "lines" that are just a single character long, otherwise should be fine
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# =========================
# CHARACTER MAPS
# =========================
characters = string.ascii_letters + string.digits + string.punctuation + " " # IAMS doesn't contain all punctuation, so this covers more characters than needed, but should still be fine
char_to_idx = {c: i+1 for i, c in enumerate(characters)}
idx_to_char = {i: c for c, i in char_to_idx.items()}
blank_label = 0

def encode_label(text):
    encoded = [char_to_idx[c] for c in text if c in char_to_idx]
    return torch.tensor(encoded, dtype=torch.long)