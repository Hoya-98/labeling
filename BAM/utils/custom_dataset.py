from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import os
from PIL import Image, ImageOps

class CustomDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.classes = sorted(os.listdir(img_dir))
        self.images = []
        self.labels = []
        self.class_name_list = []
        if transform == None:
            self.transform = transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        for idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(img_dir, class_name)
            self.class_name_list.append(class_name)

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith('.jpg'):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(idx)
                                       
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        image = ImageOps.exif_transpose(image) # 화상 이미지 회전 이슈
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
    