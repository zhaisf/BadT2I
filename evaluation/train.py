import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm
import argparse


class ModelTrainer:
    def __init__(self, gpu, train_name, train_dir, output_dir, batch_size=32, num_epochs=6, learning_rate=0.0005):
        self.device = torch.device(gpu)
        self.train_name = train_name
        self.train_dir = train_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        # Data transforms
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

        self._prepare_data()
        self._prepare_model()

    def _prepare_data(self):
        # Load and split dataset
        train_dataset = datasets.ImageFolder(self.train_dir, self.data_transforms['train'])
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(train_dataset, [train_size, val_size])

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def _prepare_model(self):
        # Load pre-trained model and modify final layer for binary classification
        model = models.resnet50(pretrained=True) if self.train_name == "object" else models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        self.model = model.to(self.device)

        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        best_model_wts = self.model.state_dict()
        best_acc = 0.0

        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch}/{self.num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                    dataloader = self.train_loader
                else:
                    self.model.eval()
                    dataloader = self.val_loader

                running_loss = 0.0
                running_corrects = 0

                # Progress bar
                with tqdm(dataloader, unit="batch") as tepoch:
                    for inputs, labels in tepoch:
                        tepoch.set_description(f"{phase} Epoch {epoch}")

                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = self.criterion(outputs, labels)

                            if phase == 'train':
                                self.optimizer.zero_grad()
                                loss.backward()
                                self.optimizer.step()

                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                        tepoch.set_postfix(loss=running_loss / len(dataloader.dataset),
                                            accuracy=running_corrects.double() / len(dataloader.dataset))

                epoch_loss = running_loss / len(dataloader.dataset)
                epoch_acc = running_corrects.double() / len(dataloader.dataset)

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = self.model.state_dict()

        print(f'Best val Acc: {best_acc:4f}')

        # Load best model weights
        self.model.load_state_dict(best_model_wts)

    def save_model(self):
        output_path = f'{self.output_dir}/resnet_finetuned_{self.train_name}.pth'
        torch.save(self.model.state_dict(), output_path)
        print(f"Model saved at {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a ResNet model for binary classification.")
    parser.add_argument("gpu", type=str, default="cuda:0", help="GPU device to use (e.g., 'cuda:0'). Default is 'cuda:0'.")
    parser.add_argument("train_name", type=str, choices=["style", "object", "pixel"], help="Training task name (style or object).")
    parser.add_argument("train_dir", type=str, help="Directory path for training data.")
    parser.add_argument("output_dir", type=str, help="Directory path to save the trained model.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    trainer = ModelTrainer(gpu=args.gpu, train_name=args.train_name, train_dir=args.train_dir, output_dir=args.output_dir)
    trainer.train()
    trainer.save_model()
