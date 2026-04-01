import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms


class CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if is_train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def train_and_eval(model, train_loader, val_loader, test_loader, epochs, lr, device, tag, weight_decay=0.0):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=weight_decay)
    model = model.to(device)

    best_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"[{tag}] Epoch {epoch}/{epochs} | "
            f"Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    model.load_state_dict(best_state)
    _, test_acc = run_epoch(model, test_loader, criterion, device)
    return test_acc


def build_vgg16_feature_extractor():
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    for p in model.features.parameters():
        p.requires_grad = False
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 10)
    return model


def build_resnet50_feature_extractor():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tf_custom = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    tf_pretrained = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_custom = datasets.CIFAR10("./data", train=True, download=True, transform=tf_custom)
    test_custom = datasets.CIFAR10("./data", train=False, download=True, transform=tf_custom)
    train_pre = datasets.CIFAR10("./data", train=True, download=True, transform=tf_pretrained)
    test_pre = datasets.CIFAR10("./data", train=False, download=True, transform=tf_pretrained)

    custom_train_len = int(0.9 * len(train_custom))
    custom_val_len = len(train_custom) - custom_train_len
    pre_train_len = int(0.9 * len(train_pre))
    pre_val_len = len(train_pre) - pre_train_len

    custom_train_ds, custom_val_ds = random_split(
        train_custom, [custom_train_len, custom_val_len], generator=torch.Generator().manual_seed(42)
    )
    pre_train_ds, pre_val_ds = random_split(
        train_pre, [pre_train_len, pre_val_len], generator=torch.Generator().manual_seed(42)
    )

    train_custom_loader = DataLoader(custom_train_ds, batch_size=128, shuffle=True, num_workers=2)
    val_custom_loader = DataLoader(custom_val_ds, batch_size=256, shuffle=False, num_workers=2)
    test_custom_loader = DataLoader(test_custom, batch_size=256, shuffle=False, num_workers=2)

    train_pre_loader = DataLoader(pre_train_ds, batch_size=64, shuffle=True, num_workers=2)
    val_pre_loader = DataLoader(pre_val_ds, batch_size=128, shuffle=False, num_workers=2)
    test_pre_loader = DataLoader(test_pre, batch_size=128, shuffle=False, num_workers=2)

    custom_model = CustomCNN(num_classes=10)
    custom_acc = train_and_eval(custom_model, train_custom_loader, val_custom_loader, test_custom_loader, epochs=8, lr=1e-3, device=device, tag="Custom CNN")

    vgg_model = build_vgg16_feature_extractor()
    vgg_acc = train_and_eval(vgg_model, train_pre_loader, val_pre_loader, test_pre_loader, epochs=5, lr=1e-3, device=device, tag="VGG16 Feature Extract")

    resnet_model = build_resnet50_feature_extractor()
    resnet_acc = train_and_eval(resnet_model, train_pre_loader, val_pre_loader, test_pre_loader, epochs=5, lr=1e-3, device=device, tag="ResNet50 Feature Extract")

    print("\n=== Task 3 Final Comparison (CIFAR-10 Test Accuracy) ===")
    print("{:<35} {:>10}".format("Model", "Accuracy"))
    print("-" * 48)
    print("{:<35} {:>10.4f}".format("Custom CNN (from scratch)", custom_acc))
    print("{:<35} {:>10.4f}".format("VGG16 Feature Extraction", vgg_acc))
    print("{:<35} {:>10.4f}".format("ResNet50 Feature Extraction", resnet_acc))


if __name__ == "__main__":
    main()
