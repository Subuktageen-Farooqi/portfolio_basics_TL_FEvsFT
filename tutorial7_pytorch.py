import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=5, tag="Model"):
    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        if test_acc > best_acc:
            best_acc = test_acc
            best_weights = copy.deepcopy(model.state_dict())
        print(
            f"[{tag}] Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
        )

    model.load_state_dict(best_weights)
    return model, best_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()

    # Feature extraction with VGG16
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    for param in vgg.features.parameters():
        param.requires_grad = False

    in_features_vgg = vgg.classifier[-1].in_features
    vgg.classifier[-1] = nn.Linear(in_features_vgg, 10)
    vgg = vgg.to(device)

    vgg_params = [p for p in vgg.parameters() if p.requires_grad]
    vgg_optimizer = optim.Adam(vgg_params, lr=1e-3)
    vgg, vgg_best_acc = train_model(
        vgg, train_loader, test_loader, criterion, vgg_optimizer, device, epochs=5, tag="VGG16 Feature Extract"
    )

    # Fine-tuning with ResNet50: phase 1 (head only)
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = nn.Linear(resnet.fc.in_features, 10)
    resnet = resnet.to(device)

    head_optimizer = optim.Adam(resnet.fc.parameters(), lr=1e-3)
    resnet, _ = train_model(
        resnet, train_loader, test_loader, criterion, head_optimizer, device, epochs=3, tag="ResNet50 Head Train"
    )

    # Fine-tuning with ResNet50: phase 2 (unfreeze last few layers)
    for name, param in resnet.named_parameters():
        if name.startswith("layer4") or name.startswith("fc"):
            param.requires_grad = True
        else:
            param.requires_grad = False

    ft_optimizer = optim.Adam((p for p in resnet.parameters() if p.requires_grad), lr=1e-5)
    resnet, resnet_best_acc = train_model(
        resnet, train_loader, test_loader, criterion, ft_optimizer, device, epochs=5, tag="ResNet50 Fine-Tune"
    )

    print("\n=== Final Comparison ===")
    print(f"VGG16 Feature Extraction Accuracy: {vgg_best_acc:.4f}")
    print(f"ResNet50 Fine-Tuning Accuracy: {resnet_best_acc:.4f}")
    if resnet_best_acc > vgg_best_acc:
        print("ResNet50 fine-tuning performed better on CIFAR-10 in this run.")
    elif resnet_best_acc < vgg_best_acc:
        print("VGG16 feature extraction performed better on CIFAR-10 in this run.")
    else:
        print("Both approaches achieved the same accuracy in this run.")


if __name__ == "__main__":
    main()
