import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def epoch_pass(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def train_model(model, train_loader, test_loader, optimizer, criterion, device, epochs, tag):
    best_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = epoch_pass(model, train_loader, criterion, device, optimizer)
        te_loss, te_acc = epoch_pass(model, test_loader, criterion, device)
        if te_acc > best_acc:
            best_acc = te_acc
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"[{tag}] Epoch {ep}/{epochs} | "
            f"Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f} | "
            f"Test Loss: {te_loss:.4f}, Test Acc: {te_acc:.4f}"
        )

    model.load_state_dict(best_state)
    return model, best_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10("./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()

    # VGG16 fine-tuning (selected later layers + classifier)
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    vgg.classifier[-1] = nn.Linear(vgg.classifier[-1].in_features, 10)

    for p in vgg.features.parameters():
        p.requires_grad = False
    for p in vgg.features[24:].parameters():  # unfreeze later conv blocks for fine-tuning
        p.requires_grad = True
    for p in vgg.classifier.parameters():
        p.requires_grad = True

    vgg = vgg.to(device)
    vgg_optimizer = optim.Adam((p for p in vgg.parameters() if p.requires_grad), lr=1e-4, weight_decay=1e-4)
    vgg, vgg_acc = train_model(vgg, train_loader, test_loader, vgg_optimizer, criterion, device, epochs=6, tag="VGG16 Fine-Tune")

    # ResNet50 feature extraction (head only)
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for p in resnet.parameters():
        p.requires_grad = False
    resnet.fc = nn.Linear(resnet.fc.in_features, 10)
    resnet = resnet.to(device)

    resnet_optimizer = optim.Adam(resnet.fc.parameters(), lr=1e-3)
    resnet, resnet_acc = train_model(
        resnet, train_loader, test_loader, resnet_optimizer, criterion, device, epochs=5, tag="ResNet50 Feature Extract"
    )

    print("\n=== Task 4 Comparison Summary ===")
    print(f"VGG16 Fine-Tuning Test Accuracy: {vgg_acc:.4f}")
    print(f"ResNet50 Feature Extraction Test Accuracy: {resnet_acc:.4f}")


if __name__ == "__main__":
    main()
