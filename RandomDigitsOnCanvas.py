import torch
import torchvision as tv
import numpy as np
from PIL import Image
import random
from tqdm import tqdm

class RandomDigitsOnCanvas(torch.utils.data.Dataset):
    def __init__(self, train=True, canvas_size=280, min_digits=1, max_digits=10):
        super().__init__()
        self.mnist = tv.datasets.MNIST(root='.', train=train, download=True)
        self.canvas_size = canvas_size
        self.min_digits = min_digits
        self.max_digits = max_digits
        self.tf = tv.transforms.ToTensor()
        self.ti = tv.transforms.ToPILImage()
        self.normalize = tv.transforms.Normalize((0.1307,), (0.3081,))

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        canvas = Image.new('L', (self.canvas_size, self.canvas_size), color=0)
        mask = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
        n_digits = random.randint(self.min_digits, self.max_digits)
        digit_infos = []

        for _ in range(n_digits):
            # 1. Randomly select a digit image
            digit_idx = random.randint(0, len(self.mnist) - 1)
            digit_img, digit_label = self.mnist[digit_idx]

            # 2. Random resize 
            scale = random.uniform(1, 1.2)
            new_size = max(1, int(28 * scale))
            digit_img = digit_img.resize((new_size, new_size), resample=Image.NEAREST)

            # 3. Random rotation
            angle = random.uniform(-8, 8)
            digit_img = digit_img.rotate(angle, resample=Image.NEAREST, expand=True, fillcolor=0)

            # 4. Find a non-overlapping position
            digit_np = np.array(digit_img)
            h, w = digit_np.shape
            max_attempts = 1000
            for attempt in range(max_attempts):
                x = random.randint(0, self.canvas_size - w)
                y = random.randint(0, self.canvas_size - h)
                overlap_area = np.sum(mask[y:y+h, x:x+w] & (digit_np > 0))
                digit_area = np.sum(digit_np > 0)
                if digit_area == 0 or overlap_area / digit_area <= 0.0005:
                    # Place digit
                    canvas.paste(Image.fromarray(digit_np), (x, y))
                    mask[y:y+h, x:x+w] = np.maximum(mask[y:y+h, x:x+w], (digit_np > 0).astype(np.uint8))
                    digit_infos.append({'label': digit_label, 'x': x, 'y': y})
                    break
            else:
                # If can't find a spot, skip this digit
                continue

        # 5. Row-wise sorting: group by y (row), then sort by x within each row
        if digit_infos:
            # Sort by y first
            digit_infos.sort(key=lambda d: d['y'])
            rows = []
            current_row = []
            row_start_y = digit_infos[0]['y']
            for d in digit_infos:
                if abs(d['y'] - row_start_y) > 28:
                    # Sort the current row by x (left to right) before appending
                    current_row.sort(key=lambda d: d['x'])
                    rows.append(current_row)
                    current_row = []
                    row_start_y = d['y']
                current_row.append(d)
            if current_row:
                current_row.sort(key=lambda d: d['x'])
                rows.append(current_row)
            # Flatten the sorted rows
            sorted_digits = [d for row in rows for d in row]
            label = [d['label'] for d in sorted_digits]
        else:
            label = []

        # 6. Convert canvas to tensor and normalize
        canvas_tensor = self.tf(canvas)
        canvas_tensor = self.normalize(canvas_tensor)

        print(f"Requested: {n_digits}, Placed: {len(digit_infos)}")

        return canvas_tensor, torch.tensor(label, dtype=torch.long)

# Example usage:
if __name__ == "__main__":
    # Generate and save synthetic training and test datasets
    train_size = 60000
    test_size = 10000
    canvas_size = 280
    print("Generating synthetic training set...")
    train_ds = RandomDigitsOnCanvas(train=True, canvas_size=canvas_size)
    train_images = torch.zeros((train_size, 1, canvas_size, canvas_size), dtype=torch.float32)
    train_labels = []
    for i in tqdm(range(train_size)):
        img, lbl = train_ds[i]
        train_images[i] = img
        train_labels.append(lbl)
    torch.save((train_images, train_labels), "synthetic_mnist_train.pt")
    print("Saved synthetic_mnist_train.pt")

    print("Generating synthetic test set...")
    test_ds = RandomDigitsOnCanvas(train=False, canvas_size=canvas_size)
    test_images = torch.zeros((test_size, 1, canvas_size, canvas_size), dtype=torch.float32)
    test_labels = []
    for i in tqdm(range(test_size)):
        img, lbl = test_ds[i]
        test_images[i] = img
        test_labels.append(lbl)
    torch.save((test_images, test_labels), "synthetic_mnist_test.pt")
    print("Saved synthetic_mnist_test.pt")

    # Optionally, show a grid of samples
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i in range(16):
        img, lbl = train_ds[i]
        ax = axes[i // 4, i % 4]
        ax.imshow(img.squeeze().numpy(), cmap='gray')
        ax.set_title(f"Label: {lbl.tolist()}", fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    plt.show()