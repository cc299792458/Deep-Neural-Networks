import os
import numpy as np
import open3d as o3d

from tqdm import tqdm
from torch.utils.data import Dataset

class ModelNet40(Dataset):
    def __init__(self, root_dir="./data/ModelNet40", split='train', categories=None, transform=None, show_progress=True):
        """
        ModelNet40 dataset class.
        :param root_dir: Path to the extracted ModelNet40 folder (default is "./data/ModelNet40").
        :param split: Dataset split, 'train' or 'test'.
        :param categories: List of categories to load, if None, load all categories.
        :param transform: Optional point cloud transformation function.
        :param show_progress: Whether to show a progress bar during data loading.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.show_progress = show_progress
        self.categories = categories if categories else self._get_all_categories()

        self.data = []
        self.labels = []
        self._load_data()

    def _get_all_categories(self):
        """Get all category names based on directory structure."""
        return sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])

    def _load_data(self):
        """Load data into memory, with optional progress bar."""
        for label, category in enumerate(self.categories):
            category_dir = os.path.join(self.root_dir, category, self.split)
            if not os.path.exists(category_dir):
                continue

            files = [f for f in os.listdir(category_dir) if f.endswith('.off')]
            if self.show_progress:
                files = tqdm(files, desc=f"Loading {category} category")

            for file in files:
                file_path = os.path.join(category_dir, file)
                vertices, _ = self._read_off(file_path)
                if vertices is None:
                    continue  # Skip invalid files
                self.data.append(vertices)
                self.labels.append(label)

    def _read_off(self, file):
        """Read a .off file and return vertices and faces."""
        with open(file, 'r') as f:
            first_line = f.readline().strip()
            if 'OFF' != first_line:
                print(f"Skipping file: {file}, First line: {first_line}")
                return None, None
            try:
                n_verts, n_faces, _ = map(int, f.readline().strip().split(' '))
                verts = [list(map(float, f.readline().strip().split(' '))) for _ in range(n_verts)]
                faces = [list(map(int, f.readline().strip().split(' ')))[1:] for _ in range(n_faces)]
            except Exception as e:
                print(f"Error reading file: {file}, error: {e}")
                return None, None
            return np.array(verts), np.array(faces)


    def __len__(self):
        """Return the size of the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return the point cloud and label at the given index."""
        point_cloud = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            point_cloud = self.transform(point_cloud)
        
        return point_cloud, label

    def visualize(self, idx):
        """Visualize the point cloud at the given index."""
        point_cloud, label = self.__getitem__(idx)
        print(f"Class: {self.categories[label]}")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    selected_categories = ['flower_pot']    # ['bottle', 'cup', 'flower_pot', 'lamp']
    dataset = ModelNet40(categories=selected_categories)
    print(f"Dataset size: {len(dataset)}")
    point_cloud, label = dataset[0]
    dataset.visualize(0)