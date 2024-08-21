"""
    ModelNet40 dataset class
"""

import os
import numpy as np
import open3d as o3d

from tqdm import tqdm
from torch.utils.data import Dataset

class ModelNet40(Dataset):
    def __init__(self, root_dir="./data/ModelNet40", split='train', categories=None, transform=None, show_progress=True):
        """
            root_dir: Path to the extracted ModelNet40 folder (default is "./data/ModelNet40").
            split: Dataset split, 'train' or 'test'.
            categories: List of categories to load, if None, load all categories.
            transform: Optional point cloud transformation function.
            show_progress: Whether to show a progress bar during data loading.
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

class PointNetModelNet40(ModelNet40):
    def __init__(self, root_dir="./data/ModelNet40", split='train', categories=None, 
                 transform=None, show_progress=True, npoints=1024, data_augmentation=True):
        super(PointNetModelNet40, self).__init__(root_dir, split, categories, transform, show_progress)
        self.npoints = npoints
        self.data_augmentation = data_augmentation

    def __getitem__(self, idx):
        point_cloud, label = super(PointNetModelNet40, self).__getitem__(idx)
        
        # Randomly sample npoints with replacement if necessary
        if point_cloud.shape[0] < self.npoints:
            choice = np.random.choice(point_cloud.shape[0], self.npoints, replace=True)
        else:
            choice = np.random.choice(point_cloud.shape[0], self.npoints, replace=False)
        
        point_cloud = point_cloud[choice, :]

        # Normalize the point cloud
        point_cloud = point_cloud - np.mean(point_cloud, axis=0)  # Center the point cloud
        point_cloud = point_cloud / np.max(np.linalg.norm(point_cloud, axis=1))  # Scale to unit sphere

        # Data augmentation
        if self.data_augmentation:
            point_cloud = self._augment(point_cloud)

        point_cloud = point_cloud.astype(np.float32)

        return point_cloud, label

    def _augment(self, point_cloud):
        # Apply random rotation around the Y-axis
        theta = np.random.uniform(0, np.pi * 2)
        rotation_matrix = np.array([[np.cos(theta), 0, np.sin(theta)],
                                    [0, 1, 0],
                                    [-np.sin(theta), 0, np.cos(theta)]])
        point_cloud = point_cloud.dot(rotation_matrix)

        # Apply random jitter
        point_cloud += np.random.normal(0, 0.02, size=point_cloud.shape)
        
        return point_cloud

if __name__ == '__main__':
    selected_categories = ['flower_pot']    
    dataset = PointNetModelNet40(categories=selected_categories)
    print(f"Dataset size: {len(dataset)}")
    point_cloud, label = dataset[0]
    dataset.visualize(0)