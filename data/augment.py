import numpy as np
import tensorflow as tf
from config.settings import config

class MNISTAugmenter:
    """Data augmentation for MNIST dataset"""
    def __init__(self):
        self.augmentation_layers = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(factor=0.05, fill_mode='constant'),
            tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1, fill_mode='constant'),
            tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='constant')
        ])
    
    def augment_batch(self, images, labels):
        """Apply augmentation to a batch of images"""
        augmented_images = self.augmentation_layers(images, training=True)
        return augmented_images, labels
    
    def augment_dataset(self, x_train, y_train):
        """Create augmented dataset"""
        # Create base dataset
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        
        # Apply augmentation
        augmented_dataset = dataset.map(
            lambda x, y: (self.augmentation_layers(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Combine with original data
        full_dataset = dataset.concatenate(augmented_dataset)
        return full_dataset
    
    def random_erasing(self, images, probability=0.5, sl=0.02, sh=0.4, r1=0.3):
        """Apply random erasing augmentation"""
        batch_size = images.shape[0]
        height, width, channels = images.shape[1:]
        
        augmented_images = np.copy(images)
        
        for i in range(batch_size):
            if np.random.rand() > probability:
                continue
                
            # Random erasing area
            area = height * width
            target_area = np.random.uniform(sl, sh) * area
            aspect_ratio = np.random.uniform(r1, 1/r1)
            
            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if w < width and h < height:
                x1 = np.random.randint(0, height - h)
                y1 = np.random.randint(0, width - w)
                
                # Erase with random value
                augmented_images[i, x1:x1+h, y1:y1+w, :] = np.random.rand(h, w, channels)
        
        return augmented_images
    
    def elastic_transform(self, images, alpha=34, sigma=4):
        """Apply elastic transformation"""
        from scipy.ndimage import gaussian_filter
        from scipy.ndimage import map_coordinates
        
        transformed_images = np.zeros_like(images)
        for i in range(images.shape[0]):
            image = images[i].squeeze()
            
            # Random displacement fields
            dx = gaussian_filter(
                (np.random.rand(*image.shape) * 2 - 1),
                sigma, mode="constant", cval=0) * alpha
            dy = gaussian_filter(
                (np.random.rand(*image.shape) * 2 - 1),
                sigma, mode="constant", cval=0) * alpha
            
            # Coordinate grid
            x, y = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing='ij')
            indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
            
            # Interpolate
            transformed = map_coordinates(image, indices, order=1).reshape(image.shape)
            transformed_images[i] = transformed[..., np.newaxis]
            
        return transformed_images