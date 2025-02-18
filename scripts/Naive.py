import os
import pandas as pd

def count_images(dataset_path):
    """Counts the number of images in each class"""
    class_counts = {
        class_name: len(os.listdir(os.path.join(dataset_path, class_name)))
        for class_name in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, class_name))
    }
    return class_counts

def get_most_frequent_class(class_counts):
    """finds the most frequent class."""
    return max(class_counts, key=class_counts.get)

def naive_classifier(image_batch, most_frequent_class):
    """A naive classifier that always predicts the most frequent class."""
    return [most_frequent_class] * len(image_batch)

def main(dataset_path):
    """Main function to evaluate naive model"""
    class_counts = count_images(dataset_path)
    most_frequent_class = get_most_frequent_class(class_counts)
    
    print("Class Distribution:")
    df = pd.DataFrame(list(class_counts.items()), columns=["Class Name", "Image Count"])
    print(df)
    
    print(f"\nMost Frequent Class: {most_frequent_class}")
    
    total_images = sum(class_counts.values())
    validation_images = int(0.2 * total_images)
    correct_predictions = int(0.2 * class_counts[most_frequent_class])
    naive_accuracy = correct_predictions / validation_images if validation_images > 0 else 0
    
    print(f"Naive Model Accuracy: {naive_accuracy:.2%}")
    
    return most_frequent_class, naive_accuracy

if __name__ == "__main__":
    main("banana_tree")
