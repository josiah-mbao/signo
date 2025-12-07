import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import argparse

class ASLDataProcessor:
    def __init__(self, data_path='data/asl_alphabet_train/asl_alphabet_train'):
        self.data_path = data_path
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )

    def extract_landmarks(self, image):
        """Extract 63 landmark features (21 points x x,y,z)"""
        results = self.mp_hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            landmarks = []
            for lm in results.multi_hand_landmarks[0].landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            return landmarks
        return None

    def process_dataset(self, samples_per_class=1000, output_file='asl_dataset.csv'):
        """Process images and create CSV dataset"""
        data = []

        # ASL alphabet letter classes
        classes = [chr(i) for i in range(ord('A'), ord('Z')+1)]

        for class_name in tqdm(classes, desc="Processing classes"):
            class_dir = os.path.join(self.data_path, class_name)

            if not os.path.isdir(class_dir):
                print(f"Directory not found: {class_dir}")
                continue

            # Get image files
            images = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]
            images = images[:samples_per_class]  # Limit samples

            processed = 0
            for img_file in tqdm(images, desc=f"Processing {class_name}", leave=False):
                img_path = os.path.join(class_dir, img_file)
                image = cv2.imread(img_path)

                if image is None:
                    continue

                landmarks = self.extract_landmarks(image)

                if landmarks:
                    data.append([class_name] + landmarks)
                    processed += 1

            print(f"{class_name}: {processed}/{len(images)} images processed")

        # Save as CSV (matches your existing format)
        columns = ['label'] + [f'landmark_{i}_{coord}' for i in range(21) for coord in ['x', 'y', 'z']]
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(output_file, index=False)
        print(f"Saved {len(data)} samples to {output_file}")

        return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process ASL dataset to landmarks')
    parser.add_argument('--data_path', default='data/asl_alphabet_train/asl_alphabet_train', help='Path to training data')
    parser.add_argument('--samples', type=int, default=800, help='Samples per class')
    parser.add_argument('--output', default='asl_dataset.csv', help='Output CSV file')

    args = parser.parse_args()

    processor = ASLDataProcessor(args.data_path)
    df = processor.process_dataset(args.samples, args.output)
