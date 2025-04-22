import torch
import cv2
import numpy as np
from sklearn.cluster import DBSCAN # type: ignore
from ls2_cluster_orb import ClusterORB
from ls2_network import LS2Network
import os
from PIL import Image
from tqdm import tqdm # type: ignore
from collections import defaultdict
from GPSLocator import geo_image
from MAVez import Coordinate
import math

class LionSight2:

    def __init__(self, num_targets, net, orb):
        self.num_targets = num_targets
        self.orb = orb
        self.net = net
        self.images = None
        self.true_points = None
    

    def load_images(self, images_directory):
        '''
        Load images from the specified path.
        '''
        image_names = [filename for filename in os.listdir(images_directory) if filename.endswith(".png")]
        images = []
        for filename in image_names:
            image = cv2.imread(os.path.join(images_directory, filename))
            if image is not None:
                image_coords = filename.split('_')[1]
                image_coords = tuple(image_coords.strip('()').split(','))
                image_coords = (int(image_coords[0]), int(image_coords[1]))
                image = (image, image_coords)
                images.append(image)
            else:
                print(f"Error loading image: {filename}")
        self.images = images

    
    def detect(self):
        '''
        Detect objects in the images using the neural network and ORB feature detector.
        '''

        # Process 
        cluster_info, cluster_centers = self.orb.process_images(self.images)

        # Initialize a list to store the results
        results = []

        # Determine which photos contain which clusters
        for center in cluster_centers:
            # Get the coordinates of the cluster center
            x, y = int(center[0]), int(center[1])

            # track the number of images containing the center and positive detections
            num_images = 0
            prediction_sum = 0

            # Check which photo contains the cluster center
            for i, image in enumerate(self.images):

                left = image[1][0]
                right = image[1][0] + image[0].shape[1]
                top = image[1][1]
                bottom = image[1][1] + image[0].shape[0]

                

                if left <= x <= right and top <= y <= bottom:
                    print(f"{left}\t<= {x}\t<=\t{right} and {top}\t<=\t{y} <=\t{bottom}")
                    num_images += 1

                    image_x = x - left
                    image_y = y - top

                    self.net.crop_to_poi(image[0], (image_x, image_y), 224)
                    output = self.net.run_net()

                    prediction_sum += output
                
            # Calculate the average prediction
            if num_images > 0:
                avg_prediction = prediction_sum / num_images
            else:
                avg_prediction = 0

            # Append the result to the list
            results.append((x, y, avg_prediction))

        return results, cluster_centers
    

    def create_scan(self, stride, top_left_coord, top_right_coord, bottom_left_coord):
        """
        Create a scan of the area defined by the coordinates.
        """
        # Calculate the heading
        lon1, lat1 = top_left_coord.lon, top_left_coord.lat
        lon2, lat2 = bottom_left_coord.lon, bottom_left_coord.lat
        lon1_rad = math.radians(lon1)
        lat1_rad = math.radians(lat1)
        lon2_rad = math.radians(lon2)
        lat2_rad = math.radians(lat2)

        delta_lon = lon2_rad - lon1_rad

        x = math.cos(lat2_rad) * math.sin(delta_lon)
        y = math.cos(lat1_rad) * math.sin(lat2_rad) - (math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon))
        heading = math.atan2(x, y)
        heading = math.degrees(heading)
        return heading


    def detect_dense(self, stride=32, crop_size=224):
        """
        Densely scan the entire stitched area, score with the network, and return top-K detections.
        """
        # Determine full bounds of stitched image
        min_x = min(img[1][0] for img in self.images)
        min_y = min(img[1][1] for img in self.images)
        max_x = max(img[1][0] + img[0].shape[1] for img in self.images)
        max_y = max(img[1][1] + img[0].shape[0] for img in self.images)

        stitched_width = max_x - min_x
        stitched_height = max_y - min_y

        print(f"Stitched image size: {stitched_width} x {stitched_height}")

        results = []

        total_positions = ((max_y - min_y - crop_size) // stride) * ((max_x - min_x - crop_size) // stride)
        progress_bar = tqdm(total=total_positions, desc="Dense CNN Scan")

        for y in range(min_y, max_y - crop_size, stride):
            for x in range(min_x, max_x - crop_size, stride):

                # Find which image contains this patch
                for img, origin in self.images:
                    img_x, img_y = origin
                    img_h, img_w = img.shape[:2]

                    # Does this image cover the crop?
                    if (img_x <= x < img_x + img_w - crop_size and
                        img_y <= y < img_y + img_h - crop_size):

                        rel_x = x - img_x
                        rel_y = y - img_y

                        # Crop the patch
                        crop = img[rel_y:rel_y+crop_size, rel_x:rel_x+crop_size]
                        self.net.img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        probability = self.net.run_net()
                        
                        # # if this should contain a true point
                        # for true_point in self.true_points:
                        #     true_x, true_y = true_point
                        #     if (x <= true_x < x + crop_size and
                        #         y <= true_y < y + crop_size):
                                
                        #         # display the crop with score
                        #         cv2.imshow(f"{score}", crop)
                        #         cv2.waitKey(0)
                        #         cv2.destroyAllWindows()

                        square_center_x = x + crop_size // 2
                        square_center_y = y + crop_size // 2
                        results.append((square_center_x, square_center_y, probability))
                        break  # Only use one image per crop

                progress_bar.update(1)

        return results
    

    def cluster_and_average(self, detections, top_k=4, eps=150, confidence_threshold=0.5):
        """
        Cluster detections using DBSCAN and return the average coordinates of the clusters.
        """
        detections = [detection for detection in detections if detection[2] > confidence_threshold]

        if not detections:
            return []

        coords = np.array([[x, y] for x, y, _ in detections])
        scores = np.array([score for _, _, score in detections])

        db = DBSCAN(eps=eps, min_samples=2).fit(coords)
        labels = db.labels_
        # Display DBSCAN results
        print("DBSCAN Results:")
        for label in set(labels):
            if label == -1:
                print(f"Noise points: {sum(labels == -1)}")
            else:
                print(f"Cluster {label}: {sum(labels == label)} points")

        clusters = defaultdict(list)

        for label, (x, y, score) in zip(labels, detections):
            if label == -1:
                continue  # Ignore noise points
            clusters[label].append((x, y, score))
        
        cluster_averages = []

        for cluster_points in clusters.values():
            cluster_points = sorted(cluster_points, key=lambda x: x[2], reverse=True)
            x, y, scores = zip(*cluster_points)
            avg_x = np.mean(x)
            avg_y = np.mean(y)
            avg_score = np.mean(scores)

            cluster_averages.append((avg_x, avg_y, avg_score))

        return sorted(cluster_averages, key=lambda x: x[2], reverse=True)[:top_k]



def main():

    import sys
    import os
    from detect_zone_generator import Runway
    import time
    import matplotlib.pyplot as plt

    runway = Runway('./runway_smaller.png', height=800, y_offset=400, ratio=8, num_targets=8)
    runway.assign_targets()
    runway.apply_motion_blur()
    photos = runway.generate_photos(20)

    # Create a directory to save photos if it doesn't exist
    output_dir = "test_photos"
    os.makedirs(output_dir, exist_ok=True)
    # Save the generated photos to the directory
    for i, photo in enumerate(photos):
        photo_to_save = (photo[0] * 255).astype(np.uint8) if photo[0].dtype != np.uint8 else photo[0]
        photo_path = os.path.join(output_dir, f"photo_{photo[1][0]},{photo[1][1]}_.png")
        cv2.imwrite(photo_path, photo_to_save)

    # Start a timer
    start_time = time.time()
    orb = ClusterORB(n_clusters=20, n_features=1024)
    net = LS2Network("ls2_2-0.pth")
    lion_sight = LionSight2(num_targets=8, net=net, orb=orb)
    lion_sight.true_points = runway.points
    lion_sight.load_images(output_dir)
    results = lion_sight.detect_dense()
    best_points = lion_sight.cluster_and_average(results, top_k=8, eps=80)

    end_time = time.time()

    runway_img = runway.runway.copy()

    # Plot the runway image
    plt.imshow(cv2.cvtColor(runway_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected Points (in {end_time - start_time:.2f} seconds)")

    # Plot true target coordinates
    for i, target in enumerate(runway.points):
        plt.scatter(target[0], target[1], c='black', marker='x', label=f"Target {i+1}")

    # Plot best cluster centers
    best_points = np.array(best_points)
    plt.scatter(best_points[:, 0], best_points[:, 1], c='red', marker='+', s=100, label="Best Points")
    
    plt.legend()
    plt.axis("off")
    plt.show()


    

if __name__ == "__main__":
    ls2 = LionSight2(num_targets=8, net=None, orb=None)

    top_left = Coordinate(0, 0)
    top_right = Coordinate(0, 100)
    bottom_left = Coordinate(100, 0)
    bottom_right = Coordinate(100, 100)

    ls2.create_scan()






        