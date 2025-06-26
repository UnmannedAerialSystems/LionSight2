import sys
import os
sys.path.append(os.path.dirname(__file__) + "/..")
import torch
import cv2
import numpy as np
from sklearn.cluster import DBSCAN # type: ignore
from LionSight2.ls2_network import LS2Network
from PIL import Image
from tqdm import tqdm # type: ignore
from collections import defaultdict
from GPSLocator.geo_image import GeoImage
from MAVez.Coordinate import Coordinate
import math

# CAMERA PARAMETERS (Raspi AI Camera)
X_RES = 2028
Y_RES = 1520
SENSOR_WIDTH = 6.2868
SENSOR_HEIGHT = 4.712
FOCAL_LENGTH = 3.863

class LionSight2:

    def __init__(self, stride, num_targets=4, crop_size=224, logger=None):
        '''
        num_targets: number of targets to detect
        entry_coord: Coordinate of the entry point (must be on the left side of the runway)
        exit_coord: Coordinate of the exit point (must be on the right side of the runway)
        width: width of the runway in meters
        stride: stride of the scan in meters
        crop_size: size of the crop in pixels
        '''
        self.num_targets = num_targets
        self.logger = logger
        self.net = LS2Network(os.path.join(os.path.dirname(__file__), "ls2_2-0.pth"), logger=logger)
        self.images = None
        self.stride = stride
        self.next_position = None
        self.horizontal_shift = 0
        self.vertical_shift = 0
        self.crop_size = crop_size
        self.entry = None
        self.exit = None
        self.length = None
        self.width = None
        self.bearing = None
        self.cross_bearing = None
        self.detections = []
    

    def set_plan(self, entry_coord, exit_coord, width):
        '''
        Set the entry and exit coordinates and width of the runway.
        This will reset the next position and horizontal/vertical shifts.
        '''

        self.entry = entry_coord
        self.exit = exit_coord
        self.length = entry_coord.distance_to(exit_coord)
        self.width = width
        self.bearing = entry_coord.bearing_to(exit_coord)
        self.cross_bearing = (self.bearing + 90) % 360
        if self.logger:
            self.logger.info(f"[LionSight2] Plan set: Entry {entry_coord}, Exit {exit_coord}, Width {width}m, Bearing {self.bearing}°")

    def load_images(self, images_directory):
        '''
        Load images from the specified path.
        '''

        images = []

        with open(os.path.join(images_directory, "image_data.txt"), 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split(',')
                filename = parts[0]
                lat = float(parts[1])
                lon = float(parts[2])
                alt = float(parts[3])
                heading = float(parts[4])

                # create a geo_image object
                geo_img = GeoImage.GeoImage(image_path=os.path.join(images_directory, filename),
                                                latitude=lat, longitude=lon, altitude=alt,
                                                roll=0, pitch=0, heading=heading,
                                                res_x=X_RES, res_y=Y_RES,
                                                focal_length=FOCAL_LENGTH,
                                                sensor_width=SENSOR_WIDTH,
                                                sensor_height=SENSOR_HEIGHT)

                # verify the image was loaded correctly
                if geo_img.image is not None:
                    images.append(geo_img)
                else:
                    print(f"Error loading image: {filename}")

        self.images = images

    # DEPRECATED: Use detect instead
    def detect_orb(self):
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
    

    def next_stride(self):
        '''
        Calculate the next stride based on the current position and the bearing.
        '''
        # check if this is the first step
        if self.next_position is None:
            # set the next position to the entry coordinate
            self.next_position = self.entry
            self.horizontal_shift = 0
            self.vertical_shift = 0
            return 1

        # increment horizontal shift
        self.horizontal_shift += self.stride

        # check if we need to move to the next row
        if self.horizontal_shift >= self.length:
            self.horizontal_shift = 0
            self.vertical_shift += self.stride
        
        # check if we have done all rows
        if self.vertical_shift >= self.width:
            return -1
        
        # calculate the next position
        across = self.entry.offset_coordinate(self.horizontal_shift, self.bearing)
        down = across.offset_coordinate(self.vertical_shift, self.cross_bearing)
        self.next_position = down

        return 1


    def detect_dense(self):
        """
        Densely scan the entire stitched area, score with the network, and return top-K detections.
        """
        if self.logger:
            self.logger.info("[LionSight2] Starting dense detection...")

        while self.next_stride() != -1:

            # Check which image contains this point
            for geo_image in self.images:

                if self.next_position in geo_image:
  
                    # get the pixel within the image representing the center of the crop
                    img_x, img_y = geo_image.get_pixels(self.next_position)
                    img_h, img_w = geo_image.image.shape[:2]

                    # Check if the point is within the image bounds
                    half_crop = self.crop_size // 2
                    if (img_x - half_crop >= 0 and img_x + half_crop < img_w and 
                        img_y - half_crop >= 0 and img_y + half_crop < img_h):

                        # Crop the patch centered at (img_x, img_y)
                        crop = geo_image.image[img_y - half_crop:img_y + half_crop, img_x - half_crop:img_x + half_crop]
                        self.net.img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        probability = self.net.run_net()
                        if self.logger:
                            self.logger.info(f"[LionSight2] Detected at {self.next_position} with probability {probability:.4f}")
                        self.detections.append((Coordinate(self.next_position.lat * 1e7, 
                                                           self.next_position.lon * 1e7, 
                                                           alt=0,
                                                           use_int=True), 
                                                probability))
                        break  # Only use one image per crop

        if self.logger:
            self.logger.info(f"[LionSight2] Dense detection completed. Found {len(self.detections)} detections.")

    

    def cluster_and_average(self, top_k=4, eps=150, confidence_threshold=0.5):
        """
        Cluster detections using DBSCAN and return the average coordinates of the clusters.
        """

        detections = [detection for detection in self.detections if detection[2] > confidence_threshold]

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


    def detect(self, top_k=4):
        """
        Perform dense detection, clustering, and averaging of results.
        Returns the top-k averaged coordinates with their scores.
        """
        if self.logger:
            self.logger.info("[LionSight2] Starting detection process...")
        if self.entry is None or self.exit is None or self.width is None:
            if self.logger:
                self.logger.error("[LionSight2] Entry, exit, or width not set. Please set the plan before detection.")
            return []

        if self.logger:
            self.logger.info(f"[LionSight2] Detecting with stride {self.stride}m and crop size {self.crop_size}px")
        self.detect_dense_test(stride=self.stride, crop_size=self.crop_size)
        if self.logger:
            self.logger.info(f"[LionSight2] Getting best points...")
        best = self.get_best_points(num_points=top_k)
        if self.logger:
            self.logger.info(f"[LionSight2] Best points: {best}")
        return best


    def get_best_points(self, num_points=4):
        return sorted(self.detections, key=lambda x: x[1])[-num_points:]
    
def get_ls2(stride=5, num_targets=4, crop_size=224, logger=None, block_emulator=False):
    '''
    Create a LionSight2 object with the specified parameters.
    entry_coord: Coordinate of the entry point (must be on the left side of the runway)
    exit_coord: Coordinate of the exit point (must be on the right side of the runway)
    width: width of the runway in meters
    stride: stride of the scan in meters
    crop_size: size of the crop in pixels
    '''

    try:
        from picamera2 import Picamera2
        return LionSight2(stride, num_targets, crop_size, logger)
    except ImportError:
        if not block_emulator:
            from . import ls2_emulator
            if logger:
                logger.warning("[LionSight2] Picamera2 not available, using emulator.")
            return ls2_emulator.LionSight2(stride, num_targets, crop_size, logger)
        return LionSight2(stride, num_targets, crop_size, logger)




def main():

    import sys
    import os
    import time
    import matplotlib.pyplot as plt
    from GPSLocator import coord_generator

    # Start a timer
    print("\n==================================")
    start_time = time.time()

    entry_point = Coordinate(38.315509271316046, -76.55080562662074, 0, use_int=False)
    exit_point = Coordinate(38.3157407480423, -76.55194738196501, 0, use_int=False)

    lion_sight = LionSight2(
                            stride=5, 
                            num_targets=4, 
                            crop_size=224,
                            logger=None,  # Replace with your logger if needed                            
                            )

    lion_sight.set_plan(entry_coord=entry_point, exit_coord=exit_point, width=30)

    lion_sight.images = coord_generator.generate_geo_images()

    lion_sight.detect_dense()
    #best_points = lion_sight.cluster_and_average(top_k=8, eps=80, confidence_threshold=0.2)
    best_points = lion_sight.get_best_points()
    end_time = time.time()
    for point in best_points:
        print(f"Point: {point[0]}, {point[1]}, Score: {point[2]}")
    print(f"\nTime taken: {end_time - start_time} seconds\n==================================")


def main2():
    entry_coord = Coordinate(40.84181406869122,-77.6975985677159,0, use_int=False)
    exit_coord = Coordinate(40.8410979035657,-77.69898679735358,0, use_int=False)

    ls2 = get_ls2(
        stride=5, 
        num_targets=4, 
        crop_size=224,
        logger=None,  # Replace with your logger if needed
        block_emulator=True  # Set to True to force using the real camera
    )

    targets = ls2.detect()

    import matplotlib.pyplot as plt
    for target in targets:
        plt.scatter(target.lon, target.lat, label=f'Target at {target}')
    plt.scatter(entry_coord.lon, entry_coord.lat, color='green', label='Entry Point')
    plt.scatter(exit_coord.lon, exit_coord.lat, color='red', label='Exit Point')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Detected Targets along the Runway')
    plt.grid()
    plt.show()
    

if __name__ == "__main__":
    main()






        