"""
LionSight2/ls2_emulator.py

Theodore Tasman
05-28-2025
PSU UAS

Emulator for the LionSight2 system, simulating its behavior and interactions.
"""

import random as rd
import sys
sys.path.append('../')  
from MAVez.Coordinate import Coordinate

class LionSight2:

    def __init__(self, stride, num_targets=4, crop_size=224):
        '''
        num_targets: number of targets to detect
        entry_coord: Coordinate of the entry point (must be on the left side of the runway)
        exit_coord: Coordinate of the exit point (must be on the right side of the runway)
        width: width of the runway in meters
        stride: stride of the scan in meters
        crop_size: size of the crop in pixels
        '''
        self.num_targets = num_targets
        self.net = None
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

    def detect(self, eps=150, confidence_threshold=0.5):
        '''
        Simulate the detection of targets along the runway.
        Returns a list of detected targets with their coordinates.
        '''
        
        if self.entry is None or self.exit is None or self.width is None:
            if self.logger:
                self.logger.error("[LionSight2] Entry, exit, or width not set. Please set the plan before detection.")
            return []
        
        entry_centered = self.entry.offset_coordinate(self.width / 2, self.cross_bearing)
        fake_targets = []

        for i in range(self.num_targets):
            current_position = entry_centered.offset_coordinate(self.length * (i + 1) / self.num_targets, self.bearing)
            current_position = current_position.offset_coordinate(rd.uniform(-self.width/1.5, self.width/1.5), self.cross_bearing)
            fake_targets.append(current_position)
        
        return fake_targets


def main():
    entry_coord = Coordinate(40.84181406869122,-77.6975985677159,0, use_int=False)
    exit_coord = Coordinate(40.8410979035657,-77.69898679735358,0, use_int=False)

    ls2 = LionSight2(
        entry_coord=entry_coord, 
        exit_coord=exit_coord, 
        width=30, 
        stride=5, 
        num_targets=4, 
        crop_size=224
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