import matplotlib.pyplot as plt
import numpy as np

class ErrorTracker:
    def __init__(self):
        self.errors = []
        self.time_stamps = []

    def record_error(self, state, target, time):
        self.errors.append(np.sqrt((target.x - state.x)**2 + (target.y - target.y)**2))
        self.time_stamps.append(time)

    def plot_errors(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.time_stamps, self.errors, marker='o')
        plt.title('State Error Over Time')
        plt.xlabel('Time')
        plt.ylabel('Error')
        plt.grid(True)
        plt.show()
