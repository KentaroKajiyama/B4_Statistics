# Re-create the histogram with 100 intervals, 1000 interval color breaks, white for the first bin, and a light grey background
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import geopandas as gpd
rcParams['lines.markersize'] = 1.0
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio','Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']


shp_path = "/Users/kajiyamakentarou/Keisu/卒論/最適配置/data/QGIS/東京２３区/メッシュあり東京２３区人口データ付き.shp"
shp_data = gpd.read_file(shp_path)
population_data = shp_data['population']
population_data_250 = np.tile(population_data/4,4)
corrected_average_population = population_data.mean()
corrected_average_population_250 = population_data.mean()/4
# Define the figure and background color
plt.figure(figsize=(10, 4))
plt.gca().set_facecolor('lightgrey')

# Create the histogram with 100 interval bins
n, bins_100, patches = plt.hist(population_data, bins=80, orientation='vertical', edgecolor='black',range=(0,8000))

# Set the colors for the bins. We need 7 colors for the intervals after the first 1000, plus white for 0-1000
# bin_colors_1000_intervals = ['#ffffff'] + list(plt.cm.Reds(np.linspace(0.1, 1, 7)))

# # Assign the bin colors, making sure the first bin (0-1000) is white
# for i, patch in enumerate(patches):
#     bin_idx = min(i // 10, 7)  # There are 74 bins, 7 color changes after the first 1000
#     plt.setp(patch, 'facecolor', bin_colors_1000_intervals[bin_idx])

n_intervals = 6  # Number of intervals for color breaks
bin_colors_2000_intervals = ['#ffffff'] + list(plt.cm.Reds(np.linspace(0, 1, n_intervals)))

# Calculate the index for color assignment for each bin
# Every 2000 units after the first 1000 units will change the color
for i, patch in enumerate(patches):
    if bins_100[i] < 1000:
        color_idx = 0
    else:
        color_idx = int((bins_100[i] - 1000) / 1000) + 1
        color_idx = min(color_idx, len(bin_colors_2000_intervals) - 1)  # Ensure index is within color list bounds
    plt.setp(patch, 'facecolor', bin_colors_2000_intervals[color_idx])


# Add a line for the average population
plt.axvline(x=corrected_average_population, color='blue', linestyle='-', linewidth=2, label='平均値')

# Set labels with "Population" and "Mean" to avoid encoding issues
plt.ylabel('Count', fontsize=14)
plt.xlabel('人口', fontsize=14)

# Add legend
plt.legend(loc='upper right')

# # Save the histogram with the specified properties
# histogram_100_color_corrected_path = '/mnt/data/histogram_population_100_intervals_color_corrected.png'
# plt.savefig(histogram_100_color_corrected_path)

# Show the plot
plt.show()

# Provide the path to the saved histogram with the specified properties
# histogram_100_color_corrected_path
