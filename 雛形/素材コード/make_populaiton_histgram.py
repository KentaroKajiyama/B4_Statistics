# Re-create the histogram with 100 intervals, 1000 interval color breaks, white for the first bin, and a light grey background
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import geopandas as gpd
rcParams['lines.markersize'] = 1.0
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio','Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']


shp_path = "/Users/kajiyamakentarou/Keisu/卒論/最適配置/実験ソース/実データ/東京２３区＿ユークリッド距離/ソースコード/2000年東京人口メッシュ.shp"
shp_data = gpd.read_file(shp_path)
population_data = shp_data['population']
corrected_average_population = population_data.mean()
print(corrected_average_population)
# # Define the figure and background color
# plt.figure(figsize=(10, 12))
# plt.gca().set_facecolor('lightgrey')

# # Create the histogram with 100 interval bins
# n, bins_100, patches = plt.hist(population_data, bins=100, orientation='horizontal', edgecolor='black')

# # Set the colors for the bins. We need 7 colors for the intervals after the first 1000, plus white for 0-1000
# bin_colors_1000_intervals = ['#ffffff'] + list(plt.cm.Reds(np.linspace(0.1, 1, 7)))

# # Assign the bin colors, making sure the first bin (0-1000) is white
# for i, patch in enumerate(patches):
#     bin_idx = min(i // 10, 7)  # There are 74 bins, 7 color changes after the first 1000
#     plt.setp(patch, 'facecolor', bin_colors_1000_intervals[bin_idx])

# # Add a line for the average population
# plt.axhline(y=corrected_average_population, color='blue', linestyle='-', linewidth=2, label='平均値')

# # Set labels with "Population" and "Mean" to avoid encoding issues
# plt.xlabel('Count', fontsize=14)
# plt.ylabel('人口', fontsize=14)

# # Add legend
# plt.legend(loc='upper right')

# # # Save the histogram with the specified properties
# # histogram_100_color_corrected_path = '/mnt/data/histogram_population_100_intervals_color_corrected.png'
# # plt.savefig(histogram_100_color_corrected_path)

# # Show the plot
# plt.show()

# Provide the path to the saved histogram with the specified properties
# histogram_100_color_corrected_path
