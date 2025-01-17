# import numpy as np
# import matplotlib.pyplot as plt
#
# # --------------------
# # Example data
# # --------------------
# plt.rcParams.update({'font.size': 40})
# tasks = ["Manipulation", "Locomotion", "Loco-manip", "Multi-robot"]
#
# # Success rates (fractions, e.g., 0.90 = 90%)
# success_rates_current = [0.80, 0.90, 0.70, 0.70]
# success_rates_baseline = [0.60, 0.80, 0.20, 0.40]
#
# # Performance error (cm) for successful trials
# perf_err_current_mean = [6.4, 4.3, 8.6, 7.9]
# perf_err_current_std  = [1.2, 0.0, 2.3, 1.8]
#
# perf_err_baseline_mean = [8.9, 6.2, 14.2, 11.9]
# perf_err_baseline_std  = [2.6, 1.5, 4.5, 3.8]
#
# # Convert success rates to percentages for plotting
# success_rates_current_percent = [sr * 100 for sr in success_rates_current]
# success_rates_baseline_percent = [sr * 100 for sr in success_rates_baseline]
#
# # X positions for each task
# x = np.arange(len(tasks))  # [0,1,2,3]
# bar_width = 0.35
#
# # --------------------
# # Plotting
# # --------------------
# fig, ax1 = plt.subplots(figsize=(8, 5))
#
# # --- Left Y-Axis: Success Rate ---
# # Current Method bars
# rects_current = ax1.bar(
#     x - bar_width/2,
#     success_rates_current_percent,
#     bar_width,
#     color='pink',
#     alpha=0.7,
#     label='Success Rate (Current)'
# )
#
# # Baseline bars
# rects_baseline = ax1.bar(
#     x + bar_width/2,
#     success_rates_baseline_percent,
#     bar_width,
#     color='skyblue',
#     alpha=0.7,
#     label='Success Rate (Baseline)'
# )
#
# ax1.set_ylabel('Success Rate (%)')
# ax1.set_ylim([0, 100])  # 0–100% for percentages
# ax1.set_xticks(x)
# ax1.set_xticklabels(tasks)
#
# # --- Right Y-Axis: Performance Error in cm ---
# ax2 = ax1.twinx()
#
# # Current Method error bars (no connecting line)
# ax2.errorbar(
#     x - bar_width/2,
#     perf_err_current_mean,
#     yerr=perf_err_current_std,
#     fmt='o',           # plot markers only
#     linestyle='',      # no line connecting markers
#     color='red',
#     capsize=5,
#     elinewidth=5,      # thicker vertical error lines
#     capthick=5,        # thicker cap lines
#     markersize=20,       # larger markers
#     markeredgewidth=2,   # thicker marker edge
#     label='Error of failed trials (Current)'
# )
#
# # Baseline error bars (no connecting line)
# ax2.errorbar(
#     x + bar_width/2,
#     perf_err_baseline_mean,
#     yerr=perf_err_baseline_std,
#     fmt='o',
#     linestyle='',
#     color='blue',
#     capsize=5,
#     elinewidth=5,      # thicker vertical error lines
#     capthick=5,        # thicker cap lines
#     markersize=20,       # larger markers
#     markeredgewidth=2,   # thicker marker edge
#     label='Error of failed trials (Baseline)',
# )
#
# ax2.set_ylabel('Performance Error (cm)')
# # Optionally set limits if desired
# ax2.set_ylim([0, max(
#     max(np.array(perf_err_current_mean) + np.array(perf_err_current_std)),
#     max(np.array(perf_err_baseline_mean) + np.array(perf_err_baseline_std))
# ) + 8.0])
#
# # --- Title and Legend ---
# # plt.title('Success Rate and Performance Error by Task', fontsize=18)
#
# # Combine legends from both axes
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=28)
#
#
#
# plt.tight_layout()
# plt.show()


##################################################################################

# import numpy as np
# import matplotlib.pyplot as plt
#
# plt.rcParams.update({'font.size': 30})
#
# # Data: means and standard deviations
# grain_mean = [4.36, 3.68, 1.85, 1.91, 1.66]
# grain_std  = [2.32, 1.67, 0.76, 0.88, 0.72]
#
# current_mean = [2.21, 1.55, 1.62, 1.33, 1.24]
# current_std  = [0.91, 0.77, 0.79, 0.55, 0.42]
#
# # Depth values
# depths = [0, 2, 4, 6, 8]
#
# # Create an array of indices for the x-axis positions
# x = np.arange(len(depths))  # [0, 1, 2, 3, 4]
#
# # Width of each bar
# width = 0.35
#
# # Create the figure and axes
# fig, ax = plt.subplots(figsize=(8, 6))
#
# # Plot bars for GRAIN
# bars_grain = ax.bar(
#     x - width/2,       # Shift left by width/2 so that we can place Current bars next to it
#     grain_mean,        # Mean values
#     width,             # Bar width
#     yerr=grain_std,    # Error bars
#     capsize=5,         # Add caps to error bars
#     color='skyblue',
#     label='GRAIN'      # Legend label
# )
#
# # Plot bars for Current
# bars_current = ax.bar(
#     x + width/2,       # Shift right by width/2
#     current_mean,       # Mean values
#     width,             # Bar width
#     yerr=current_std,  # Error bars
#     capsize=5,         # Add caps to error bars
#     color='pink',
#     label='DiffusiveGRAIN'    # Legend label
# )
#
# # Set x-axis ticks and labels
# ax.set_xticks(x)
# ax.set_xticklabels(depths)
#
# # Labels and title
# ax.set_xlabel('Distance (cm)')
# ax.set_ylabel('Prediction Error (cm)')
#
# # Add grid and legend
# # ax.grid(True, linestyle='--', alpha=0.7)
# ax.legend()
#
# plt.tight_layout()
# plt.show()

###
import numpy as np
import matplotlib.pyplot as plt

# Data
means = [3.35, 3.13, 3.03, 2.97, 2.96]
stds = [0.74, 0.64, 0.46, 0.51, 0.44]
x_labels = [0, 2, 4, 6, 8]

# Create a list/array of x positions for the bars
x_positions = np.arange(len(means))

# Create the figure and axis
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the bars with error bars
ax.bar(
    x_positions,        # x-coordinates of the bars
    means,              # heights of the bars
    yerr=stds,          # error bar lengths
    capsize=5,          # length of the error bar caps
    color='skyblue',    # color of bars
    edgecolor='k',      # bar border color
    alpha=0.8
)

# Set x-axis tick positions and labels
ax.set_xticks(x_positions)
ax.set_xticklabels(x_labels)

# Set y-axis range
ax.set_ylim([0, 7])

# Add labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Mean Value')
ax.set_title('Histogram with Error Bars')

# Optional: add grid
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()

###
#######################################################################

# import numpy as np
# import matplotlib.pyplot as plt
# plt.rcParams.update({'font.size': 36})
# # --- Data --------------------------------------------------------------------
#
# labels = ["MF", "LT", "RT", "FE", "LFE", "RFE"]
#
# # 1) Fore-aft displacement (cm)
# mean_fore_aft = np.array([2.61, 0.62, 0.59, 0.26, 0.16, 0.15])
# std_fore_aft  = np.array([0.51, 0.24, 0.27, 0.09, 0.02, 0.03])
#
# # 2) Horizontal displacement (cm)
# mean_horizontal = np.array([-0.06, 0.71, -0.76, 0.02, 0.14, -0.15])
# std_horizontal  = np.array([0.24, 0.26, 0.18, 0.05, 0.03, 0.03])
#
# # 3) Orientation changes (rads)
# mean_orientation = np.array([-0.02, 0.14, -0.14, 0.00, 0.01, 0.01])
# std_orientation  = np.array([0.04, 0.03, 0.03, 0.01, 0.01, 0.01])
#
# # --- Plotting -----------------------------------------------------------------
#
# # Create subplots: 1 row, 3 columns
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#
# # Common x positions for bars
# x = np.arange(len(labels))
#
# # --- Subplot 1: Fore-aft displacement ---
# axes[0].bar(x,
#             mean_fore_aft,
#             yerr=std_fore_aft,
#             capsize=5,
#             color="skyblue",
#             edgecolor="black")
# axes[0].set_xticks(x)
# axes[0].set_xticklabels(labels)
# axes[0].set_ylabel("fore-aft displacement (cm)")
# # axes[0].set_title("Fore-Aft")
#
# # --- Subplot 2: Horizontal displacement ---
# axes[1].bar(x,
#             mean_horizontal,
#             yerr=std_horizontal,
#             capsize=5,
#             color="lightgreen",
#             edgecolor="black")
# axes[1].set_xticks(x)
# axes[1].set_xticklabels(labels)
# axes[1].set_ylabel("horizontal displacement (cm)")
# # axes[1].set_title("Horizontal")
#
# # --- Subplot 3: Orientation changes ---
# axes[2].bar(x,
#             mean_orientation,
#             yerr=std_orientation,
#             capsize=5,
#             color="salmon",
#             edgecolor="black")
# axes[2].set_xticks(x)
# axes[2].set_xticklabels(labels)
# axes[2].set_ylabel("orientation changes (rads)")
# # axes[2].set_title("Orientation")
#
# # Adjust layout so labels don’t overlap
# plt.tight_layout()
# plt.show()


#######################################################################


# import numpy as np
# import matplotlib.pyplot as plt
# plt.rcParams.update({'font.size': 36})
# # ---------------------------------
# # Data
# # ---------------------------------
# # X-axis labels (can be integers or strings)
# x_labels = [0, 2, 4, 6, 8]
#
# # Fore-aft axis displacement data (y-values)
# # with x-axis = Horizontal distance
# mean_fore_aft = np.array([7.15, 6.81, 6.25, 6.34, 6.41])
# std_fore_aft  = np.array([0.81, 0.42, 0.38, 0.34, 0.36])
#
# # Horizontal axis displacement data (y-values)
# # with x-axis = Fore-aft distance
# mean_horizontal = np.array([4.32, 2.81, 5.81, 6.21, 6.33])
# std_horizontal  = np.array([1.43, 0.51, 0.45, 0.40, 0.26])
#
# # ---------------------------------
# # Plotting
# # ---------------------------------
# # Create a figure with 2 subplots, side by side
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#
# # For bar positioning on the x-axis, convert x_labels into an array
# x_positions = np.arange(len(x_labels))
#
# # -----------------------
# # Subplot 1: Fore-aft axis displacement
# # -----------------------
# axes[0].bar(
#     x_positions,
#     mean_fore_aft,
#     yerr=std_fore_aft,
#     capsize=5,         # size of the error-bar caps
#     color="skyblue",
#     edgecolor="black"
# )
# axes[0].set_xticks(x_positions)
# axes[0].set_xticklabels(x_labels)
# axes[0].set_xlabel("Horizontal distance (cm)")
# axes[0].set_ylabel("Fore-aft Displacement (cm)")
# # axes[0].set_title("Obstacle Displacement: Fore-Aft Axis")
#
# # -----------------------
# # Subplot 2: Horizontal axis displacement
# # -----------------------
# axes[1].bar(
#     x_positions,
#     mean_horizontal,
#     yerr=std_horizontal,
#     capsize=5,
#     color="lightgreen",
#     edgecolor="black"
# )
# axes[1].set_xticks(x_positions)
# axes[1].set_xticklabels(x_labels)
# axes[1].set_xlabel("Fore-aft distance (cm)")
# axes[1].set_ylabel("Fore-aft Displacement (cm)")
# # axes[1].set_title("Obstacle Displacement: Horizontal Axis")
#
# # Add dashed line at y = 6.34
# axes[1].axhline(
#     y=6.34,
#     color='red',
#     linestyle='--',
#     linewidth=3.0,
#     label='y = 6.34'
# )
# axes[0].axhline(
#     y=6.34,
#     color='red',
#     linestyle='--',
#     linewidth=3.0,
#     label='y = 6.34'
# )
#
# axes[0].set_ylim([0, 8])
# axes[1].set_ylim([0, 8])
#
#
#
# # Make layout neat
# plt.tight_layout()
#
# # Display the plots
# plt.show()

####################################################################################

# import numpy as np
# import matplotlib.pyplot as plt
#
# plt.rcParams.update({'font.size': 26})
# # 1) Define the x-axis labels
# x_labels = ["Both", "Left", "Right"]
# x = np.arange(len(x_labels))  # [0, 1, 2]
#
# # 2) Define data for the top row (Fore-aft displacement)
# #    Each entry corresponds to a subplot in row 1, columns 1-3
# means_top = [
#     [6.5, 1.9, 1.8],   # (1,1)
#     [5.0, 4.2, 1.6],   # (1,2)
#     [5.2, 5.1, 0.0]    # (1,3)
# ]
# stds_top = [
#     [0.95, 0.32, 0.35],
#     [0.83, 0.68, 0.29],
#     [0.77, 0.85, 0.00]
# ]
#
# # 3) Define data for the bottom row (Lateral displacement)
# #    Each entry corresponds to a subplot in row 2, columns 1-3
# means_bottom = [
#     [ 0.08, -1.96,  2.02],  # (2,1)
#     [-0.65, -1.90,  1.05],  # (2,2)
#     [-2.60, -2.50,  0.00]   # (2,3)
# ]
# stds_bottom = [
#     [0.16, 0.26, 0.16],
#     [0.17, 0.19, 0.21],
#     [0.67, 0.70, 0.00]
# ]
#
# # 4) Create the figure and the 2x3 array of axes
# fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6), sharex=False)
#
# # 5) Plot the top row (Fore-aft displacement)
# for col in range(3):
#     ax = axes[0, col]
#     ax.bar(x, means_top[col], yerr=stds_top[col], capsize=5, color='C0', alpha=0.7)
#     ax.set_xticks(x)
#     ax.set_xticklabels(x_labels)
#     ax.set_ylabel("Fore-aft displacement (cm)")
#     ax.set_title(f"{col*15} degrees")
#     ax.set_ylim([0, 8])
#
# # 6) Plot the bottom row (Lateral displacement)
# for col in range(3):
#     ax = axes[1, col]
#     ax.bar(x, means_bottom[col], yerr=stds_bottom[col], capsize=5, color='C1', alpha=0.7)
#     ax.set_xticks(x)
#     ax.set_xticklabels(x_labels)
#     ax.set_ylabel("Lateral displacement (cm)")
#     ax.set_ylim([-4, 4])
#
# plt.tight_layout()
# plt.show()
