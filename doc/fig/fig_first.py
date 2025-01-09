import matplotlib.pyplot as plt
from tueplots import bundles
from tueplots.constants.color import rgb

# Update global settings with JMLR base and your customizations
plt.rcParams.update(bundles.jmlr2001())

# Additional customizations for Times New Roman and 12 pt font
plt.rcParams.update({"font.size": 12})

# Example plot with updated settings
x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]

plt.plot(x, y)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("JMLR Styled Plot with Custom Adjustments")
plt.savefig("jmlr_custom_plot.pdf")
plt.show()
