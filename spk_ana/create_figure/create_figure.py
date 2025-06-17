import matplotlib.pyplot as plt


class CreatFigure:
    def __init__(self, width_mm, aspect_ratio=3 / 4, is_remove_ur_edge=1):
        """
        Initialize figure settings
        :param width_mm: Figure width in millimeters
        :param aspect_ratio: Width to height ratio, default is 3/4
        :param is_remove_ur_edge: Whether to remove top and right spines, default is 1 (remove)
        """
        self.width_mm = width_mm
        self.aspect_ratio = aspect_ratio
        self.is_remove_ur_edge = is_remove_ur_edge
        self.fig, self.ax = self._create_figure()
        self.set_font_and_ax()

    def _create_figure(self):
        """
        Create figure and set basic style
        :return: fig, ax
        """
        # Convert width from millimeters to inches
        width_inch = self.width_mm / 25.4
        height_inch = width_inch / self.aspect_ratio

        # Create figure and subplot
        fig, ax = plt.subplots(figsize=(width_inch, height_inch))
        return fig, ax

    def set_font_and_ax(self, title_font_size=7):
        """
        Set unified font sizes, axis ticks, title size, and line widths
        """
        # Set title font and size
        self.ax.set_title(self.ax.get_title(), fontname='Arial', fontsize=title_font_size)

        # Set x and y axis labels font and size
        self.ax.set_xlabel(self.ax.get_xlabel(), fontname='Arial', fontsize=7)
        self.ax.set_ylabel(self.ax.get_ylabel(), fontname='Arial', fontsize=7)

        # Set tick parameters for both axes
        self.ax.tick_params(axis='both', which='major', labelsize=5, width=0.5, length=1.5, pad=1)

        # Set spine line widths
        for spine in self.ax.spines.values():
            spine.set_linewidth(0.5)  # 0.5pt line width

        # Optionally remove top and right spines
        if self.is_remove_ur_edge:
            self.ax.spines['top'].set_visible(False)
            self.ax.spines['right'].set_visible(False)

            # If legend exists, set its font to Arial with size 5pt
        legend = self.ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(5)  # Font size 5pt
                text.set_fontfamily('Arial')  # Font family Arial

    def set_colorbar(self, cbar):
        """
        Configure colorbar properties.

        Parameters:
        cbar: matplotlib.colorbar.Colorbar  object
        """
        # Set colorbar label font size
        if hasattr(cbar.ax.yaxis, 'label'):
            cbar.ax.yaxis.label.set_fontsize(7)  # Directly set label font size

        # Set tick label font size
        cbar.ax.tick_params(labelsize=5)

        # Set colorbar outline width
        cbar.outline.set_linewidth(0.5)

        # Set tick width and length
        cbar.ax.tick_params(width=0.5, length=1.5)

        # Adjust distance between colorbar and tick labels
        cbar.ax.yaxis.set_tick_params(pad=1)  # Adjust pad value as needed

    def set_title(self, title):
        """
        Set figure title
        :param title: Title text
        """
        self.ax.set_title(title, fontname='Arial', fontsize=7)

    def set_labels(self, xlabel, ylabel):
        """
        Set x and y axis labels
        :param xlabel: x-axis label text
        :param ylabel: y-axis label text
        """
        self.ax.set_xlabel(xlabel, fontname='Arial', fontsize=7)
        self.ax.set_ylabel(ylabel, fontname='Arial', fontsize=7)

    def show(self):
        """
        Display the figure
        """
        plt.show()

    def savepng(self, filename, dpi=1200):
        """
        Save figure as PNG file with no white margins.

        Parameters:
        filename: Filename (including path and extension, e.g. 'output.png')
        dpi: Resolution, default is 1200
        """
        self.fig.savefig(
            f"{filename}.png",
            dpi=dpi,
            bbox_inches='tight',  # Remove white margins
            pad_inches=0  # Further remove white margins
        )
        print(f"{filename}.png was saved.")

    def savepdf(self, filename):
        """
        Save figure as PDF file with no white margins.

        Parameters:
        filename: Filename (including path and extension, e.g. 'output.pdf')
        """
        self.fig.savefig(
            f"{filename}.pdf",
            bbox_inches='tight',  # Remove white margins
            pad_inches=0  # Further remove white margins
        )
        print(f"{filename}.pdf was saved.")


# Example usage
if __name__ == "__main__":
    # Create a figure with 60mm width
    figure = CreatFigure(width_mm=60)

    # Set title and axis labels
    figure.set_title("Sample  Title")
    figure.set_labels("X  Axis Label", "Y Axis Label")

    # Display figure
    figure.show()

    # Save figure
    figure.savepng("output")