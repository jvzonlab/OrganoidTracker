from matplotlib.backend_bases import KeyEvent
from matplotlib.figure import Figure
from imaging import Particle, Experiment
from imaging.visualizer import Visualizer, activate
from linking_analysis import mother_finder
from typing import List
import matplotlib.pyplot as plt


class CellDivisionVisualizer(Visualizer):
    """Shows cells that are about to divide.
    Use the left/right arrow keys to move to the next cell division.
    Press M to exit this view."""

    _mother_index: int
    _all_mothers = List[Particle]

    def __init__(self, experiment: Experiment, figure: Figure, mother: Particle):
        super().__init__(experiment, figure)
        self._all_mothers = None

        graph = experiment.particle_links()
        if graph is not None:
            self._all_mothers = list(mother_finder.find_mothers(graph))
            self._all_mothers.sort(key=lambda particle: particle.frame_number())

        self._mother_index = self._find_closest_mother_index(mother)

    def _find_closest_mother_index(self, particle: Particle) -> int:
        try:
            return self._all_mothers.index(particle)
        except ValueError:
            # Try nearest mother
            close_match = None
            frame_match = None

            for mother in self._all_mothers:
                if mother.frame_number() == particle.frame_number():
                    frame_match = mother
                    if mother.z == particle.z:
                        close_match = mother

            if close_match is not None:
                return self._all_mothers.index(close_match)
            if frame_match is not None:
                return self._all_mothers.index(frame_match)
            return -1  # Give up

    def draw_view(self):
        self._clear_axis()
        if self._mother_index < 0 or self._mother_index >= len(self._all_mothers):
            if len(self._all_mothers) == 0:
                plt.title("No mothers found. Is the linking data missing?")
            else:
                plt.title("No mother found at mouse position."
                          "\nPress the right arrow key to view the first mother in the sample.")
            plt.draw()
            return

        self._zoom_to_mother()
        self._show_image()

        mother = self._all_mothers[self._mother_index]
        plt.title("Mother " + str(self._mother_index + 1) + "/" + str(len(self._all_mothers)) + "\n" + str(mother))

        plt.draw()

    def _zoom_to_mother(self):
        mother = self._all_mothers[self._mother_index]
        self._ax.set_xlim(mother.x - 50, mother.x + 50)
        self._ax.set_ylim(mother.y - 50, mother.y + 50)
        self._ax.set_autoscale_on(False)

    def _show_image(self):
        mother = self._all_mothers[self._mother_index]
        image_stack = self._experiment.get_frame(mother.frame_number()).load_images()
        if image_stack is not None:
            image = self._ax.imshow(image_stack[int(mother.z)], cmap="gray")
            plt.colorbar(mappable=image, ax=self._ax)

    def _goto_next(self):
        self._mother_index += 1
        if self._mother_index >= len(self._all_mothers):
            self._mother_index = 0
        self.draw_view()

    def _goto_previous(self):
        self._mother_index -= 1
        if self._mother_index < 0:
            self._mother_index = len(self._all_mothers) - 1
        self.draw_view()

    def _goto_full_image(self):
        from imaging.image_visualizer import StandardImageVisualizer

        if self._mother_index < 0 or self._mother_index >= len(self._all_mothers):
            # Don't know where to go
            image_visualizer = StandardImageVisualizer(self._experiment, self._fig)
        else:
            mother = self._all_mothers[self._mother_index]
            image_visualizer = StandardImageVisualizer(self._experiment, self._fig,
                                               frame_number=mother.frame_number(), z=int(mother.z))
        activate(image_visualizer)

    def _on_key_press(self, event: KeyEvent):
        if event.key == "left":
            self._goto_previous()
        elif event.key == "right":
            self._goto_next()
        elif event.key == "m":
            self._goto_full_image()
