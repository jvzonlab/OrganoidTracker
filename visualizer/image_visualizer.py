import math

import cv2
from typing import Optional, Iterable, List, Tuple, Union

from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.figure import Figure
from matplotlib.colors import Colormap
from networkx import Graph
from numpy import ndarray
from tifffile import tifffile

import core
from core import Experiment, TimePoint, Particle
from gui import launch_window, Window, dialog
from gui.dialog import popup_figure, prompt_int, popup_error
from linking import particle_flow
from linking_analysis import volume_and_intensity_graphs
from particle_detection import single_particle_detection
from visualizer import Visualizer, activate, DisplaySettings


def show(experiment: Experiment):
    """Creates a standard visualizer for an experiment."""
    window = launch_window(experiment)
    visualizer = StandardImageVisualizer(window)
    activate(visualizer)


class AbstractImageVisualizer(Visualizer):
    """A generic image visualizer."""

    MAX_Z_DISTANCE: int = 3
    DEFAULT_SIZE = (30, 500, 500)

    _time_point: TimePoint = None
    _time_point_images: ndarray = None
    _z: int
    __drawn_particles: List[Particle]
    _display_settings: DisplaySettings

    # The color map should typically not be transferred when switching to another viewer, so it is not part of the
    # display_settings property
    _color_map: Union[str, Colormap] = "gray"

    def __init__(self, window: Window, time_point_number: Optional[int] = None, z: int = 14,
                 display_settings: DisplaySettings = None):
        super().__init__(window)

        self._display_settings = DisplaySettings() if display_settings is None else display_settings
        if time_point_number is None:
            time_point_number = window.get_experiment().first_time_point_number()
        self._time_point, self._time_point_images = self._load_time_point(time_point_number)
        self._z = int(z)
        self._clamp_z()
        self.__drawn_particles = []

    def _load_time_point(self, time_point_number: int) -> Tuple[TimePoint, ndarray]:
        time_point = self._experiment.get_time_point(time_point_number)
        if self._display_settings.show_images:
            if self._display_settings.show_reconstruction:
                time_point_images = self.reconstruct_image(time_point, self._guess_image_size(time_point))
            else:
                time_point_images = self.load_image(time_point, self._display_settings.show_next_time_point)
        else:
            time_point_images = None

        return time_point, time_point_images

    def _export_images(self):
        if self._time_point_images is None:
            raise core.UserError("No images loaded", "Saving images failed: there are no images loaded")
        file = dialog.prompt_save_file("Save 3D file as...", [("TIF file", "*.tif")])
        if file is None:
            return
        flat_image = self._time_point_images.ravel()

        image_shape = self._time_point_images.shape
        if len(image_shape) == 3 and isinstance(self._color_map, Colormap):
            # Convert grayscale image to colored using the stored color map
            images: ndarray = self._color_map(flat_image, bytes=True)[:,0:3]
            new_shape = (image_shape[0], image_shape[1], image_shape[2], 3)
            images = images.reshape(new_shape)
        else:
            images = cv2.convertScaleAbs(self._time_point_images, alpha=256 / self._time_point_images.max(), beta=0)
        tifffile.imsave(file, images)

    def _guess_image_size(self, time_point):
        images_for_size = self._time_point_images
        if images_for_size is None:
            images_for_size = self.load_image(time_point, False)
        size = images_for_size.shape if images_for_size is not None else self.DEFAULT_SIZE
        return size

    def draw_view(self):
        self._clear_axis()
        self.__drawn_particles.clear()
        self._draw_image()
        errors = self._draw_particles()
        self._draw_extra()
        self._window.set_figure_title(self._get_figure_title(errors))

        self._fig.canvas.draw()

    def _draw_image(self):
        if self._time_point_images is not None:
            self._ax.imshow(self._time_point_images[self._z], cmap=self._color_map)

    def _get_figure_title(self, errors: int) -> str:
        title = "Time point " + str(self._time_point.time_point_number()) + "    (z=" + str(self._z) + ")"
        if errors != 0:
            title += " (changes: " + str(errors) + ")"
        return title

    def _must_show_other_time_points(self) -> bool:
        return True

    def _draw_extra(self):
        pass # Subclasses can override this

    def _draw_particles(self) -> int:
        """Draws particles and links. Returns the amount of non-equal links in the image"""

        # Draw particles
        self._draw_particles_of_time_point(self._time_point)

        # Next time point
        can_show_other_time_points = self._must_show_other_time_points() and (
                                         self._experiment.particle_links() is not None
                                         or self._experiment.particle_links_scratch() is not None)
        if self._display_settings.show_next_time_point or can_show_other_time_points:
            # Only draw particles of next/previous time point if there is linking data, or if we're forced to
            try:
                self._draw_particles_of_time_point(self._experiment.get_next_time_point(self._time_point), color='red')
            except KeyError:
                pass  # There is no next time point, ignore

        # Previous time point
        if not self._display_settings.show_next_time_point and can_show_other_time_points:
            try:
                self._draw_particles_of_time_point(self._experiment.get_previous_time_point(self._time_point),
                                                   color='blue')
            except KeyError:
                pass  # There is no previous time point, ignore

        # Draw links
        errors = 0
        if can_show_other_time_points:
            for particle in self._time_point.particles():
                errors += self._draw_links(particle)

        return errors

    def _draw_particles_of_time_point(self, time_point: TimePoint, color: str = core.COLOR_CELL_CURRENT):
        dt = time_point.time_point_number() - self._time_point.time_point_number()
        for particle in time_point.particles():
            dz = self._z - round(particle.z)

            # Draw the particle itself (as a square or circle, depending on its depth)
            self._draw_particle(particle, color, dz, dt)

    def _draw_particle(self, particle: Particle, color: str, dz: int, dt: int):
        # Draw error marker
        if abs(dz) <= self.MAX_Z_DISTANCE:
            graph = self._experiment.particle_links_scratch() or self._experiment.particle_links()
            if graph is not None and particle in graph and "error" in graph.nodes[particle]:
                self._draw_error(particle, dz)

        # Draw particle marker
        self._time_point.get_shape(particle).draw2d(particle.x, particle.y, dz, dt, self._ax, color)

        self.__drawn_particles.append(particle)

    def _draw_error(self, particle: Particle, dz: int):
        self._ax.plot(particle.x, particle.y, 'X', color='black', markeredgecolor='white',
                      markersize=19 - abs(dz), markeredgewidth=2)

    def _draw_links(self, particle: Particle) -> int:
        """Draws links between the particles. Returns 1 if there is 1 error: the baseline links don't match the actual
        links.
        """
        links_normal = self._get_links(self._experiment.particle_links_scratch(), particle)
        links_baseline = self._get_links(self._experiment.particle_links(), particle)

        self._draw_given_links(particle, links_normal, line_style='dotted', line_width=3)
        self._draw_given_links(particle, links_baseline)

        # Check for errors
        if self._experiment.particle_links_scratch() is not None and self._experiment.particle_links() is not None:
            if links_baseline != links_normal:
                return 1
        return 0

    def _draw_given_links(self, particle, links, line_style='solid', line_width=1):
        for linked_particle in links:
            if abs(linked_particle.z - self._z) > self.MAX_Z_DISTANCE\
                    and abs(particle.z - self._z) > self.MAX_Z_DISTANCE:
                continue
            if linked_particle.time_point_number() < particle.time_point_number():
                # Drawing to past
                if not self._display_settings.show_next_time_point:
                    self._ax.plot([particle.x, linked_particle.x], [particle.y, linked_particle.y], linestyle=line_style,
                                  color=core.COLOR_CELL_PREVIOUS, linewidth=line_width)
            else:
                self._ax.plot([particle.x, linked_particle.x], [particle.y, linked_particle.y], linestyle=line_style,
                              color=core.COLOR_CELL_NEXT, linewidth=line_width)

    def _get_links(self, network: Optional[Graph], particle: Particle) -> Iterable[Particle]:
        if network is None:
            return []
        try:
            return network[particle]
        except KeyError:
            return []

    def _get_particle_at(self, x: Optional[int], y: Optional[int]) -> Optional[Particle]:
        """Wrapper of get_closest_particle that makes use of the fact that we can lookup all particles ourselves."""
        return self.get_closest_particle(self.__drawn_particles, x, y, None, max_distance=5)

    def get_extra_menu_options(self):
        def time_point_prompt():
            min_str = str(self._experiment.first_time_point_number())
            max_str = str(self._experiment.last_time_point_number())
            given = prompt_int("Time point", "Which time point do you want to go to? (" + min_str + "-" + max_str
                               + ", inclusive)")
            if given is None:
                return
            if not self._move_to_time(given):
                popup_error("Out of range", "Oops, time point " + str(given) + " is outside the range " + min_str + "-"
                            + max_str + ".")
        return {
            **super().get_extra_menu_options(),
            "File/Export-Export image...": self._export_images,
            "View/Toggle-Toggle showing two time points (" + DisplaySettings.KEY_SHOW_NEXT_IMAGE_ON_TOP.upper() + ")":
                self._toggle_showing_next_time_point,
            "View/Toggle-Toggle showing images (" + DisplaySettings.KEY_SHOW_IMAGES.upper() + ")":
                self._toggle_showing_images,
            "View/Toggle-Toggle showing reconstruction (" + DisplaySettings.KEY_SHOW_RECONSTRUCTION.upper() + ")":
                self._toggle_showing_reconstruction,
            "Navigate/Layer-Above layer (Up)": lambda: self._move_in_z(1),
            "Navigate/Layer-Below layer (Down)": lambda: self._move_in_z(-1),
            "Navigate/Time-Next time point (Right)": lambda: self._move_in_time(1),
            "Navigate/Time-Previous time point (Left)": lambda: self._move_in_time(-1),
            "Navigate/Time-Other time point... (/t*)": time_point_prompt
        }

    def _on_key_press(self, event: KeyEvent):
        if event.key == "up":
            self._move_in_z(1)
        elif event.key == "down":
            self._move_in_z(-1)
        elif event.key == "left":
            self._move_in_time(-1)
        elif event.key == "right":
            self._move_in_time(1)
        elif event.key == DisplaySettings.KEY_SHOW_NEXT_IMAGE_ON_TOP:
            self._toggle_showing_next_time_point()
        elif event.key == DisplaySettings.KEY_SHOW_IMAGES:
            self._toggle_showing_images()
        elif event.key == DisplaySettings.KEY_SHOW_RECONSTRUCTION:
            self._toggle_showing_reconstruction()

    def _on_command(self, command: str) -> bool:
        if command[0] == "t":
            time_point_str = command[1:]
            try:
                new_time_point_number = int(time_point_str.strip())
                self._move_to_time(new_time_point_number)
            except ValueError:
                self._update_status("Cannot read number: " + time_point_str)
            return True
        if command == "help":
            self._update_status("/t20: Jump to time point 20 (also works for other time points)")
            return True
        return False

    def _toggle_showing_next_time_point(self):
        self._display_settings.show_next_time_point = not self._display_settings.show_next_time_point
        self.refresh_view()

    def _toggle_showing_images(self):
        self._display_settings.show_images = not self._display_settings.show_images
        self.refresh_view()

    def _toggle_showing_reconstruction(self):
        self._display_settings.show_reconstruction = not self._display_settings.show_reconstruction
        self.refresh_view()

    def _move_in_z(self, dz: int):
        old_z = self._z
        self._z += dz

        self._clamp_z()

        if self._z != old_z:
            self.draw_view()

    def _clamp_z(self):
        if self._z < 0:
            self._z = 0
        if self._time_point_images is not None and self._z >= len(self._time_point_images):
            self._z = len(self._time_point_images) - 1

    def _move_to_time(self, new_time_point_number: int) -> bool:
        try:
            self._time_point, self._time_point_images = self._load_time_point(new_time_point_number)
            self._move_in_z(0)  # Caps z to allowable range
            self.draw_view()
            self._update_status("Moved to time point " + str(new_time_point_number) + "!")
            return True
        except KeyError:
            self._update_status("Unknown time point: " + str(new_time_point_number) + " (range is "
                                + str(self._experiment.first_time_point_number()) + " to "
                                + str(self._experiment.last_time_point_number()) + ", inclusive)")
            return False

    def _move_in_time(self, dt: int):
        self._color_map = AbstractImageVisualizer._color_map

        old_time_point_number = self._time_point.time_point_number()
        new_time_point_number = old_time_point_number + dt
        try:
            self._time_point, self._time_point_images = self._load_time_point(new_time_point_number)
            self._move_in_z(0)  # Caps z to allowable range
            self.draw_view()
            self._update_status(self.__doc__)
        except KeyError:
            pass

    def refresh_view(self):
        self._move_in_time(0)  # This makes the viewer reload the image


class StandardImageVisualizer(AbstractImageVisualizer):
    """Cell and image viewer

    Moving: left/right moves in time, up/down in the z-direction and type '/t30' + ENTER to jump to time point 30
    Viewing: N shows next frame in red, current in green and T shows trajectory of cell under the mouse cursor
    Cell lists: M shows mother cells, E shows detected errors and D shows differences between two loaded data sets
    Editing: L shows an editor for links                    Other: S shows the detected shape, F the detected flow"""

    def __init__(self, window: Window, time_point_number: Optional[int] = None, z: int = 14,
                 display_settings: Optional[DisplaySettings] = None):
        super().__init__(window, time_point_number=time_point_number, z=z, display_settings=display_settings)

    def _on_mouse_click(self, event: MouseEvent):
        if event.dblclick and event.button == 1:
            particle = self._get_particle_at(event.xdata, event.ydata)
            if particle is not None:
                self.__display_cell_division_scores(particle)
        else:
            super()._on_mouse_click(event)

    def __display_cell_division_scores(self, particle):
        cell_divisions = list(self._time_point.mother_scores(particle))
        cell_divisions.sort(key=lambda d: d.score.total(), reverse=True)
        displayed_items = 0
        text = ""
        for scored_family in cell_divisions:
            if displayed_items >= 4:
                text += "... and " + str(len(cell_divisions) - displayed_items) + " more"
                break
            text += str(displayed_items + 1) + ". " + str(scored_family.family) + ", score: " \
                    + str(scored_family.score.total()) + "\n"
            displayed_items += 1
        if text:
            self._update_status("Possible cell division scores:\n" + text)
        else:
            self._update_status("No cell division scores found")

    def get_extra_menu_options(self):
        return {
            **super().get_extra_menu_options(),
            "Edit/Manual-Links... (L)": self._show_link_editor,
            "Edit/Manual-Positions... (P)": self._show_position_editor,
            "Edit/Automatic-Cell detection...": self._show_cell_detector,
            "View/Linking-Linking differences (D)": self._show_linking_differences,
            "View/Linking-Linking errors and warnings (E)": self._show_linking_errors,
            "View/Cell-Cell divisions (/divisions)": self._show_mother_cells,
            "View/Cell-Cell deaths (/deaths)": self._show_dead_cells,
            "View/Cell-Cell volumes...": self._show_cell_volumes,
            "View/Cell-Cell intensities...": self._show_cell_intensities
        }

    def _on_key_press(self, event: KeyEvent):
        if event.key == "t":
            particle = self._get_particle_at(event.xdata, event.ydata)
            if particle is not None:
                from visualizer.track_visualizer import TrackVisualizer
                track_visualizer = TrackVisualizer(self._window, particle)
                activate(track_visualizer)
        elif event.key == "e":
            particle = self._get_particle_at(event.xdata, event.ydata)
            self._show_linking_errors(particle)
        elif event.key == "d":
            particle = self._get_particle_at(event.xdata, event.ydata)
            self._show_linking_differences(particle)
        elif event.key == "l":
            self._show_link_editor()
        elif event.key == "p":
            self._show_position_editor()
        elif event.key == "s":
            particle = self._get_particle_at(event.xdata, event.ydata)
            if particle is not None:
                self.__show_shape(particle)
        elif event.key == "f":
            particle = self._get_particle_at(event.xdata, event.ydata)
            links = self._experiment.particle_links_scratch()
            if particle is not None and links is not None:
                self._update_status("Flow toward previous frame: " +
                                    str(particle_flow.get_flow_to_previous(links, self._time_point, particle)) +
                                   "\nFlow towards next frame: " +
                                    str(particle_flow.get_flow_to_next(links, self._time_point, particle)))
        else:
            super()._on_key_press(event)

    def _show_cell_detector(self):
        if self._experiment.get_image_stack(self._time_point) is None:
            dialog.popup_error("No images", "There are no images loaded, so we cannot detect cells.")
            return
        from visualizer.detection_visualizer import DetectionVisualizer
        activate(DetectionVisualizer(self._window, self._time_point.time_point_number(), self._z,
                                     self._display_settings))

    def _show_mother_cells(self):
        from visualizer.cell_division_visualizer import CellDivisionVisualizer
        track_visualizer = CellDivisionVisualizer(self._window)
        activate(track_visualizer)

    def _show_cell_volumes(self):
        def draw(figure: Figure):
            volume_and_intensity_graphs.plot_volumes(self._experiment, figure)
        dialog.popup_figure(draw)

    def _show_cell_intensities(self):
        def draw(figure: Figure):
            volume_and_intensity_graphs.plot_intensities(self._experiment, figure)
        dialog.popup_figure(draw)

    def _show_linking_errors(self, particle: Optional[Particle] = None):
        from visualizer.errors_visualizer import ErrorsVisualizer
        warnings_visualizer = ErrorsVisualizer(self._window, particle)
        activate(warnings_visualizer)

    def _show_linking_differences(self, particle: Optional[Particle] = None):
        from visualizer.differences_visualizer import DifferencesVisualizer
        differences_visualizer = DifferencesVisualizer(self._window, particle)
        activate(differences_visualizer)

    def _show_link_editor(self):
        from visualizer.link_editor import LinkEditor
        link_editor = LinkEditor(self._window, time_point_number=self._time_point.time_point_number(), z=self._z)
        activate(link_editor)

    def _show_position_editor(self):
        from visualizer.position_editor import PositionEditor
        position_editor = PositionEditor(self._window, time_point_number=self._time_point.time_point_number(),
                                         z=self._z)
        activate(position_editor)

    def _on_command(self, command: str) -> bool:
        if command == "deaths":
            self._show_dead_cells()
            return True
        if command == "divisions":
            self._show_mother_cells()
            return True
        if command == "help":
            self._update_status("Available commands:\n"
                               "/deaths - views cell deaths.\n"
                               "/divisions - views cell divisions.\n"
                               "/t20 - jumps to time point 20 (also works for other time points")
            return True
        return super()._on_command(command)

    def _show_dead_cells(self):
        from visualizer.cell_death_visualizer import CellDeathVisualizer
        activate(CellDeathVisualizer(self._window, None))

    def __show_shape(self, particle: Particle):
        image_stack = self._time_point_images if not self._display_settings.show_next_time_point \
            else self._experiment.get_image_stack(self._time_point)
        if image_stack is None:
            return  # No images loaded
        image = image_stack[int(particle.z)]
        x, y, r = int(particle.x), int(particle.y), 16
        image_local = image[y - r:y + r, x - r:x + r]
        result_image, ellipses = single_particle_detection.perform(image_local)

        def show_segmentation(figure: Figure):
            figure.gca().imshow(result_image)
            for ellipse in ellipses:
                figure.gca().add_artist(ellipse)


        popup_figure(show_segmentation)


