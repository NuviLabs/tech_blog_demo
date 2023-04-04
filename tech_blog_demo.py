import cv2
import json
import numpy as np

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from helper import pix2point, json2results, get_food_mask

DEPTH_INTRINSICS_1280x720 = {
    "coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
    "fx": 922.386962890625,
    "fy": 921.6760864257812,
    "height": 720,
    "ppx": 634.551513671875,
    "ppy": 350.37774658203125,
    "width": 1280
}

class AppWindow:
    """
    The main application window for the Open3D GUI.
    """

    MENU_SHOW_SETTINGS = 1

    def __init__(self, width: int, height: int):
        """
        Initializes the main application window.

        Args:
            width: The width of the window in pixels.
            height: The height of the window in pixels.
        """
        self.label_3d_lst = []

        # Initialize the GUI application
        app = gui.Application.instance
        app.initialize()

        # Set the font for the application
        font = gui.FontDescription()
        font.add_typeface_for_language("NanumSquareR", "ko")
        app.set_font(gui.Application.DEFAULT_FONT_ID, font)

        # Create the main window
        self.window = app.create_window("Open3D", width, height)
        w = self.window  # to make the code more concise

        # Create the scene widget for the 3D visualization
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)

        em = w.theme.font_size
        separation_height = int(round(0.5 * em))

        # Create the settings panel
        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        # Create the "Load PCD" button and add it to the settings panel
        lgbutton = gui.Button('RUN')
        lgbutton.horizontal_padding_em = 0.4 * em
        lgbutton.set_on_clicked(self._on_click_load_pcd)
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(lgbutton)
        h.add_stretch()
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(lgbutton)

        # Create the "Clear" button and add it to the settings panel
        clear_button = gui.Button('CLEAR')
        clear_button.horizontal_padding_em = 0.375 * em
        clear_button.set_on_clicked(self._on_clear)
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(clear_button)
        h.add_stretch()
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(h)

        # Create the checkbox group for the geometries and add it to the settings panel
        self.geo_check_boxes = gui.CollapsableVert("Geometries", 0.25 * em,
                                                   gui.Margins(em, 0, 0, 0))
        self.proxy = gui.WidgetProxy()
        self.geo_check_boxes.add_child(self.proxy)
        self._settings_panel.add_fixed(separation_height * 2)
        self._settings_panel.add_child(self.geo_check_boxes)

        # Create the "Reset camera position" button and add it to the settings panel
        self.camera_btn = gui.Button("Reset camera position")
        self.camera_btn.horizontal_padding_em = 0.19 * em
        self.camera_btn.set_on_clicked(self._set_camera)
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(self.camera_btn)
        h.add_stretch()
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(h)
        self._settings_panel.add_fixed(separation_height)

        # Set the function to be called whenever the window layout changes
        w.set_on_layout(self._on_layout)   
        # Add the scene widget and settings panel to the window
        w.add_child(self._scene)
        w.add_child(self._settings_panel)

        # If the menu bar has not been created yet
        if gui.Application.instance.menubar is None:   
            # Create a settings menu with an option to show the settings panel
            settings_menu = gui.Menu()
            settings_menu.add_item("Show sidebar", AppWindow.MENU_SHOW_SETTINGS)
            settings_menu.set_checked(AppWindow.MENU_SHOW_SETTINGS, True)

            # Create a top-level menu called "Settings" that contains the settings menu
            menu = gui.Menu()
            menu.add_menu("Settings", settings_menu)

            # Set the application's menu bar to the newly created menu
            gui.Application.instance.menubar = menu

        # Set the function to be called when the "Show sidebar" menu item is activated
        w.set_on_menu_item_activated(AppWindow.MENU_SHOW_SETTINGS, self._on_menu_toggle_settings_panel)

        app.run()   # Start the main event loop of the application

    def _set_camera(self):
        """Sets up the camera in the scene based on the bounding box and center of the scene."""
        # Get the bounding box of the scene and its center
        bounds = self._scene.scene.bounding_box
        center = bounds.get_center()
        # Set up the camera with a 60 degree field of view, the bounds of the scene,
        # and the center of the scene
        self._scene.setup_camera(60, bounds, center)
        # Look at the center of the scene from a point 500 units above it and from a
        # position that is behind the center of the scene
        self._scene.look_at(center, center - [0,0,500], np.array([0, -1, 0]))

    def _on_layout(self, layout_context):
        """Updates the layout of the scene and settings panel based on the given layout context."""
        # Get the content rectangle of the window
        r = self.window.content_rect
        # Set the frame of the scene to the content rectangle
        self._scene.frame = r
        # Calculate the width and height of the settings panel
        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        # Set the frame of the settings panel to be at the right edge of the content rectangle
        # with a width of `width` and a height of `height`
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width, height)

    def _on_clear(self):
        """Clears the scene geometry and removes 3D labels from the scene."""
        # Clear the geometry of the scene and reset a list of geometry names
        self._scene.scene.clear_geometry()
        self.all_geo_names = []
        # Remove any 3D labels from the scene if there are any
        if self.label_3d_lst:
            for label in self.label_3d_lst:
                self._scene.remove_3d_label(label)
        # Calculate the font size of the window theme and create a vertically scrollable layout
        # with a margin of `em`
        em = self.window.theme.font_size
        toggle_layout = gui.ScrollableVert(0.33 * em, gui.Margins(em, 0, 0, 0))
        # Set the widget of the `proxy` object to be the toggle layout
        self.proxy.set_widget(toggle_layout)

    def _on_click_load_pcd(self):
        """
        Callback function for loading a point cloud data (pcd) file.

        Reads a .png, .npy, and .json file that contain color, depth and semantic segmentation mask respectively.
        Converts the JSON file to results.
        Processes the results to extract food mask and corresponding 3D coordinates of the food points.
        Creates point clouds for each food object and adds it to the scene.
        """
        # Load the .png, .npy and .json files
        png = cv2.cvtColor(cv2.imread("data/demo.jpg"), cv2.COLOR_RGB2BGR)
        npy = np.load("data/demo.npy", allow_pickle=True)
        with open("data/demo.json", "r") as f:
            json_file = json.load(f)

        # Convert the JSON file to results
        results = json2results(json_file, npy.shape)

        # Get the food mask and 3D coordinates of the food points
        mesh_list = []
        mesh_info = results["class_names"]
        ranked_food_mask, food_indices = get_food_mask(results)
        real_coords = pix2point(npy, DEPTH_INTRINSICS_1280x720)

        # Create point clouds for each food object and add it to the scene
        for i in food_indices:
            food_mask = ranked_food_mask == i
            food_rgb = (png[food_mask] - 255) / 255
            food_depth = real_coords[food_mask]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.vstack(food_depth))
            pcd.colors = o3d.utility.Vector3dVector(np.vstack(food_rgb))
            mesh_list.append(pcd)

        # Clear the scene before adding new objects
        self._on_clear()

        # Add the point clouds to the scene and create 3D labels for each object
        em = self.window.theme.font_size
        toggle_layout = gui.ScrollableVert(0.33 * em, gui.Margins(em, 0, 0, 0))
        for i, label_name in enumerate(mesh_info):
            cur_geo = mesh_list[i]
            self.all_geo_names.append(label_name)

            # Add the point cloud to the scene
            mat = rendering.MaterialRecord()
            self._scene.scene.add_geometry(label_name, cur_geo, mat)

            # Create a 3D label for the object
            points = np.asarray(cur_geo.points)
            sorted_idx = np.argsort(points, axis=0)
            y_mid = sorted_idx[:, 1][sorted_idx.shape[0] // 2]
            point = points[y_mid]
            label = label_name
            label_3d = self._scene.add_3d_label(point, f"{label}")
            self.label_3d_lst.append(label_3d)

            # Create a checkbox to show/hide the object
            checkbox = gui.Checkbox(label_name)
            checkbox.set_on_checked(self._on_show_geo)
            checkbox.checked = True
            toggle_layout.add_child(checkbox)

        # Add the checkboxes to the layout
        self.proxy.set_widget(toggle_layout)

        # Set the camera to view the scene
        self._set_camera()

    def _on_show_geo(self, geo):
        """
        Callback function triggered when a checkbox for a geometry is checked/unchecked.
        Shows/hides the corresponding geometry in the scene based on the checkbox state.
        """
        toggle_layout = self.proxy.get_widget()
        checkboxes = toggle_layout.get_children()
        for i, checkbox in enumerate(checkboxes):
            checked = checkbox.checked
            # show/hide geometry based on checkbox state
            self._scene.scene.show_geometry(self.all_geo_names[i], checked)

    def _on_menu_toggle_settings_panel(self):
        """
        Callback function triggered when the "Show Settings" menu item is toggled.
        Toggles the visibility of the settings panel and updates the menu item state.
        """
        self._settings_panel.visible = not self._settings_panel.visible
        gui.Application.instance.menubar.set_checked(
            AppWindow.MENU_SHOW_SETTINGS, self._settings_panel.visible)


def main():
    w = AppWindow(900, 720)

if __name__ == "__main__":
    main()