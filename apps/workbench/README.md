# Workbench

This folder will contain the browser UI for:

1. importing and browsing capture sessions;
2. drawing boxes and entering transcriptions;
3. reviewing and correcting annotations;
4. creating dataset releases;
5. launching and monitoring training;
6. comparing predictions with verified labels;
7. running image, video, and live-camera inference.

The workbench must call functions from `container_vision`. Dataset, training,
and inference logic should remain usable without the UI.

The first UI should favor a small Gradio application. A separate frontend/API
architecture can be introduced later if deployment or team boundaries require
it.

