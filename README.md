# Audio Analysis and Chord Detection with Visualization

This Python script analyzes an audio file (specifically `.mp3` format), detects chords using chroma features, visualizes the chroma and dominant frequency over time, and generates a songbook in a text file.

## Features

* **Audio Loading and Analysis:** Uses `librosa` to load audio files and extract chroma features and dominant frequencies.
* **Chord Detection:** Detects chords by comparing chroma features to predefined chord templates.
* **Real-time Visualization:** Displays a dynamic bar chart of chroma intensities and updates the detected chord and dominant frequency in real-time as the audio plays.
* **Audio Playback:** Plays the audio file using `pygame`.
* **Songbook Generation:** Saves the detected chords to a `.txt` file (songbook) in a specified directory.
* **Event Handling:** Stops audio playback and animation when the plot is closed or when the audio finishes.

## Dependencies

* `librosa`
* `numpy`
* `matplotlib`
* `matplotlib.animation`
* `pygame`

You can install these dependencies using pip:

```bash
pip install librosa numpy matplotlib pygame
