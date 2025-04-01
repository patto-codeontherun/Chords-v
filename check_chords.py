import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pygame
import os

# Global variable to stop audio and animation
stop_audio = False


def analyze_whole_file(file_path):
    y, sr = librosa.load(file_path)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=12, n_octaves=7)

    # Chord templates for major and minor chords
    chord_templates = np.array([
        [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # C
        [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],  # C#
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],  # D
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],  # D#
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],  # E
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # F
        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],  # F#
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # G
        [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],  # G#
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # A
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # A#
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],  # B
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # Cm
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # C#m
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],  # Dm
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],  # D#m
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # Em
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],  # Fm
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],  # F#m
        [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # Gm
        [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # G#m
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # Am
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],  # A#m
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],  # Bm
    ])

    # Prepare chord names
    chord_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
                   'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'A#m', 'Bm']

    chord_indices = np.argmax(np.dot(chord_templates, chroma), axis=0)
    detected_chords = [chord_names[idx] for idx in chord_indices]

    S = librosa.stft(y)
    freqs = librosa.fft_frequencies(sr=sr)
    magnitudes = np.abs(S)
    dominant_freq_indices = np.argmax(magnitudes, axis=0)
    dominant_freqs = freqs[dominant_freq_indices]

    return chroma, detected_chords, dominant_freqs


def update_plot(frame, chroma, chords, freqs, ax, text):
    global stop_audio
    if stop_audio:  # Stop updating if audio is stopped
        return ax

    ax.clear()

    # Equalizer-style bars
    bar_width = 0.6  # Adjust bar width as needed
    colors = plt.cm.viridis(np.linspace(0, 1, 12))  # Gradient color map

    for i in range(12):
        ax.bar(i, chroma[:, frame][i], bar_width, color=colors[i])

    ax.set_ylim(0, np.max(chroma))
    ax.set_xlim(-0.5, 11.5)
    ax.set_xticks(range(12))
    ax.set_xticklabels(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'], color='white')  # White labels
    ax.set_ylabel('Chroma Intensity', color='white')  # White label
    ax.tick_params(axis='y', colors='white')  # White y ticks
    ax.set_facecolor('black')  # Black background
    ax.spines['bottom'].set_color('white')  # White spines
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')

    text.set_text(f"Time: {frame * 512 / 22050:.2f}s\nChord: {chords[frame]}\nFreq: {freqs[frame]:.0f} Hz")
    text.set_color('yellow')  # Bright text color

    return ax


def save_songbook(chords, directory, song_name):
    os.makedirs(directory, exist_ok=True)  # Create the directory if it doesn't exist
    songbook_path = os.path.join(directory, 'songbook.txt')

    with open(songbook_path, 'w') as f:
        f.write(f"Song: {song_name}\n\n")  # Write the song name at the top
        for chord in chords:
            f.write(f"{chord}\n")  # Write each chord in a new line

    print(f"Songbook saved: {songbook_path}")


def play_audio_and_show_analysis(file_path):
    global stop_audio
    stop_audio = False  # Reset stop flag
    chroma, chords, freqs = analyze_whole_file(file_path)

    # Extract the song name from the file path
    song_name = os.path.basename(file_path).replace('.mp3', '')

    pygame.mixer.init()
    pygame.display.init()  # Initialize the display module
    pygame.mixer.music.load(file_path)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='black')  # Black figure background
    text = ax.text(0.95, 0.95, '', horizontalalignment='right', verticalalignment='top', transform=ax.transAxes,
                   color='yellow')  # Bright text

    anim = FuncAnimation(fig, update_plot, frames=chroma.shape[1],
                         fargs=(chroma, chords, freqs, ax, text),
                         interval=512 / 22050 * 1000, repeat=False)

    pygame.mixer.music.play()

    def on_close(event):
        global stop_audio
        stop_audio = True  # Set stop flag to True when the plot is closed
        songbook_dir = 'songbook'  # Specify the directory for saving the songbook
        save_songbook(chords, songbook_dir, song_name)  # Save songbook with the song name

    # Connect the close event
    plt.gcf().canvas.mpl_connect('close_event', on_close)

    # Check if audio ends
    pygame.mixer.music.set_endevent(pygame.USEREVENT)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop_audio = True
                pygame.quit()
                plt.close(fig)
                return
            elif event.type == pygame.USEREVENT:
                stop_audio = True  # Stop the audio and animation when the song ends
                anim.event_source.stop()  # Stop the animation when the song ends
                save_songbook(chords, 'songbook', song_name)  # Save songbook with the song name
                plt.close(fig)  # Close the plot

        plt.pause(0.1)


play_audio_and_show_analysis('input/Ed Sheeran - Shivers (Lyrics).mp3')