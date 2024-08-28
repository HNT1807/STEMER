
import streamlit as st
import os
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
import numpy as np
import tempfile
import zipfile
import io
import soundfile as sf

# Set page config
st.set_page_config(page_title="STEMER", layout="wide")

# Custom CSS for centering and styling
st.markdown("""
<style>
.stApp {
    max-width: 800px;
    margin: 0 auto;
    float: none;
}
.st-emotion-cache-1v0mbdj {
    width: 100%;
}
.stButton > button {
    display: block;
    margin: 0 auto;
}
.streamlit-expanderHeader {
    font-size: 1em;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Centered title and subtitle
st.markdown("<h1 style='text-align: center;'>STEMER</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Batch Audio Stem Separation</h3>", unsafe_allow_html=True)


def is_float_dtype(dtype):
    return np.issubdtype(dtype, np.floating)


def prevent_clip(wav, mode='rescale'):
    if not is_float_dtype(wav.dtype):
        raise ValueError("Input must be a float array")

    if mode == 'rescale':
        peak = np.abs(wav).max()
        if peak > 1:
            wav = wav / peak
    elif mode == 'clamp':
        wav = np.clip(wav, -1, 1)
    else:
        raise ValueError(f"Invalid mode {mode}")
    return wav


def process_audio(audio_file, model, selected_stems, output_dir, progress_callback):
    # Read the audio file directly from the UploadedFile object
    audio_data, sample_rate = sf.read(io.BytesIO(audio_file.getvalue()))

    progress_callback(0.1)  # 10% progress after reading file

    # Convert to torch tensor
    audio = torch.tensor(audio_data.T, dtype=torch.float32)
    audio = audio.unsqueeze(0)  # Add batch dimension

    progress_callback(0.2)  # 20% progress after converting to tensor

    # Apply the model
    sources = apply_model(model, audio, split=True, device="cpu")
    sources = sources.squeeze(0).cpu().numpy()

    progress_callback(0.5)  # 50% progress after applying model

    # Ensure correct ordering of stem names
    stem_names = model.sources
    stem_files = []
    for i, (name, source) in enumerate(zip(stem_names, sources)):
        if name in selected_stems:
            source = source.T
            stem_path = os.path.join(output_dir, f"{audio_file.name}_{name}.wav")
            sf.write(stem_path, source, sample_rate)
            stem_files.append((name, stem_path))
        progress_callback(0.5 + 0.5 * (i + 1) / len(stem_names))

    return stem_files


# Main app logic
def main():
    # Load Demucs model
    @st.cache_resource
    def load_model():
        return get_model("htdemucs")

    model = load_model()

    # File uploader
    uploaded_files = st.file_uploader(label="", accept_multiple_files=True, type=['mp3', 'wav', 'aif'])

    if uploaded_files:
        file_stem_selections = {}
        for audio_file in uploaded_files:
            with st.expander(f"File: {audio_file.name}", expanded=False):
                stem_options = ["vocals", "drums", "bass", "other"]
                selected_stems = st.multiselect(f"Select the stems you want", stem_options,
                                                default=stem_options, key=audio_file.name)
                file_stem_selections[audio_file.name] = selected_stems

        if st.button("Process Files"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            with tempfile.TemporaryDirectory() as tmpdirname:
                all_stems = []
                for i, audio_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {audio_file.name}...")

                    def update_progress(file_progress):
                        total_progress = (i + file_progress) / len(uploaded_files)
                        progress_bar.progress(total_progress)

                    stems = process_audio(audio_file, model, file_stem_selections[audio_file.name], tmpdirname,
                                          update_progress)
                    all_stems.extend(stems)

                # Create ZIP file
                status_text.text("Creating ZIP file...")
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                    for stem_name, stem_path in all_stems:
                        zip_file.write(stem_path, os.path.basename(stem_path))

                # Offer download
                status_text.text("Ready for download!")

                # Center the download button using markdown
                st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                st.download_button(
                    label="Download Stems",
                    data=zip_buffer.getvalue(),
                    file_name="selected_stems.zip",
                    mime="application/zip"
                )
                st.markdown("</div>", unsafe_allow_html=True)

            progress_bar.progress(1.0)
            status_text.text("All files processed successfully!")


if __name__ == "__main__":
    main()
