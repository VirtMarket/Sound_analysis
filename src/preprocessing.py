from pydub import AudioSegment
from pathlib import Path

base_path = Path('../data')
room_path = base_path / 'room'
shops_path = base_path / 'shops'
test_audio_path = shops_path / 'lewiatan_001.wav'

# split to 4sec chunks

t_start = 250  # cut the first 250ms, which might be strangely recorded
t_end = t_start + 4000

original_audio = AudioSegment.from_file(test_audio_path, format='wav')
length = len(original_audio)
channels = original_audio.channels
no_frames = original_audio.frame_count()

i = 0
while t_end <= length:
    extracted_audio = original_audio[t_start:t_end]
    export_path = shops_path / 'lewiatan_001_{}.wav'.format(str(i).zfill(3))
    extracted_audio.export(export_path, format="wav")
    t_start = t_start + 4000
    t_end = t_end + 4000
    i = i + 1

print(length)
#newAudio = newAudio[t1:t2]
#newAudio.export('newSong.wav', format="wav")