import os
import json
import argparse # Parser for command-line options, arguments and sub-commands
from tqdm import tqdm  # A Fast, Extensible Progress Bar for Python and CLI
from google.cloud import storage
from google.cloud import speech_v1p1beta1 as speech
from google.protobuf.json_format import MessageToDict
from typing import *
import gconfig # google cloud config file

def main(args):
    print("Started!!!!")
    # Check if the output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # List all audio files in the bucket.
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(gconfig.BUCKET_NAME)
    blobs = bucket.list_blobs()
    # 'blobs is a list of Google blob objects. We need to extract filenames.
    original_filenames = [b.name for b in blobs]

    # Create a signle Google API client and configuration to reuse
    client = speech.SpeechClient()
    rc = speech.types.RecognitionConfig(
        encoding = speech.RecognitionConfig.AudioEncoding.FLAC,
        language_code = "th-TH",
        enable_word_confidence=True,
        enable_word_time_offsets=True,
        enable_speaker_diarization=True,
        diarization_speaker_count=2,
        audio_channel_count=1,
        model="default", # phone_call model is not support th-TH
    ) 

    # Skip already completed files.
    filenames: List[str] = []
    for filename in original_filenames:
        output_fqn = os.path.join(
            args.output_dir, filename.replace(".flac", ".json")
        )
        if os.path.exists(output_fqn):
            continue
        else:
            filenames.append(filename)

    print(f"Saving json output to: {args.output_dir}")
    print(f"Transcribing {len(filenames)} files from bucket: {gconfig.BUCKET_NAME}")

    for filename in tqdm(filenames):
        # Run Automatic Speech Recognition
        audio = speech.types.RecognitionAudio(
            uri=f"gs://{gconfig.BUCKET_NAME}/{filename}"
        )

        #response = client.long_running_recognize()

        #ret = transcribe(client=client, rc=rc, audio=audio)

        transcribe2(client=client, rc=rc, audio=audio)

        # Save the output to json file.
        #with open(output_fqn, "w") as pointer:
        #    json.dump(ret, pointer, indent=2, separators=(",", ": "))


def transcribe2(client: speech.SpeechClient, rc: speech.types.RecognitionConfig, audio: speech.types.RecognitionAudio):
    operation = client.long_running_recognize(config=rc, audio=audio)

    print("Waiting for operation to complete...")
    response = operation.result(timeout=None)

    transcript = ""
    confidence = 0.0
    count = 0

    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    for result in response.results:
        transcript += result.alternatives[0].transcript
        confidence += result.alternatives[0].confidence
        count += 1
        # The first alternative is the most likely one for this portion.
        print(u"Transcript: {}".format(result.alternatives[0].transcript))
        print("Confidence: {}".format(result.alternatives[0].confidence))
    
    confidence = confidence / count
    print(u"Transcript: {}".format(transcript))
    print("Average Confidence {}".format(confidence))
    print("Completed...")


def transcribe(client: speech.SpeechClient, rc: speech.types.RecognitionConfig, audio: speech.types.RecognitionAudio):
    operation = client.long_running_recognize(config=rc, audio=audio)
    response = operation.result()
    result = MessageToDict(response)
    return result

if __name__ == "__main__":
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gconfig.KEY
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=str, help="Location for the transcription output.")
    args = parser.parse_args()
    main(args)