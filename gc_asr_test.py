from pythainlp import sent_tokenize, word_tokenize
from attacut import tokenize, Tokenizer
# import jiwer
import io
import evaluation.util
import gcloud_config
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= gcloud_config.KEY
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/Users/thanaphon/Desktop/GCP Speech-to-Text/your-json-file.json"

from stopwatch import Stopwatch

def transcribe_file_with_diarization():
    """Transcribe the given audio file synchronously with diarization."""
    # [START speech_transcribe_diarization_beta]
    from google.cloud import speech_v1p1beta1 as speech
    client = speech.SpeechClient()

    speech_file = 'resources/voice_tom2.wav'

    with open(speech_file, 'rb') as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        #sample_rate_hertz=8000,
        audio_channel_count=2,
        language_code='th-TH',
        enable_speaker_diarization=True,
        diarization_speaker_count=2)

    print('Waiting for operation to complete...')
    response = client.recognize(config=config, audio=audio)

    # The transcript within each result is separate and sequential per result.
    # However, the words list within an alternative includes all the words
    # from all the results thus far. Thus, to get all the words with speaker
    # tags, you only have to take the words list from the last result:
    result = response.results[-1]

    words_info = result.alternatives[0].words

    # Printing out the output:
    for word_info in words_info:
        print(u"word: '{}', speaker_tag: {}".format(
            word_info.word, word_info.speaker_tag))
    # [END speech_transcribe_diarization_beta]

def transcribe_file_with_multichannel():
    """Transcribe the given audio file synchronously with multi channel."""
    # [START speech_transcribe_multichannel_beta]
    from google.cloud import speech_v1p1beta1 as speech
    client = speech.SpeechClient()

    speech_file = 'resources/voice_tom2.wav'

    with open(speech_file, 'rb') as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        #sample_rate_hertz=16000,
        language_code='th-TH',
        audio_channel_count=2,
        enable_separate_recognition_per_channel=True)

    response = client.recognize(config=config, audio=audio)

    for i, result in enumerate(response.results):
        alternative = result.alternatives[0]
        print('-' * 20)
        print('First alternative of result {}'.format(i))
        print(u'Transcript: {}'.format(alternative.transcript))
        print(u'Channel Tag: {}'.format(result.channel_tag))
    # [END speech_transcribe_multichannel_beta]

def transcribe_file_with_auto_punctuation():
    from google.cloud import speech_v1p1beta1 as speech
    client = speech.SpeechClient()

    speech_file = 'resources/Google_Gnome.wav'

    with io.open(speech_file, 'rb') as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code='en-US',
        enable_automatic_punctuation=True
    )

    response = client.recognize(config=config, audio=audio)
    
    for i, result in enumerate(response.results):
        alternative = result.alternatives[0]
        print('-' * 20)
        print(u'First alternative of result {}'.format(i))
        print(u'Transcript: {}'.format(alternative.transcript))
    # [END speech_transcribe_auto_punctuation_beta]

def transcribe_file_with_word_level_confidence():
    """Transcribe the given audio file synchronously with word level confidence."""
    # [START speech_transcribe_word_level_confidence_beta]
    from google.cloud import speech_v1p1beta1 as speech
    client = speech.SpeechClient()

    speech_file = 'resources/voice_tom2.wav'

    with open(speech_file, 'rb') as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        #sample_rate_hertz=16000,
        language_code='th-TH',
        audio_channel_count=2,
        enable_word_confidence=True)

    response = client.recognize(config=config, audio=audio)

    for i, result in enumerate(response.results):
        alternative = result.alternatives[0]
        print('-' * 20)
        print('First alternative of result {}'.format(i))
        print(u'Transcript: {}'.format(alternative.transcript))
        print(u'First Word and Confidence: ({}, {})'.format(
            alternative.words[0].word, alternative.words[0].confidence))
    # [END speech_transcribe_word_level_confidence_beta]

def transcribe_file_with_multiple_channels():
    """Transcribe the given audio file synchronously with multiple channels"""
    # [START speech_transcribe_audio_with_multiple_channels]
    from google.cloud import speech_v1p1beta1 as speech
    client = speech.SpeechClient()

    speech_file = 'resources/voice_tom2.wav'
    
    with open(speech_file, "rb") as audio_file:
        content = audio_file.read()
        
    audio = speech.RecognitionAudio(content=content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        #sample_rate_hertz=44100,
        language_code="th-TH",
        audio_channel_count=2,
        enable_separate_recognition_per_channel=True,
    )
    
    response = client.recognize(config=config, audio=audio)

    for i, result in enumerate(response.results):
        alternative = result.alternatives[0]
        print("-" * 20)
        print("First alternative of result {}".format(i))
        print(u"Transcript: {}".format(alternative.transcript))
        print(u"Channel Tag: {}".format(result.channel_tag))
    # [END speech_transcribe_audio_with_multiple_channels]

def my_transcribe():
    from google.cloud import speech_v1p1beta1 as speech
    client = speech.SpeechClient()

    speech_file = 'resources/voice_tom2.wav'
    # speech_file = 'resources/voice_tom_southern.wav'
    
    with open(speech_file, "rb") as audio_file:
        content = audio_file.read()
        
    audio = speech.RecognitionAudio(content=content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        #sample_rate_hertz=44100,
        language_code="th-TH",
        audio_channel_count=2, # 2 (stereo), 1 (mono)
        enable_word_confidence=True,
        enable_word_time_offsets=True,
        enable_speaker_diarization=True,
        diarization_speaker_count=2,
        model="default",
    )
    
    print("Waiting for operation to complete...")
    response = client.recognize(config=config, audio=audio)

    for i, result in enumerate(response.results):
        alternative = result.alternatives[0]
        print("-" * 30)
        #print(u"Transcript: {}".format(alternative.transcript))
        print("Confidence: {}".format(alternative.confidence))
        print(u"Channel Tag: {}".format(result.channel_tag))
        ground_truth = get_ground_truth_text()
        hypothesis = str(alternative.transcript)
        print("Ground Truth: ", get_ground_truth_text())
        print("Hypothesis: ", hypothesis)

        atta = Tokenizer(model="attacut-sc")
        gt_word_tokenize = atta.tokenize(ground_truth)
        hp_word_tokenize = atta.tokenize(hypothesis)

        # gt_word_tokenize = word_tokenize(ground_truth, engine="newmm") # default=newmm, longest
        # hp_word_tokenize = word_tokenize(hypothesis, engine="newmm")

        print("Ground Truth Word Tokenize:", gt_word_tokenize)
        print("Hypothesis Word Tokenize:", hp_word_tokenize)
        error = evaluation.util.word_error_rate(hp_word_tokenize, gt_word_tokenize)
        print("WER: ", error)
    # [END my_transcribe]

def get_ground_truth_text() -> str:
    ground_truth_words_file = 'resources/gt.txt'
    with open(ground_truth_words_file, "r") as gt_file:
        content = gt_file.read()
        # print("Ground Truth: {}".format(content))
    return content

if __name__ == '__main__':
    stopwatch = Stopwatch()
    stopwatch.start()
    #transcribe_file_with_auto_punctuation()
    #transcribe_file_with_word_level_confidence()
    #transcribe_file_with_multichannel()
    #transcribe_file_with_diarization()
    #transcribe_file_with_multiple_channels()
    my_transcribe()
    stopwatch.stop()

    print("Duration:", str(stopwatch)); 

    #print(get_ground_truth_text())

    # ground_truth_words_file = 'resources/gt.txt'
    
    # with open(ground_truth_words_file, "r") as gt_file:
    #     content = gt_file.read()
    #     print("Ground Truth: {}".format(content))
    #     print("newmm:", word_tokenize(content))

    # gt = " ".join(gts[hash_][speaker])
    # pred = " ".join(preds[hash_][speaker])
    # wer = evaluation.util.word_error_rate(pred, gt)