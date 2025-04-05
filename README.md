# 📚 Table of Contents

1. [Introduction](#introduction)
2. [Task 1: Semantic Chunking of a YouTube Video](#task-1-semantic-chunking-of-a-youtube-video)
   - [Problem Statement](#problem-statement)
   - [Judgement Criteria](#judgement-criteria)
   - [My Approach](#my-approach)
     - [Model Choice: openai/whisper-large-v3](#model-choice-openaiwhisper-large-v3)
     - [Problems and Limitations Identified](#problems-and-limitations-identified)
     - [Technique used to mitigate](#technique-used-to-mitigate)
     - [Transcription, Alignment and Semantic Chunking](#transcription-alignment-and-semantic-chunking)
     - [Methodology](#methodology)
     - [Strengths and Weaknesses](#strengths-and-weaknesses-of-the-approach)
     - [Future Improvements](#future-improvements)
   - [Bonus 2: Utilizing Ground-Truth Transcripts](#utilizing-ground-truth-transcripts)

3. [Task 2: EDA of New Testament Audio and Text](#task-2-exploratory-data-analysis-of-new-testament-audio-and-text)
   - [Problem Statement](#problem-statement-1)
   - [Judgement Criteria](#judgement-criteria-1)



# Speech-Team-Hiring-Task---SARVAM.AI
This was the task done by me for the position of Summer Intern in the Speech Team of Sarvam.ai. Though they haven't replied or seen the task ever :~)

The uploaded IPYNB file contains the task descriptions and a detailed implementation of all my work, including thorough comments and markdowns to explain the code as well as the logic behind it.


# Task 1: Semantic Chunking of a Youtube Video

**Problem Statement:**

The objective is to extract high-quality, meaningful (semantic) segments from the specified YouTube video: [Watch Video](https://www.youtube.com/watch?v=Sby1uJ_NFIY).

Suggested workflow:
1. **Download Video and Extract Audio:** Download the video and separate the audio component.
2. **Transcription of Audio:** Utilize an open-source Speech-to-Text model to transcribe the audio. *Provide an explanation of the chosen model and any techniques used to enhance the quality of the transcription.*
3. **Time-Align Transcript with Audio:** *Describe the methodology and steps for aligning the transcript with the audio.*
4. **Semantic Chunking of Data:** Slice the data into audio-text pairs, using both semantic information from the text and voice activity information from the audio, with each audio-chunk being less than 15s in length. *Explain the logic used for semantic chunking and discuss the strengths and weaknesses of your approach.*

**Judgement Criteria:**

1. **Precision-Oriented Evaluation:** The evaluation focuses on precision rather than recall. Higher scores are achieved by reporting fewer but more accurate segments rather than a larger number of segments with inaccuracies. Segment accuracy is determined by:
   - **Transcription Quality:** Accuracy of the text transcription for each audio chunk.
   - **Segment Quality:** Semantic richness of the text segments.
   - **Timestamp Accuracy:** Precision of the start and end times for each segment. Avoid audio cuts at the start or end of a segment.
   
2. **Detailed Explanations:** Provide reasoning behind each step in the process.
3. **Generalization:** Discuss the general applicability of your approach, potential failure modes on different types of videos, and adaptation strategies for other languages.
4. **[Bonus-1]** **Gradio-app Interface:** Wrap your code in a gradio-app which takes in youtube link as input and displays the output in a text-box.
5. **[Bonus-2]** **Utilizing Ground-Truth Transcripts:** Propose a method to improve the quality of your transcript using a ground-truth transcript provided as a single text string. Explain your hypothesis for this approach. *Note that code-snippet isn't required for this question.*

  As an example - for the audio extracted from [yt-link](https://www.youtube.com/watch?v=ysLiABvVos8), how can we leverage transcript scraped from [here](https://www.newsonair.gov.in/bulletins-detail/english-morning-news-7/), to improve the overall transcription quality of segments?
  
---

# Task 2: Exploratory Data Analysis of New Testament Audio and Text

**Problem Statement:**

The objective of this task is to conduct a comprehensive exploratory data analysis (EDA) on the audio and text data of the 260 chapters of the New Testament in your mother tongue (excluding English). The data should be obtained through web scraping from [Faith Comes By Hearing](https://www.faithcomesbyhearing.com/).

The workflow for this task should include:
1. **Web Scraping:** Systematically download the audio files and their corresponding textual content for each of the 260 chapters of the New Testament from the specified website.
2. **Data Preparation:** Organize the data by chapters, ensuring each audio file is matched with its corresponding text.
3. **Exploratory Data Analysis:** Analyze the data to uncover patterns, anomalies, or insights that could benefit applications such as Text to Speech (TTS) and Speech to Text (STT) technologies. Your analysis should explore various facets of the data, including audio quality, text clarity, and alignment between text and spoken content.

**Judgement Criteria:**

Your submission will be evaluated based on:
- **Efficiency and Reliability of Web Scraping Techniques:** The methods and tools used for downloading the chapters efficiently and reliably.
- **Data Analysis Methods:** The techniques and approaches used for analyzing the audio and text data.
- **Quality of Data Analysis:** How effectively the analysis addresses potential applications for the Speech team, including TTS and STT technologies.
- **Creativity in Analysis:** Innovative approaches in data handling and analysis, and the use of relevant metrics to assess data quality and applicability.
---
# My Approach : 


### **Model Choice**: [openai/whisper-large-v3](https://github.com/openai/whisper)

### **Reasons**:


1.   Whisper models, which have been trained on 680k hours of labelled data, show a strong ability to generalise to a wide range of datasets and domains without requiring fine tuning.
2.  Over many existing ASR systems(For example: Wav2Vec2), the whisper models exhibit improved robustness to accents, background noise, technical language, as well as zero shot translation from multiple languages into English; and that accuracy on speech recognition and translation is near the state-of-the-art level.
3. Support for multiple Languages.
4. Inbuilt capability of producing word-level timestamps.

### **Problems and Limitations Identified**:

* **Hallucinations**: Because the models are trained in a weakly supervised manner using large-scale noisy data, the predictions may include texts that are not actually spoken in the audio input (i.e. hallucination).
* **Somewhat inaccurate timestamp Prediction**: The model, explicitly near non-speech segments produce a off the mark timestamps of words. For example: I observed that the model predicted the timestamps of the word including time when there was only applauses or music.

### **Technique used to mitigate:**
* **Masking of Non-Speech-segments in the audio**: <br>
 * I have used **pyannote/segmentation-3.0 model** for Voice Activity Detection to **filter out non-speech-segment** in the audio (applauses, silence, only music in the background etc.). <br>
 * After filtering out the non-speech segments, I **masked those segments** of the audio. <br>
 * This could **potentially avoid confusing the model into predicting texts that were not spoken(hallucinations)** also, this would **segment out the speech and non speech regions effectively for the model to accurately predict the timestamps for words spoken near the non-speech segments**.



## **Transcription, Allignment and Semantic Chunking**

#### **The Workflow**
1. Download the audio
2. Convert it into the right format.
3. Get timestamps corresponding to non-speech-segments using [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0).
4. Apply mask in the non-speech segments of the audio (using output of 3.).
5. Get transcription as well as the word-level timestamps from the model.
6. Form sentence level timestamps using the word-level timestamps keeping in mind the max_duration of 15 seconds (i.e. each sentence has duration less than 15 seconds).
7. Get embeddings corresponding to each sentence using hugging face embeddings.
8. Calculate cosine similarity and thus calculate cosine-distance between **sentence[i]** and **sentence[i+1].**
9. Create Chunks using a threshold cosine-distance and max duration limits along with non_speech_segment information. More details in Methodology section.


### **Methodology**

**1. Download and Convert Audio:**
The first step involves downloading the audio from the video url and converting it into a suitable format for further processing, typically .mp3 file.

**2. Non-Speech Segment Detection:**
Using `pyannote/segmentation-3.0`, we identify non-speech segments in the audio. This model provides timestamps for segments where no speech is detected. These segments are crucial for our chunking process as they naturally delineate potential boundaries between meaningful segments of speech.

**3. Masking Non-Speech Segments:**
We use the timestamps from the previous step to mask out non-speech segments in the audio. This helps in improving transcription and allignment quality by
1. **Reducing Noise**: Eliminates background noise, reducing false positives.
2. **Improving Focus**: Ensures the model processes only relevant speech, enhancing recognition.

**4. Transcription and Word-Level Timestamps:**
We use a Speech-to-Text [openai/whisper-large-v3](https://github.com/openai/whisper) model to transcribe the audio and get word-level timestamps.

**5. Sentence-Level Timestamps:**
Using the word-level timestamps, we aggregate them into sentence-level timestamps. While forming these sentences, we ensure that each sentence's duration does not exceed 15 seconds. This involves detecting sentence boundaries or if the sentence reaches max duration without violating 15 seconds limit.

**6. Sentence Embeddings:**
Each sentence is converted into an embedding using a model from Hugging Face (e.g., `sentence-transformers`). Sentence embeddings capture the semantic meaning of the sentences in a high-dimensional space, allowing for similarity comparisons.

**7. Cosine Similarity and Distance Calculation:**
We calculate the cosine similarity between the embeddings of consecutive sentences. The cosine distance (1 - cosine similarity) indicates how different two sentences are.

**8. Chunk Creation:**
Using the cosine distance and non-speech segment information, we create audio-text pairs. The logic for chunking is as follows:
- **Threshold Cosine Distance:** If the cosine distance between two sentences exceeds a certain threshold, it suggests a significant semantic shift, and thus a new chunk boundary is created.(Chunk Threshold is set at 95th percentile value in all the cosine-distances.)
- **Max Duration Limit:** Ensure that no chunk exceeds 15 seconds in length, even if the semantic similarity suggests they could be grouped together.
- **Non-Speech Segments:** Prioritize non-speech segments as natural chunk boundaries to maintain coherence and improve the quality of the chunks.

### **Strengths and Weaknesses of the Approach**

**Strengths:**
1. **Semantic Awareness:** Using sentence embeddings and cosine similarity ensures that the chunks are semantically meaningful, enhancing the quality of the extracted segments.
2. **Natural Boundaries:** Incorporating non-speech segments helps in creating natural and coherent chunks.
3. **Flexibility:** The approach is flexible and can be adapted to different languages and domains by changing the underlying models for transcription and embedding.

**Weaknesses:**
1. **Dependency on Model Accuracy:** The quality of transcription and embedding models directly impacts the effectiveness of the chunking. Inaccurate transcriptions and allignment effects the quality of the chunk. Whilst whisper models do produce highly accurate transcriptions, the corresponding timestamps are at the utterance-level, not per word, and can be inaccurate sometimes. This is because it uses Dynamic Time Warping underneath.
- **Note:** [openai/whisper-large-v3](https://github.com/openai/whisper) has the capability to allign the transcript at word level by passing the return_timestamps = 'word' in the pipeline. Whisper's internal processor (WhisperTimeStampLogitsProcessor) analyzes the model's own predictions (words and their probabilities) to infer word boundaries and introduce timestamps.
It leverages techniques like attention mechanisms and learned boundary markers within the model's output itself.<br>
- **NOTE:** See future improvements section for more deatils on how to mitigate this problem with Whisper.

2. **Threshold Sensitivity:** The choice of cosine distance threshold is crucial and may require fine-tuning for different types of content to balance between over-segmentation and under-segmentation.
3. **Edge Cases:** In cases where speech is continuous without clear non-speech segments, or presence of several filler words(ummmm, hmmm, grunts etc.) continuously, the method might struggle to create meaningful chunks without exceeding the duration limit.


**Future Improvements:**
1. **Inspirations from WhisperX:** Phoneme-Based ASR are a suite of models finetuned to recognise the smallest unit of speech distinguishing one word from another, e.g. the element p in "tap".
 - Popular Example is Wav2Vec2 2.0.
 - WhisperX is a library built on top of Whisper that utilizes forced alignment for improved timestamp accuracy.
 - It combines Whisper's predictions with a separate phoneme recognition model (e.g., wav2vec 2.0) to achieve this.
 - Phoneme recognition models identify individual sounds (phonemes) in speech.
 - By aligning these phonemes with the audio and the predicted words, WhisperX refines the timestamps.


## **Utilizing Ground-Truth Transcripts:**
Since a ground-truth transcript is provided as a single text, we can fine tune the model in a supervised way.

**Methodology:** <br>
1. Firstly we could divide both the transcribed audio (from the model) and the ground-truth transcript into smaller segments by aligning them into smaller sentences/phrases.
2. For each segment, we could compare the model's transcription segment with the ground-truth segment and identify differences between the segments such as identifying common types of errors (e.g., specific misrecognized words, missed context).
3. Next we collect these pairs of segments: <br> (model's transcript segment,corresponding ground truth segment) <br> as training dataset with model's transcript segment serving as input and corresponding ground truth segment serving as target.
4. Next we train the model, adjusting the model parameters to reduce the errors identified. Hence, the model is fined-tuned in a supervised manner.
5. We should also create a validation dataset so as to evaluate the model.

One example for this could be the following : I have observed the model transcript the "OpenHathi" part as "OpenHearty" and "sarvam.ai" part as "severon.ai".

e.g.
```
Example-1:
Model's Transcription Segement (input):" We  wanted  to  start  with  playing  a  video  of  what  **OpenHearty**  does."
Corresponding Ground truth (target):" We  wanted  to  start  with  playing  a  video  of  what  **OpenHathi**  does."

Example-2:
Model's Transcription Segement (input):" I  encourage  all  of  you  to  go  to  the  website  **serveron.ai**  and  check  it  out."
Corresponding Ground truth (target):" I  encourage  all  of  you  to  go  to  the  website  **sarvam.ai**  and  check  it  out."
```

