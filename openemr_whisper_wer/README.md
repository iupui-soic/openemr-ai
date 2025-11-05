# (Whisper Models) Word Error Rate
This repository has code to evaluate the performance of whisper-edge and whisper-web for transcription over a dataset of 
audio files and six simulated patient doctor conversations recorded.

## System Requirements:
* Python version 3.9.6
* Git, Git Hub (To clone the repository)
* pip install -r requirements

## Execution: 

Run
* **calculate_wer.py** and **mean_wer.py** to get WER for whisper-web (base,tiny),whisper edge for simulated patient doctor conversation.
* **wer_edge_kaggle.py**: To get the average word error rate for (**whisper_edge_transcripts_with_wer**).
* **whisper_edge_wer_map.py**: To get the mappings for whisper edge translation errors words (**whisper_edge_transcripts_with_alignment.csv**).
* **batch_ww_base.py**: To get the Word error rate (**dataset_ww_trans_base_with_wer**) and error word mappings for whisper web base.en model (**dataset_ww_trans_base_with_wer_alignment.csv**).
* **batch_ww_tiny.py**: To get the Word error rate (**dataset_ww_trans_tiny_with_wer**) and error word mappings for whisper web tiny.en model (**dataset_ww_trans_tiny_wer_alignment.csv**).

In collab run 
* **whisperlargev3turbo_c6.ipynb**: To get the WER and error word alignment for whisper large v3 turbo
* **wlargev3turbo_k_ds.ipynb**: To get the WER and error word alignment for whisper large v3 turbo

