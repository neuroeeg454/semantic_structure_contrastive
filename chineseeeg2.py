from torch.utils.data import DataLoader, ConcatDataset
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Union, Any
import h5py
import os
import mne
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm
import jieba.posseg as pseg
from task.pos_aware_retrieval_task import map_jieba_to_category
import warnings
warnings.filterwarnings('ignore')

class AlignedEEGTextDataset(Dataset):
    
    def __init__(self,
                 aligned_results_dir: str,
                 eeg_data_dir: str,
                 eeg_base_path: str = "",
                 task_level: str = 'word',
                 modality: str = 'reading',
                 task_type: str = 'retrieval',
                 subject_ids: Optional[List[str]] = None,
                 session_ids: Optional[List[str]] = None,
                 run_ids: Optional[List[str]] = None,
                 fixed_window_ms: int = 300,
                 sampling_rate: int = 1000,
                 text_encoder_name: str = "Qwen/Qwen3-Embedding-0.6B",
                 max_text_length: int = 32,
                 normalize_eeg: bool = True,
                 cache_eeg: bool = True,
                 preload_all: bool = True,
                 selected_channels: Optional[List[int]] = None,
                 ignore_category: Optional[Union[str, List[str]]] = None,
                 **kwargs):

        self.aligned_dir = aligned_results_dir
        self.eeg_dir = eeg_data_dir
        self.eeg_base_path = eeg_base_path
        self.task_level = task_level
        self.modality = modality
        self.task_type = task_type
        self.sampling_rate = sampling_rate
        self.normalize_eeg = normalize_eeg
        self.cache_eeg = cache_eeg
        self.preload_all = preload_all
        self.selected_channels = selected_channels

        self.ignore_categories_set = set()
        if ignore_category:
            if isinstance(ignore_category, str):
                cats = [c.strip() for c in ignore_category.split(',')]
                self.ignore_categories_set = {c.lower() for c in cats if c.strip()}
            elif isinstance(ignore_category, list):
                self.ignore_categories_set = {str(c).lower() for c in ignore_category}
            
            if self.ignore_categories_set:
                self.filtered_count = 0
                self.total_count_before_filter = 0

        self.text_encoder_name = text_encoder_name
        self.max_text_length = max_text_length
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        self.fixed_window_ms = fixed_window_ms
        self.fixed_window_points = int(fixed_window_ms * sampling_rate / 1000)

        self.csv_files = self._collect_csv_files(subject_ids, session_ids, run_ids)
        
        if not self.csv_files:
            raise ValueError("no matching csv files")


        self.h5_cache = {}
        self.begn_offset_cache = {}
        self.cache_size_limit = 100

        self.samples = self._load_all_samples()

        if self.preload_all:
            self._preload_and_preprocess_all_eeg_data()
        else:
            self.eeg_tensor_cache = {}

        self._print_statistics()
    
    def _collect_csv_files(self, subject_ids, session_ids, run_ids):

        csv_files = []
        base_path = Path(self.aligned_dir)
        
        subjects = subject_ids if subject_ids else ["*"]
        sessions = session_ids if session_ids else ["*"]
        runs = run_ids if run_ids else ["*"]
        
        for subject in subjects:
            for session in sessions:
                for run in runs:
                    pattern = f"sub-{subject}/ses-{session}/task-{self.modality}/run-{run}"
                    search_path = base_path / pattern
                    
                    if search_path.exists():
                        for csv_file in search_path.glob("*.csv"):
                            csv_files.append(str(csv_file))
        
        return csv_files
    
    def _load_all_samples(self) -> List[Dict]:

        all_samples = []
        
        for csv_file in self.csv_files:
            try:
                metadata = self._extract_metadata_from_path(csv_file)
                df = pd.read_csv(csv_file)
                sentence_rows = df[df['kind'] == 'sentence']

                if csv_file not in self.begn_offset_cache:
                    begn_offset = 0.0
                    if 'begn_offset' in df.columns:
                        begn_offset = float(df['begn_offset'].iloc[0])
                    self.begn_offset_cache[csv_file] = begn_offset
                
                for _, row in sentence_rows.iterrows():
                    if self.task_level == 'character':
                        char_samples = self._create_character_samples(metadata, row, csv_file)
                        all_samples.extend(char_samples)
                    
                    elif self.task_level == 'word':
                        word_samples = self._create_word_samples(metadata, row, csv_file)
                        all_samples.extend(word_samples)
                    
                    elif self.task_level == 'sentence':
                        sentence_samples = self._create_sentence_samples(metadata, row, csv_file)
                        all_samples.extend(sentence_samples)
            
            except Exception as e:
                continue
        
        return all_samples
    
    def _extract_metadata_from_path(self, csv_path: str) -> Dict:
        import re
        
        pattern = r"sub-(?P<subject>[^/]+)/ses-(?P<session>[^/]+)/task-(?P<task>[^/]+)/run-(?P<run>[^/]+)"
        match = re.search(pattern, csv_path)
        
        if match:
            metadata = match.groupdict()
            metadata['csv_path'] = csv_path
            return metadata
        
        return {
            'subject': 'unknown',
            'session': 'unknown',
            'task': self.modality,
            'run': 'unknown',
            'csv_path': csv_path
        }
    
    def _get_begn_offset(self, csv_path: str) -> float:
        if csv_path in self.begn_offset_cache:
            return self.begn_offset_cache[csv_path]
        
        try:
            df = pd.read_csv(csv_path, nrows=1)
            begn_offset = 0.0
            if 'begn_offset' in df.columns:
                begn_offset = float(df['begn_offset'].iloc[0])
            self.begn_offset_cache[csv_path] = begn_offset
            return begn_offset
        except Exception as e:
            return 0.0
    
    def _create_character_samples(self, metadata: Dict, csv_row: pd.Series, csv_path: str) -> List[Dict]:
        samples = []
        
        sentence_text = csv_row['segment']
        sentence_timestart = csv_row['timestart']
        sentence_audio_start = csv_row['audio_start']

        sentence_chars = json.loads(csv_row['sentence_chars']) if pd.notna(csv_row['sentence_chars']) else []

        
        if not sentence_chars:
            return samples

        begn_offset = self._get_begn_offset(csv_path)
        
        for char_idx, char_info in enumerate(sentence_chars):
            char = char_info['char']
            char_audio_start = char_info['start']
            char_audio_end = char_info['end']
            tone = char_info['tone']
            try:
                tone_int = int(tone)
            except Exception:
                continue
            if tone_int < 1 or tone_int > 5:
                continue

            eeg_char_start = sentence_timestart + (char_audio_start - sentence_audio_start)
            eeg_char_end = sentence_timestart + (char_audio_end - sentence_audio_start)
            char_duration = eeg_char_end - eeg_char_start

            char_duration_ms = char_duration * 1000
            min_char_duration_ms = 0
            max_char_duration_ms = 1000
            
            if (char_duration_ms < min_char_duration_ms or 
                char_duration_ms > max_char_duration_ms or
                char_duration <= 0):
                continue

            sentence_duration = csv_row['duration']
            sentence_end = sentence_timestart + sentence_duration
            
            if eeg_char_start < sentence_timestart:
                eeg_char_start = sentence_timestart
            if eeg_char_end > sentence_end:
                eeg_char_end = sentence_end
            
            char_duration = eeg_char_end - eeg_char_start
            if char_duration <= 0:
                continue
            
            sample = {
                **metadata,
                'text': char,
                'audio_start': char_audio_start,
                'audio_end': char_audio_end,
                'eeg_start': eeg_char_start,
                'eeg_end': eeg_char_end,
                'duration': char_duration,
                'begn_offset': begn_offset,
                'pinyin': tone_int,
                'sentence_text': sentence_text,
                'sentence_start': sentence_timestart,
                'char_index': char_idx,
                'sample_type': 'character'
            }
            samples.append(sample)
        
        return samples
    
    def _create_word_samples(self, metadata: Dict, csv_row: pd.Series, csv_path: str) -> List[Dict]:
        samples = []
        
        sentence_text = csv_row['segment']
        sentence_timestart = csv_row['timestart']
        sentence_audio_start = csv_row['audio_start']
        
        try:
            word_alignments = json.loads(csv_row['word_alignment']) if pd.notna(csv_row['word_alignment']) else []
        except:
            word_alignments = []
        
        if not word_alignments:
            return samples

        begn_offset = self._get_begn_offset(csv_path)
        
        for word_idx, word_info in enumerate(word_alignments):
            word_text = word_info['word']
            word_audio_start = word_info['start']
            word_audio_end = word_info['end']
            word_aligned = word_info.get('aligned', True)
            
            if not word_aligned:
                continue

            eeg_word_start = sentence_timestart + (word_audio_start - sentence_audio_start)
            eeg_word_end = sentence_timestart + (word_audio_end - sentence_audio_start)
            word_duration = eeg_word_end - eeg_word_start
            word_duration_ms = word_duration * 1000
            min_word_duration_ms = 10
            max_word_duration_ms = 2000
            
            if (word_duration_ms < min_word_duration_ms or 
                word_duration_ms > max_word_duration_ms or
                word_duration <= 0):
                continue

            sentence_duration = csv_row['duration']
            sentence_end = sentence_timestart + sentence_duration
            
            if eeg_word_start < sentence_timestart:
                eeg_word_start = sentence_timestart
            if eeg_word_end > sentence_end:
                eeg_word_end = sentence_end
            
            word_duration = eeg_word_end - eeg_word_start
            if word_duration <= 0:
                continue
            
            if self.ignore_categories_set:
                self.total_count_before_filter += 1
                words = list(pseg.cut(word_text))
                if words:
                    cat = map_jieba_to_category(words[0].flag)
                    if cat.lower() in self.ignore_categories_set:
                        self.filtered_count += 1
                        continue
            
            sample = {
                **metadata,
                'text': word_text,
                'audio_start': word_audio_start,
                'audio_end': word_audio_end,
                'eeg_start': eeg_word_start,
                'eeg_end': eeg_word_end,
                'duration': word_duration,
                'begn_offset': begn_offset,
                'word_length': len(word_text),
                'similarity': word_info.get('similarity', 1.0),
                'sentence_text': sentence_text,
                'sentence_start': sentence_timestart,
                'word_index': word_idx,
                'sample_type': 'word'
            }
            samples.append(sample)
        
        return samples
    
    def _create_sentence_samples(self, metadata: Dict, csv_row: pd.Series, csv_path: str) -> List[Dict]:
        samples = []
        
        sentence_text = csv_row['segment']
        if pd.isna(sentence_text) or str(sentence_text).strip() == "":
            return samples

        sentence_timestart = csv_row['timestart']
        sentence_duration = csv_row['duration']
        sentence_end = sentence_timestart + sentence_duration

        begn_offset = self._get_begn_offset(csv_path)

        window_sec = self.fixed_window_ms / 1000
        if sentence_duration <= window_sec:

            padding_before = (window_sec - sentence_duration) / 2
            padding_after = window_sec - sentence_duration - padding_before
            
            window_start = sentence_timestart - padding_before
            window_end = sentence_end + padding_after
            strategy = 'padded'
        else:
            middle_point = sentence_timestart + sentence_duration / 2
            window_start = middle_point - (window_sec / 2)
            window_end = middle_point + (window_sec / 2)
            
            if window_start < sentence_timestart:
                window_start = sentence_timestart
                window_end = window_start + window_sec
            elif window_end > sentence_end:
                window_end = sentence_end
                window_start = window_end - window_sec
            
            strategy = 'center_crop'
        
        sample = {
            **metadata,
            'text': sentence_text,
            'eeg_start': window_start,
            'eeg_end': window_end,
            'duration': window_end - window_start,
            'begn_offset': begn_offset,
            'sentence_text': sentence_text,
            'sentence_start': sentence_timestart,
            'sentence_duration': sentence_duration,
            'strategy': strategy,
            'sample_type': 'sentence'
        }
        
        samples.append(sample)
        
        return samples
    
    def _preload_and_preprocess_all_eeg_data(self):

        run_groups = {}
        for idx, sample in enumerate(tqdm(self.samples, desc="组织样本")):
            run_key = self._get_run_key(sample)
            if run_key not in run_groups:
                run_groups[run_key] = {
                    'samples': []
                }
            run_groups[run_key]['samples'].append((idx, sample))

        self.preloaded_eeg_tensors = [None] * len(self.samples)
        

        for run_key, run_info in tqdm(run_groups.items(), desc="loading EEG data"):
            try:

                h5_data = self._load_h5_eeg(run_info['samples'][0][1])
                if h5_data is None:
                    continue
                
                eeg_data, sfreq = h5_data

                samples_processed = 0
                for idx, sample in run_info['samples']:

                    eeg_tensor = self._extract_and_preprocess_eeg(eeg_data, sfreq, sample)

                    self.preloaded_eeg_tensors[idx] = eeg_tensor
                    samples_processed += 1
                
            except Exception as e:
                continue
        
        processed_count = len([t for t in self.preloaded_eeg_tensors if t is not None])
        print(f"EEG数据预加载完成，预处理了 {processed_count} 个样本的EEG数据")
    
    def _extract_and_preprocess_eeg(self, eeg_data: np.ndarray, sfreq: float, sample: Dict) -> torch.Tensor:

        try:
            begn_offset = sample.get('begn_offset', 0.0)
            adjusted_start_time = sample['eeg_start'] + begn_offset
            adjusted_end_time = sample['eeg_end'] + begn_offset
            start_sample = int(adjusted_start_time * sfreq)
            end_sample = int(adjusted_end_time * sfreq)
            start_sample = max(0, start_sample)
            end_sample = min(eeg_data.shape[1], end_sample)
            
            if start_sample >= end_sample:

                n_channels = eeg_data.shape[0]
                n_samples_target = int((adjusted_end_time - adjusted_start_time) * sfreq)
                extracted_eeg = np.zeros((n_channels, n_samples_target))

            else:

                extracted_eeg = eeg_data[:, start_sample:end_sample]
                expected_samples = int((adjusted_end_time - adjusted_start_time) * sfreq)
                if extracted_eeg.shape[1] < expected_samples:

                    padding = np.zeros((extracted_eeg.shape[0], expected_samples - extracted_eeg.shape[1]))
                    extracted_eeg = np.concatenate([extracted_eeg, padding], axis=1)

                elif extracted_eeg.shape[1] > expected_samples:

                    extracted_eeg = extracted_eeg[:, :expected_samples]

                else:

                    pass

            if sfreq != self.sampling_rate:
                from scipy import signal
                resample_factor = self.sampling_rate / sfreq
                n_samples_new = int(extracted_eeg.shape[1] * resample_factor)

                eeg_resampled = np.zeros((extracted_eeg.shape[0], n_samples_new))
                for ch_idx in range(extracted_eeg.shape[0]):
                    eeg_resampled[ch_idx, :] = signal.resample(
                        extracted_eeg[ch_idx, :], n_samples_new
                    )
                extracted_eeg = eeg_resampled
            if self.normalize_eeg:

                mean = np.mean(extracted_eeg, axis=1, keepdims=True)
                std = np.std(extracted_eeg, axis=1, keepdims=True) + 1e-8
                extracted_eeg = (extracted_eeg - mean) / std
            eeg_tensor = self._preprocess_to_fixed_window(extracted_eeg)
            if self.selected_channels is not None:
                eeg_tensor = eeg_tensor[self.selected_channels, :]
                
            return eeg_tensor
            
        except Exception as e:
            n_out_channels = len(self.selected_channels) if self.selected_channels is not None else 128
            return torch.zeros((n_out_channels, self.fixed_window_points))
    
    def _preprocess_to_fixed_window(self, eeg_data: np.ndarray) -> torch.Tensor:
        """
        Args:
            eeg_data: EEG data [n_channels, n_timepoints]
            
        Returns:
            torch.Tensor: processed EEG tensor [channels, fixed_window_points]
        """
        n_channels, n_timepoints = eeg_data.shape
        
        if n_timepoints < self.fixed_window_points:
            # padding
            pad_points = self.fixed_window_points - n_timepoints
            pad_data = np.zeros((n_channels, pad_points))
            eeg_data = np.concatenate([eeg_data, pad_data], axis=1)
        elif n_timepoints > self.fixed_window_points:
            # central cutting
            start_idx = (n_timepoints - self.fixed_window_points) // 2
            eeg_data = eeg_data[:, start_idx:start_idx + self.fixed_window_points]

        return torch.FloatTensor(eeg_data)
    
    def _get_run_key(self, sample: Dict) -> str:
        return f"{sample['subject']}_{sample['session']}_{sample['run']}"
    
    def _load_h5_eeg(self, sample: Dict):
        """
        Args:
            sample: 包含元数据的样本字典
            
        Returns:
            tuple: (eeg_data, sfreq) 或 None
        """
        try:
            subject = sample['subject']
            session = sample['session']
            task = sample['task']
            run = sample['run']
            h5_file_base = f"sub-{subject}_ses-{session}_task-{task}_run-{run}_eeg"
            h5_relative_path = f"sub-{subject}/ses-{session}/eeg/{h5_file_base}.h5"
            if self.eeg_base_path:
                h5_path = os.path.join(self.eeg_base_path, h5_relative_path)
            else:
                h5_path = os.path.join(self.eeg_dir, h5_relative_path)}
            
            if not os.path.exists(h5_path):
                alt_paths = [
                    os.path.path.join(self.eeg_base_path if self.eeg_base_path else self.eeg_dir, 
                                f"sub-{subject}/ses-{session}/{h5_file_base}.h5"),
                    os.path.path.join(self.eeg_base_path if self.eeg_base_path else self.eeg_dir,
                                f"{subject}/{session}/{h5_file_base}.h5")
                ]
                
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        h5_path = alt_path
                        break
                else:
                    return None
            else:
                pass
            if h5_path in self.h5_cache:
                h5f = self.h5_cache[h5_path]
            else:
                h5f = h5py.File(h5_path, 'r')
                self.h5_cache[h5_path] = h5f
                h5f = self.h5_cache[h5_path]

            eeg_data = h5f['eeg_data'][:]  # [n_channels, n_samples]
            sfreq = h5f['info'].attrs.get('sfreq', 1000.0)  # 默认1000Hz
            
            return eeg_data, sfreq
            
        except Exception as e:
            return None
    
    def _load_eeg_data_on_demand(self, sample: Dict) -> torch.Tensor:

        cache_key = f"{sample['subject']}_{sample['session']}_{sample['run']}_{sample['eeg_start']:.3f}_{sample['eeg_end']:.3f}"

        if cache_key in self.eeg_tensor_cache:
            return self.eeg_tensor_cache[cache_key]
        
        try:
            h5_data = self._load_h5_eeg(sample)
            if h5_data is None:
                return torch.zeros((128, self.fixed_window_points))
            
            eeg_data, sfreq = h5_data
            eeg_tensor = self._extract_and_preprocess_eeg(eeg_data, sfreq, sample)
            if self.cache_eeg:
                self.eeg_tensor_cache[cache_key] = eeg_tensor
                if len(self.eeg_tensor_cache) > self.cache_size_limit * 10:
                    keys = list(self.eeg_tensor_cache.keys())
                    for key in keys[:len(keys)//2]:
                        del self.eeg_tensor_cache[key]
            
            return eeg_tensor
            
        except Exception as e:
            n_out_channels = len(self.selected_channels) if self.selected_channels is not None else 128
            return torch.zeros((n_out_channels, self.fixed_window_points))
    
    def _preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:

        encoded = self.tokenizer(
            text,
            max_length=self.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {k: v.squeeze(0) for k, v in encoded.items()}
    
    def _print_statistics(self):
        if self.task_level == 'character':
            char_durations = [s['duration'] for s in self.samples]
            chars = [s['text'] for s in self.samples]
            unique_chars = set(chars)
        elif self.task_level == 'word':
            word_durations = [s['duration'] for s in self.samples]
            words = [s['text'] for s in self.samples]
            unique_words = set(words)
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.preload_all:
            eeg_tensor = self.preloaded_eeg_tensors[idx]
            if eeg_tensor is None:
                n_out_channels = len(self.selected_channels) if self.selected_channels is not None else 128
                eeg_tensor = torch.zeros((n_out_channels, self.fixed_window_points))
        else:
            eeg_tensor = self._load_eeg_data_on_demand(sample)
        if self.task_type == 'tone':
            # ========== Tonetask：using pinyin instead of text ==========

            pinyin_tone = sample.get('pinyin', -1)
            pinyin_tone = int(pinyin_tone)
            pinyin_tone = pinyin_tone - 1

            text = str(pinyin_tone) if pinyin_tone != -1 else "0"
            text_data = {
                'input_ids': torch.tensor([pinyin_tone], dtype=torch.long),
                'attention_mask': torch.tensor([1], dtype=torch.long)
            }

        else:

            text = sample['text']
            text_data = self._preprocess_text(text)

        return {
            'eeg': eeg_tensor,  # [channels, timepoints]
            'text': text_data,
            'text_str': text,
            'sample_info': sample,
            'task_level': self.task_level
        }
    
    def __del__(self):
        for h5f in self.h5_cache.values():
            try:
                h5f.close()
            except:
                pass


class EEGTextAlignmentDataModule:
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def _get_default_text_encoder_name(self, config: Dict[str, Any]) -> str:
        text_encoder_type = config.get('text_encoder', {}).get('text_encoder_type', 'qwen')
        
        if text_encoder_type == 'qwen':
            return "Qwen/Qwen3-Embedding-0.6B"
        elif text_encoder_type == 'hybrid':
            return "hybrid"
        else:
            return "Qwen/Qwen3-Embedding-0.6B"
    
    def create_dataloaders(self, batch_size: int = 32, num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
        eeg_processing = self.config.get('eeg_processing', {})
        fixed_window_ms = eeg_processing.get('fixed_window_ms', self.config.get('fixed_window_ms', 300))
        sampling_rate = eeg_processing.get('sampling_rate', self.config.get('sampling_rate', 1000))
        print(f"Dataset Config: fixed_window_ms={fixed_window_ms}, sampling_rate={sampling_rate}")

        def alignment_collate_fn(batch):
            eeg_list = [item['eeg'] for item in batch]
            eeg_batch = torch.stack(eeg_list, dim=0)  # [batch, channels, timepoints]

            text_keys = ['input_ids', 'attention_mask']
            text_data = {}
            for key in text_keys:
                if key in batch[0]['text']:
                    text_data[key] = torch.stack([item['text'][key] for item in batch])
            text_str_list = [item['text_str'] for item in batch]
            task_level = batch[0]['task_level']
            
            return {
                'eeg': eeg_batch,
                'text': text_data,
                'text_str': text_str_list,
                'task_level': task_level
            }
        

        train_dataset = AlignedEEGTextDataset(
            aligned_results_dir=self.config.get('aligned_results_dir'),
            eeg_data_dir=self.config.get('eeg_data_dir'),
            task_level=self.config.get('task_level', 'word'),
            modality=self.config.get('modality', 'reading'),
            task_type=self.config.get('task_type', 'retrieval'),
            subject_ids=self.config.get('train_subjects', ['f1']),
            session_ids=self.config.get('train_sessions', ['littleprince']),
            run_ids=self.config.get('train_runs', ['11']),
            fixed_window_ms=fixed_window_ms,
            sampling_rate=sampling_rate,
            text_encoder_name=self.config.get('text_encoder_name', self._get_default_text_encoder_name(self.config)),
            max_text_length=self.config.get('max_text_length', 32),
            normalize_eeg=self.config.get('normalize_eeg', True),
            cache_eeg=self.config.get('cache_eeg', True),
            preload_all=self.config.get('preload_all', True),
            selected_channels=self.config.get('eeg_processing', {}).get('selected_channels', self.config.get('selected_channels', None)),
            ignore_category=self.config.get('ignore_category', None)
        )
        
        # 验证数据集
        val_dataset = AlignedEEGTextDataset(
            aligned_results_dir=self.config.get('aligned_results_dir'),
            eeg_data_dir=self.config.get('eeg_data_dir'),
            task_level=self.config.get('task_level', 'word'),
            modality=self.config.get('modality', 'reading'),
            task_type=self.config.get('task_type', 'retrieval'),
            subject_ids=self.config.get('val_subjects', ['f1']),
            session_ids=self.config.get('val_sessions', ['littleprince']),
            run_ids=self.config.get('val_runs', ['21']),
            fixed_window_ms=fixed_window_ms,
            sampling_rate=sampling_rate,
            text_encoder_name=self.config.get('text_encoder_name', self._get_default_text_encoder_name(self.config)),
            max_text_length=self.config.get('max_text_length', 32),
            normalize_eeg=self.config.get('normalize_eeg', True),
            cache_eeg=self.config.get('cache_eeg', True),
            preload_all=self.config.get('preload_all', True),
            selected_channels=self.config.get('eeg_processing', {}).get('selected_channels', self.config.get('selected_channels', None)),
            ignore_category=self.config.get('ignore_category', None)
        )
        
        # 测试数据集
        test_dataset = AlignedEEGTextDataset(
            aligned_results_dir=self.config.get('aligned_results_dir'),
            eeg_data_dir=self.config.get('eeg_data_dir'),
            task_level=self.config.get('task_level', 'word'),
            modality=self.config.get('modality', 'reading'),
            task_type=self.config.get('task_type', 'retrieval'),
            subject_ids=self.config.get('test_subjects', ['f1']),
            session_ids=self.config.get('test_sessions', ['littleprince']),
            run_ids=self.config.get('test_runs', ['23']),
            fixed_window_ms=fixed_window_ms,
            sampling_rate=sampling_rate,
            text_encoder_name=self.config.get('text_encoder_name', self._get_default_text_encoder_name(self.config)),
            max_text_length=self.config.get('max_text_length', 32),
            normalize_eeg=self.config.get('normalize_eeg', True),
            cache_eeg=self.config.get('cache_eeg', True),
            preload_all=self.config.get('preload_all', True),
            selected_channels=self.config.get('eeg_processing', {}).get('selected_channels', self.config.get('selected_channels', None)),
            ignore_category=self.config.get('ignore_category', None)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=alignment_collate_fn,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=alignment_collate_fn,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=alignment_collate_fn,
            drop_last=False
        )
        
        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    test_alignment_dataset()
