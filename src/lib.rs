use std::time::Duration;
use rodio::source::SeekError;
use rodio::Source;

fn get_symmetric_hanning_window(window_length: usize) -> Vec<f32> {
    let mut window = vec![0.0_f32; window_length];
    let scale = 2.0 * std::f32::consts::PI / window_length as f32;
    for n in 0..window_length {
        window[n] = 0.5 * (1.0 - (n as f32 * scale).cos());
    }
    window
}

fn dot_product_multi(
    a: &[Vec<f32>],
    frame_offset_a: usize,
    b: &[Vec<f32>],
    frame_offset_b: usize,
    channels: usize,
    num_frames: usize,
    dot_product: &mut [f32],
) {
    for k in 0..channels {
        let ch_a = &a[k][frame_offset_a..frame_offset_a + num_frames];
        let ch_b = &b[k][frame_offset_b..frame_offset_b + num_frames];
        let mut sum = 0.0_f32;
        for n in 0..num_frames {
            sum += ch_a[n] * ch_b[n];
        }
        dot_product[k] = sum;
    }
}

fn similarity_measure(
    dot_prod: &[f32],
    energy_target: &[f32],
    energy_candidate: &[f32],
    channels: usize,
) -> f32 {
    let epsilon = 1e-12_f32;
    let mut sum = 0.0_f32;
    for n in 0..channels {
        sum += dot_prod[n] * energy_target[n]
            / (energy_target[n] * energy_candidate[n] + epsilon).sqrt();
    }
    sum
}

fn quadratic_interpolation(
    y_values: &[f32; 3],
    extremum: &mut f32,
    extremum_value: &mut f32,
) {
    let a = 0.5 * (y_values[2] + y_values[0]) - y_values[1];
    let b = 0.5 * (y_values[2] - y_values[0]);
    let c = y_values[1];

    if a == 0.0 {
        *extremum = 0.0;
        *extremum_value = y_values[1];
    } else {
        *extremum = -b / (2.0 * a);
        *extremum_value = a * (*extremum) * (*extremum) + b * (*extremum) + c;
    }
}

fn decimated_search(
    decimation: usize,
    exclude_interval: (isize, isize),
    target_block: &[Vec<f32>],
    target_block_frames: usize,
    search_segment: &[Vec<f32>],
    search_segment_frames: usize,
    channels: usize,
    energy_target_block: &[f32],
    energy_candidate_blocks: &[f32],
    dot_prod: &mut [f32],
) -> usize {
    let num_candidate_blocks = search_segment_frames - (target_block_frames - 1);
    let mut similarity = [0.0_f32; 3];

    let mut n = 0;
    dot_product_multi(
        target_block, 0,
        search_segment, n,
        channels,
        target_block_frames,
        dot_prod,
    );
    similarity[0] = similarity_measure(
        dot_prod,
        energy_target_block,
        &energy_candidate_blocks[0..channels],
        channels,
    );

    let mut best_similarity = similarity[0];
    let mut optimal_index = 0;

    n += decimation;
    if n >= num_candidate_blocks {
        return 0;
    }

    dot_product_multi(
        target_block, 0,
        search_segment, n,
        channels,
        target_block_frames,
        dot_prod,
    );
    similarity[1] = similarity_measure(
        dot_prod,
        energy_target_block,
        &energy_candidate_blocks[n * channels..(n + 1) * channels],
        channels,
    );

    n += decimation;
    if n >= num_candidate_blocks {
        return if similarity[1] > similarity[0] { decimation.min(num_candidate_blocks - 1) } else { 0 };
    }

    while n < num_candidate_blocks {
        dot_product_multi(
            target_block, 0,
            search_segment, n,
            channels,
            target_block_frames,
            dot_prod,
        );
        similarity[2] = similarity_measure(
            dot_prod,
            energy_target_block,
            &energy_candidate_blocks[n * channels..(n + 1) * channels],
            channels,
        );

        if (similarity[1] > similarity[0] && similarity[1] >= similarity[2]) ||
           (similarity[1] >= similarity[0] && similarity[1] > similarity[2])
        {
            let mut normalized_candidate_index = 0.0_f32;
            let mut candidate_similarity = 0.0_f32;
            quadratic_interpolation(&similarity, &mut normalized_candidate_index, &mut candidate_similarity);

            let candidate_index = (n - decimation) as isize
                + (normalized_candidate_index * decimation as f32 + 0.5).floor() as isize;
            
            let in_exclude = candidate_index >= exclude_interval.0 && candidate_index <= exclude_interval.1;
            if candidate_similarity > best_similarity && !in_exclude {
                optimal_index = (candidate_index.max(0) as usize).min(num_candidate_blocks - 1);
                best_similarity = candidate_similarity;
            }
        } else if n + decimation >= num_candidate_blocks {
            let in_exclude = (n as isize) >= exclude_interval.0 && (n as isize) <= exclude_interval.1;
            if similarity[2] > best_similarity && !in_exclude {
                optimal_index = n.min(num_candidate_blocks - 1);
                best_similarity = similarity[2];
            }
        }

        similarity[0] = similarity[1];
        similarity[1] = similarity[2];
        n += decimation;
    }

    optimal_index.min(num_candidate_blocks - 1)
}

fn full_search(
    low_limit: usize,
    high_limit: usize,
    exclude_interval: (isize, isize),
    target_block: &[Vec<f32>],
    target_block_frames: usize,
    search_block: &[Vec<f32>],
    _search_block_frames: usize,
    channels: usize,
    energy_target_block: &[f32],
    energy_candidate_blocks: &[f32],
    dot_prod: &mut [f32],
) -> usize {
    let mut best_similarity = -f32::MAX;
    let mut optimal_index = 0;

    for n in low_limit..=high_limit {
        let n_isize = n as isize;
        if n_isize >= exclude_interval.0 && n_isize <= exclude_interval.1 {
            continue;
        }

        dot_product_multi(
            target_block, 0,
            search_block, n,
            channels,
            target_block_frames,
            dot_prod,
        );

        let similarity = similarity_measure(
            dot_prod,
            energy_target_block,
            &energy_candidate_blocks[n * channels..(n + 1) * channels],
            channels,
        );

        if similarity > best_similarity {
            best_similarity = similarity;
            optimal_index = n;
        }
    }

    optimal_index
}

fn compute_optimal_index(
    search_block: &[Vec<f32>],
    search_block_frames: usize,
    target_block: &[Vec<f32>],
    target_block_frames: usize,
    energy_candidate_blocks: &mut [f32],
    channels: usize,
    exclude_interval: (isize, isize),
    energy_target_block: &mut [f32],
    dot_prod: &mut [f32],
) -> usize {
    let num_candidate_blocks = search_block_frames - (target_block_frames - 1);
    let search_decimation = 5;

    multi_channel_moving_block_energies(
        search_block,
        channels,
        target_block_frames,
        energy_candidate_blocks,
    );

    dot_product_multi(
        target_block, 0,
        target_block, 0,
        channels,
        target_block_frames,
        energy_target_block,
    );

    let optimal_index = decimated_search(
        search_decimation,
        exclude_interval,
        target_block,
        target_block_frames,
        search_block,
        search_block_frames,
        channels,
        energy_target_block,
        energy_candidate_blocks,
        dot_prod,
    );

    let lim_low = optimal_index.saturating_sub(search_decimation);
    let lim_high = (optimal_index + search_decimation).min(num_candidate_blocks - 1);

    full_search(
        lim_low,
        lim_high,
        exclude_interval,
        target_block,
        target_block_frames,
        search_block,
        search_block_frames,
        channels,
        energy_target_block,
        energy_candidate_blocks,
        dot_prod,
    )
}

fn multi_channel_moving_block_energies(
    input: &[Vec<f32>],
    channels: usize,
    frames_per_block: usize,
    energy: &mut [f32],
) {
    let input_frames = input[0].len();
    let num_blocks = input_frames - (frames_per_block - 1);

    for k in 0..channels {
        let input_channel = &input[k];

        // First block of channel k.
        let mut sum = 0.0_f32;
        for m in 0..frames_per_block {
            let val = input_channel[m];
            sum += val * val;
        }
        energy[k] = sum;

        for n in 1..num_blocks {
            let slide_out = input_channel[n - 1];
            let slide_in = input_channel[n + frames_per_block - 1];
            energy[k + n * channels] = energy[k + (n - 1) * channels]
                - slide_out * slide_out
                + slide_in * slide_in;
        }
    }
}

fn peek_audio_with_zero_prepend(
    input_buffer: &[Vec<f32>],
    start_idx: usize,
    channels: usize,
    read_offset_frames: isize,
    dest: &mut [Vec<f32>],
    dest_frames: usize,
) {
    let mut write_offset = 0;
    let mut num_frames_to_read = dest_frames;
    let mut actual_read_offset = read_offset_frames;

    if read_offset_frames < 0 {
        let num_zero_frames_appended = (-read_offset_frames) as usize;
        let num_zero_frames_appended = num_zero_frames_appended.min(num_frames_to_read);
        actual_read_offset = 0;
        num_frames_to_read -= num_zero_frames_appended;
        write_offset = num_zero_frames_appended;

        for ch in 0..channels {
            dest[ch][0..num_zero_frames_appended].fill(0.0);
        }
    }

    if num_frames_to_read > 0 {
        for i in 0..channels {
            let actual_start = (actual_read_offset as usize) + start_idx;
            dest[i][write_offset..write_offset + num_frames_to_read].copy_from_slice(
                &input_buffer[i][actual_start..actual_start + num_frames_to_read]
            );
        }
    }
}

#[allow(dead_code)]
struct WsolaState {
    min_playback_rate: f32,
    max_playback_rate: f32,
    ola_window_size_ms: f32,
    wsola_search_interval_ms: f32,

    channels: usize,
    sample_rate: u32,

    muted_partial_frame: f64,
    output_time: f64,
    search_block_center_offset: usize,
    search_block_index: isize,
    num_candidate_blocks: usize,
    target_block_index: isize,
    ola_window_size: usize,
    ola_hop_size: usize,
    num_complete_frames: usize,
    wsola_output_started: bool,

    ola_window: Vec<f32>,
    transition_window: Vec<f32>,

    wsola_output: Vec<Vec<f32>>,
    wsola_output_size: usize,
    optimal_block: Vec<Vec<f32>>,
    search_block: Vec<Vec<f32>>,
    search_block_size: usize,
    target_block: Vec<Vec<f32>>,
    input_buffer: Vec<Vec<f32>>,
    input_buffer_start_idx: usize,
    
    input_buffer_final_frames: usize,
    input_buffer_added_silence: usize,
    energy_candidate_blocks: Vec<f32>,
    optimal_index: usize,
    is_final: bool,

    energy_target_block: Vec<f32>,
    dot_prod: Vec<f32>,
}

impl WsolaState {
    fn new(
        channels: usize,
        sample_rate: u32,
        min_playback_rate: f32,
        max_playback_rate: f32,
        ola_window_size_ms: f32,
        wsola_search_interval_ms: f32,
    ) -> Self {
        let mut num_candidate_blocks = (wsola_search_interval_ms * sample_rate as f32 / 1000.0) as usize;
        num_candidate_blocks = num_candidate_blocks.max(1);
        let mut ola_window_size = (ola_window_size_ms * sample_rate as f32 / 1000.0) as usize;
        ola_window_size = ola_window_size.max(2);
        ola_window_size += ola_window_size & 1;
        let ola_hop_size = ola_window_size / 2;

        let search_block_center_offset = num_candidate_blocks / 2 + (ola_window_size / 2 - 1);
        let ola_window = get_symmetric_hanning_window(ola_window_size);
        let transition_window = get_symmetric_hanning_window(2 * ola_window_size);

        let wsola_output_size = ola_window_size + ola_hop_size;

        let wsola_output = vec![vec![0.0_f32; wsola_output_size]; channels];
        let optimal_block = vec![vec![0.0_f32; ola_window_size]; channels];
        let search_block_size = num_candidate_blocks + (ola_window_size - 1);
        let search_block = vec![vec![0.0_f32; search_block_size]; channels];
        let target_block = vec![vec![0.0_f32; ola_window_size]; channels];
        
        let initial_size = 4 * ola_window_size.max(search_block_size);
        let input_buffer = vec![Vec::with_capacity(initial_size); channels];

        let energy_candidate_blocks = vec![0.0_f32; channels * num_candidate_blocks];
        let energy_target_block = vec![0.0_f32; channels];
        let dot_prod = vec![0.0_f32; channels];

        WsolaState {
            min_playback_rate,
            max_playback_rate,
            ola_window_size_ms,
            wsola_search_interval_ms,
            channels,
            sample_rate,
            muted_partial_frame: 0.0,
            output_time: 0.0,
            search_block_center_offset,
            search_block_index: 0,
            num_candidate_blocks,
            target_block_index: 0,
            ola_window_size,
            ola_hop_size,
            num_complete_frames: 0,
            wsola_output_started: false,
            ola_window,
            transition_window,
            wsola_output,
            wsola_output_size,
            optimal_block,
            search_block,
            search_block_size,
            target_block,
            input_buffer,
            input_buffer_start_idx: 0,
            input_buffer_final_frames: 0,
            input_buffer_added_silence: 0,
            energy_candidate_blocks,
            optimal_index: 0,
            is_final: false,
            energy_target_block,
            dot_prod,
        }
    }

    fn reset(&mut self) {
        for ch in 0..self.channels {
            self.input_buffer[ch].clear();
            self.wsola_output[ch].fill(0.0);
        }
        self.input_buffer_start_idx = 0;
        self.input_buffer_final_frames = 0;
        self.input_buffer_added_silence = 0;
        self.output_time = 0.0;
        self.search_block_index = 0;
        self.target_block_index = 0;
        self.num_complete_frames = 0;
        self.wsola_output_started = false;
        self.muted_partial_frame = 0.0;
        self.is_final = false;
    }

    fn input_buffer_frames(&self) -> usize {
        self.input_buffer[0].len() - self.input_buffer_start_idx
    }

    fn seek_buffer(&mut self, frames: usize) {
        assert!(self.input_buffer_frames() >= frames);
        if self.input_buffer_final_frames > 0 {
            self.input_buffer_final_frames = self.input_buffer_final_frames.saturating_sub(frames);
        }
        self.input_buffer_start_idx += frames;

        // Amortized shift to prevent unbounded growth.
        let threshold = 4096.max(2 * self.ola_window_size.max(self.search_block_size));
        if self.input_buffer_start_idx >= threshold {
            self.compact_input_buffer();
        }
    }

    fn compact_input_buffer(&mut self) {
        if self.input_buffer_start_idx == 0 {
            return;
        }
        let start = self.input_buffer_start_idx;
        for i in 0..self.channels {
            let len = self.input_buffer[i].len();
            self.input_buffer[i].copy_within(start..len, 0);
            self.input_buffer[i].truncate(len - start);
        }
        self.input_buffer_start_idx = 0;
    }

    fn set_final(&mut self) {
        if !self.is_final {
            self.input_buffer_final_frames = self.input_buffer_frames();
            self.is_final = true;
        }
    }

    fn target_is_within_search_region(&self) -> bool {
        self.target_block_index >= self.search_block_index
            && self.target_block_index + self.ola_window_size as isize
                <= self.search_block_index + self.search_block_size as isize
    }

    fn get_optimal_block(&mut self) {
        let exclude_interval_length_frames = 160;
        if self.target_is_within_search_region() {
            self.optimal_index = self.target_block_index as usize;
            peek_audio_with_zero_prepend(
                &self.input_buffer,
                self.input_buffer_start_idx,
                self.channels,
                self.target_block_index,
                &mut self.optimal_block,
                self.ola_window_size,
            );
        } else {
            peek_audio_with_zero_prepend(
                &self.input_buffer,
                self.input_buffer_start_idx,
                self.channels,
                self.target_block_index,
                &mut self.target_block,
                self.ola_window_size,
            );
            peek_audio_with_zero_prepend(
                &self.input_buffer,
                self.input_buffer_start_idx,
                self.channels,
                self.search_block_index,
                &mut self.search_block,
                self.search_block_size,
            );

            let last_optimal = self.target_block_index
                - self.ola_hop_size as isize
                - self.search_block_index;
            let exclude_interval = (
                last_optimal - exclude_interval_length_frames / 2,
                last_optimal + exclude_interval_length_frames / 2,
            );

            let mut optimal_index = compute_optimal_index(
                &self.search_block,
                self.search_block_size,
                &self.target_block,
                self.ola_window_size,
                &mut self.energy_candidate_blocks,
                self.channels,
                exclude_interval,
                &mut self.energy_target_block,
                &mut self.dot_prod,
            );

            optimal_index = (optimal_index as isize + self.search_block_index) as usize;
            peek_audio_with_zero_prepend(
                &self.input_buffer,
                self.input_buffer_start_idx,
                self.channels,
                optimal_index as isize,
                &mut self.optimal_block,
                self.ola_window_size,
            );

            for k in 0..self.channels {
                let opt = &mut self.optimal_block[k];
                let tgt = &self.target_block[k];
                for n in 0..self.ola_window_size {
                    opt[n] = opt[n] * self.transition_window[n]
                        + tgt[n] * self.transition_window[self.ola_window_size + n];
                }
            }
            self.optimal_index = optimal_index;
        }

        self.target_block_index = (self.optimal_index + self.ola_hop_size) as isize;
    }

    fn clamp_playback_rate(&self, rate: f32) -> f32 {
        if rate.is_nan() || rate <= 0.0 {
            1.0
        } else {
            rate.clamp(self.min_playback_rate, self.max_playback_rate)
        }
    }

    fn get_updated_time(&self, playback_rate: f32) -> f64 {
        let playback_rate = self.clamp_playback_rate(playback_rate);
        self.output_time + self.ola_hop_size as f64 * playback_rate as f64
    }

    fn get_search_block_index(&self, output_time: f64) -> isize {
        (output_time - self.search_block_center_offset as f64 + 0.5).floor() as isize
    }

    fn frames_needed(&self, playback_rate: f32) -> isize {
        let playback_rate = self.clamp_playback_rate(playback_rate);
        let next_time = self.get_updated_time(playback_rate);
        let search_idx = self.get_search_block_index(next_time);
        
        let target_needed = self.target_block_index + self.ola_window_size as isize - self.input_buffer_frames() as isize;
        let search_needed = search_idx + self.search_block_size as isize - self.input_buffer_frames() as isize;
        
        target_needed.max(search_needed).max(0)
    }

    fn can_perform_wsola(&self, playback_rate: f32) -> bool {
        let playback_rate = self.clamp_playback_rate(playback_rate);
        self.frames_needed(playback_rate) <= 0
    }

    fn add_input_buffer_final_silence(&mut self, playback_rate: f32) {
        let playback_rate = self.clamp_playback_rate(playback_rate);
        let needed = self.frames_needed(playback_rate);
        if needed <= 0 {
            return;
        }

        let needed_usize = needed as usize;
        for ch in 0..self.channels {
            let len = self.input_buffer[ch].len();
            self.input_buffer[ch].resize(len + needed_usize, 0.0);
        }
        self.input_buffer_added_silence += needed_usize;
    }

    fn run_one_wsola_iteration(&mut self, playback_rate: f32) -> bool {
        let playback_rate = self.clamp_playback_rate(playback_rate);
        if !self.can_perform_wsola(playback_rate) {
            return false;
        }

        let next_output_time = self.output_time + self.ola_hop_size as f64 * playback_rate as f64;
        self.output_time = next_output_time;
        self.search_block_index = (next_output_time - self.search_block_center_offset as f64 + 0.5).floor() as isize;

        self.remove_old_input_frames();

        assert!(self.search_block_index + self.search_block_size as isize <= self.input_buffer_frames() as isize);

        self.get_optimal_block();

        for k in 0..self.channels {
            if self.wsola_output_started {
                for n in 0..self.ola_hop_size {
                    let out_idx = self.num_complete_frames + n;
                    self.wsola_output[k][out_idx] = self.wsola_output[k][out_idx] * self.ola_window[self.ola_hop_size + n]
                        + self.optimal_block[k][n] * self.ola_window[n];
                }
                let dest_start = self.num_complete_frames + self.ola_hop_size;
                self.wsola_output[k][dest_start..dest_start + self.ola_hop_size].copy_from_slice(
                    &self.optimal_block[k][self.ola_hop_size..self.ola_window_size]
                );
            } else {
                self.wsola_output[k][self.num_complete_frames..self.num_complete_frames + self.ola_window_size].copy_from_slice(
                    &self.optimal_block[k][0..self.ola_window_size]
                );
            }
        }

        self.num_complete_frames += self.ola_hop_size;
        self.wsola_output_started = true;
        true
    }

    fn remove_old_input_frames(&mut self) {
        let earliest_used_index = self.target_block_index.min(self.search_block_index);
        if earliest_used_index <= 0 {
            return;
        }

        let frames = earliest_used_index as usize;
        self.seek_buffer(frames);
        self.target_block_index -= earliest_used_index;
        self.output_time -= earliest_used_index as f64;
        self.search_block_index -= earliest_used_index;
    }

    fn write_completed_frames_to(&mut self, requested_frames: usize, dest: &mut [Vec<f32>], dest_offset: usize) -> usize {
        let rendered_frames = self.num_complete_frames.min(requested_frames);
        if rendered_frames == 0 {
            return 0;
        }

        for ch in 0..self.channels {
            for f in 0..rendered_frames {
                dest[ch][dest_offset + f] = self.wsola_output[ch][f];
            }
            self.wsola_output[ch].copy_within(rendered_frames..self.wsola_output_size, 0);
            self.wsola_output[ch][self.wsola_output_size - rendered_frames..self.wsola_output_size].fill(0.0);
        }

        self.num_complete_frames -= rendered_frames;
        rendered_frames
    }

    fn read_input_buffer(&mut self, dest_size: usize, dest: &mut [Vec<f32>]) -> usize {
        let target_idx = self.target_block_index.max(0) as usize;
        let frames_to_copy = dest_size.min(self.input_buffer_frames().saturating_sub(target_idx));
        if frames_to_copy == 0 {
            return 0;
        }

        let start_idx = self.input_buffer_start_idx;
        for i in 0..self.channels {
            let actual_start = target_idx + start_idx;
            dest[i][0..frames_to_copy].copy_from_slice(
                &self.input_buffer[i][actual_start..actual_start + frames_to_copy]
            );
        }
        self.seek_buffer(frames_to_copy);
        frames_to_copy
    }

    fn fill_buffer(&mut self, dest: &mut [Vec<f32>], dest_size: usize, playback_rate: f32) -> usize {
        let playback_rate = self.clamp_playback_rate(playback_rate);

        if self.input_buffer_final_frames > 0 {
            self.add_input_buffer_final_silence(playback_rate);
        }

        let slower_step = (self.ola_window_size as f32 * playback_rate).ceil() as usize;
        let faster_step = (self.ola_window_size as f32 / playback_rate).ceil() as usize;

        if self.ola_window_size <= faster_step && slower_step >= self.ola_window_size {
            if self.wsola_output_started {
                self.wsola_output_started = false;
                let sync_time = self.target_block_index;
                self.output_time = sync_time as f64;
                self.search_block_index = self.get_search_block_index(self.output_time);
                self.remove_old_input_frames();
            }

            return self.read_input_buffer(dest_size, dest);
        }

        let mut rendered_frames = 0;
        loop {
            let wrote = self.write_completed_frames_to(dest_size - rendered_frames, dest, rendered_frames);
            rendered_frames += wrote;
            
            if rendered_frames >= dest_size {
                break;
            }

            if !self.run_one_wsola_iteration(playback_rate) {
                break;
            }
        }
        rendered_frames
    }

    #[allow(dead_code)]
    fn frames_available(&self, playback_rate: f32) -> bool {
        (self.input_buffer_final_frames > self.target_block_index.max(0) as usize && self.input_buffer_final_frames > 0)
            || self.can_perform_wsola(playback_rate)
            || self.num_complete_frames > 0
    }
}

pub struct Wsola<I>
where
    I: Source,
{
    input: I,
    speed: f32,
    
    min_playback_rate: f32,
    max_playback_rate: f32,
    ola_window_size_ms: f32,
    wsola_search_interval_ms: f32,

    channels: rodio::ChannelCount,
    sample_rate: rodio::SampleRate,

    state: Option<WsolaState>,
    
    output_samples: Vec<rodio::Sample>,
    output_samples_pos: usize,
    
    inner_eof: bool,

    // Reusable buffers to avoid allocations in `next()`
    temp_buffer: Vec<Vec<f32>>,
    temp_frame: Vec<f32>,
    fill_dest: Vec<Vec<f32>>,
}

impl<I> Wsola<I>
where
    I: Source,
{
    pub fn new(input: I, speed: f32) -> Self {
        let channels = input.channels();
        let sample_rate = input.sample_rate();
        let speed = if speed.is_nan() || speed <= 0.0 { 1.0 } else { speed };
        let min_playback_rate = 0.25;
        let max_playback_rate = 8.0;
        let speed = speed.clamp(min_playback_rate, max_playback_rate);

        let channels_usize = channels.get() as usize;
        let temp_buffer = vec![Vec::new(); channels_usize];
        let temp_frame = vec![0.0; channels_usize];
        let chunk_size = 256;
        let fill_dest = vec![vec![0.0; chunk_size]; channels_usize];

        Self {
            input,
            speed,
            min_playback_rate,
            max_playback_rate,
            ola_window_size_ms: 12.0,
            wsola_search_interval_ms: 40.0,
            channels,
            sample_rate,
            state: None,
            output_samples: Vec::new(),
            output_samples_pos: 0,
            inner_eof: false,
            temp_buffer,
            temp_frame,
            fill_dest,
        }
    }

    pub fn with_params(
        input: I,
        mut speed: f32,
        mut min_playback_rate: f32,
        mut max_playback_rate: f32,
        mut ola_window_size_ms: f32,
        mut wsola_search_interval_ms: f32,
    ) -> Self {
        let channels = input.channels();
        let sample_rate = input.sample_rate();

        if min_playback_rate.is_nan() || min_playback_rate <= 0.0 {
            min_playback_rate = 0.25;
        } else {
            min_playback_rate = min_playback_rate.max(0.01);
        }

        if max_playback_rate.is_nan() || max_playback_rate < min_playback_rate {
            max_playback_rate = min_playback_rate.max(8.0);
        }

        if speed.is_nan() || speed <= 0.0 {
            speed = 1.0;
        }
        speed = speed.clamp(min_playback_rate, max_playback_rate);

        if ola_window_size_ms.is_nan() || ola_window_size_ms <= 0.0 {
            ola_window_size_ms = 12.0;
        } else {
            ola_window_size_ms = ola_window_size_ms.max(1.0);
        }

        if wsola_search_interval_ms.is_nan() || wsola_search_interval_ms <= 0.0 {
            wsola_search_interval_ms = 40.0;
        } else {
            wsola_search_interval_ms = wsola_search_interval_ms.max(1.0);
        }

        let channels_usize = channels.get() as usize;
        let temp_buffer = vec![Vec::new(); channels_usize];
        let temp_frame = vec![0.0; channels_usize];
        let chunk_size = 256;
        let fill_dest = vec![vec![0.0; chunk_size]; channels_usize];

        Self {
            input,
            speed,
            min_playback_rate,
            max_playback_rate,
            ola_window_size_ms,
            wsola_search_interval_ms,
            channels,
            sample_rate,
            state: None,
            output_samples: Vec::new(),
            output_samples_pos: 0,
            inner_eof: false,
            temp_buffer,
            temp_frame,
            fill_dest,
        }
    }

    pub fn set_speed(&mut self, speed: f32) {
        let speed = if speed.is_nan() || speed <= 0.0 { 1.0 } else { speed };
        self.speed = speed.clamp(self.min_playback_rate, self.max_playback_rate);
    }

    pub fn playback_speed(&self) -> f32 {
        self.speed
    }

    pub fn into_inner(self) -> I {
        self.input
    }

    fn ensure_state(&mut self) -> &mut WsolaState {
        let channels = self.input.channels().get() as usize;
        let sample_rate = self.input.sample_rate().get();
        
        assert_eq!(
            channels,
            self.channels.get() as usize,
            "Wsola: input source channel count changed mid-stream"
        );
        assert_eq!(
            sample_rate,
            self.sample_rate.get(),
            "Wsola: input source sample rate changed mid-stream"
        );

        if self.state.is_none() {
            self.state = Some(WsolaState::new(
                channels,
                sample_rate,
                self.min_playback_rate,
                self.max_playback_rate,
                self.ola_window_size_ms,
                self.wsola_search_interval_ms,
            ));
            self.output_samples.clear();
            self.output_samples_pos = 0;
            self.inner_eof = false;
        }

        self.state.as_mut().unwrap()
    }
}

impl<I> Iterator for Wsola<I>
where
    I: Source,
{
    type Item = rodio::Sample;

    fn next(&mut self) -> Option<Self::Item> {
        if self.output_samples_pos < self.output_samples.len() {
            let sample = self.output_samples[self.output_samples_pos];
            self.output_samples_pos += 1;
            return Some(sample);
        }

        self.output_samples.clear();
        self.output_samples_pos = 0;

        let speed = self.speed;
        let channels = self.input.channels().get() as usize;

        let needed = {
            let state = self.ensure_state();
            state.frames_needed(speed)
        };

        for ch in 0..channels {
            self.temp_buffer[ch].clear();
        }
        let mut pulled_frames = 0;

        if needed > 0 && !self.inner_eof {
            for _ in 0..needed {
                self.temp_frame.fill(0.0);
                let mut read_ok = true;
                for ch in 0..channels {
                    if let Some(sample) = self.input.next() {
                        self.temp_frame[ch] = sample as f32;
                    } else {
                        read_ok = false;
                        break;
                    }
                }

                if read_ok {
                    for ch in 0..channels {
                        self.temp_buffer[ch].push(self.temp_frame[ch]);
                    }
                    pulled_frames += 1;
                } else {
                    self.inner_eof = true;
                    break;
                }
            }
        }

        let inner_eof = self.inner_eof;
        self.ensure_state();
        let state = self.state.as_mut().unwrap();
        if pulled_frames > 0 {
            for ch in 0..channels {
                state.input_buffer[ch].extend_from_slice(&self.temp_buffer[ch]);
            }
        }
        if inner_eof {
            state.set_final();
        }

        let chunk_size = 256;
        let rendered_frames = state.fill_buffer(&mut self.fill_dest, chunk_size, speed);

        if rendered_frames > 0 {
            self.output_samples.reserve(rendered_frames * channels);
            for f in 0..rendered_frames {
                for ch in 0..channels {
                    self.output_samples.push(self.fill_dest[ch][f] as rodio::Sample);
                }
            }
            
            let sample = self.output_samples[0];
            self.output_samples_pos = 1;
            Some(sample)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let buffered = self.output_samples.len().saturating_sub(self.output_samples_pos);
        let (lower, upper) = self.input.size_hint();
        
        let lower_scaled = if self.speed > 0.0 {
            (lower as f32 / self.speed) as usize
        } else {
            lower
        };
        
        let upper_scaled = upper.map(|u| {
            if self.speed > 0.0 {
                (u as f32 / self.speed) as usize
            } else {
                u
            }
        });
        
        (buffered.saturating_add(lower_scaled), upper_scaled.map(|u| buffered.saturating_add(u)))
    }
}

impl<I> Source for Wsola<I>
where
    I: Source,
{
    fn current_span_len(&self) -> Option<usize> {
        None
    }

    fn channels(&self) -> rodio::ChannelCount {
        self.channels
    }

    fn sample_rate(&self) -> rodio::SampleRate {
        self.sample_rate
    }

    fn total_duration(&self) -> Option<Duration> {
        self.input.total_duration().map(|d| d.div_f32(self.speed))
    }

    fn try_seek(&mut self, pos: Duration) -> Result<(), SeekError> {
        let pos_accounting_for_speedup = pos.mul_f32(self.speed);
        self.input.try_seek(pos_accounting_for_speedup)?;
        if let Some(state) = &mut self.state {
            state.reset();
        }
        self.output_samples.clear();
        self.output_samples_pos = 0;
        self.inner_eof = false;
        Ok(())
    }
}

pub trait WsolaSourceExt: Source + Sized {
    fn wsola(self, speed: f32) -> Wsola<Self> {
        Wsola::new(self, speed)
    }
}

impl<I: Source> WsolaSourceExt for I {}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockSource {
        samples: Vec<rodio::Sample>,
        pos: usize,
        channels: u16,
        sample_rate: u32,
    }

    impl Iterator for MockSource {
        type Item = rodio::Sample;
        fn next(&mut self) -> Option<Self::Item> {
            if self.pos < self.samples.len() {
                let s = self.samples[self.pos];
                self.pos += 1;
                Some(s)
            } else {
                None
            }
        }
        fn size_hint(&self) -> (usize, Option<usize>) {
            let remaining = self.samples.len().saturating_sub(self.pos);
            (remaining, Some(remaining))
        }
    }

    impl Source for MockSource {
        fn current_span_len(&self) -> Option<usize> {
            Some(self.samples.len().saturating_sub(self.pos))
        }
        fn channels(&self) -> rodio::ChannelCount {
            rodio::ChannelCount::new(self.channels).unwrap()
        }
        fn sample_rate(&self) -> rodio::SampleRate {
            rodio::SampleRate::new(self.sample_rate).unwrap()
        }
        fn total_duration(&self) -> Option<Duration> {
            None
        }
        fn try_seek(&mut self, _pos: Duration) -> Result<(), SeekError> {
            self.pos = 0;
            Ok(())
        }
    }

    struct ChangingMockSource {
        samples: Vec<rodio::Sample>,
        pos: usize,
        channels: u16,
        sample_rate: u32,
        change_at: usize,
        new_channels: u16,
    }

    impl Iterator for ChangingMockSource {
        type Item = rodio::Sample;
        fn next(&mut self) -> Option<Self::Item> {
            if self.pos < self.samples.len() {
                let s = self.samples[self.pos];
                self.pos += 1;
                Some(s)
            } else {
                None
            }
        }
    }

    impl Source for ChangingMockSource {
        fn current_span_len(&self) -> Option<usize> {
            None
        }
        fn channels(&self) -> rodio::ChannelCount {
            if self.pos >= self.change_at {
                rodio::ChannelCount::new(self.new_channels).unwrap()
            } else {
                rodio::ChannelCount::new(self.channels).unwrap()
            }
        }
        fn sample_rate(&self) -> rodio::SampleRate {
            rodio::SampleRate::new(self.sample_rate).unwrap()
        }
        fn total_duration(&self) -> Option<Duration> {
            None
        }
        fn try_seek(&mut self, _pos: Duration) -> Result<(), SeekError> {
            Ok(())
        }
    }

    #[test]
    fn test_parameter_sanitization() {
        let input = MockSource {
            samples: vec![0.0 as rodio::Sample; 1000],
            pos: 0,
            channels: 2,
            sample_rate: 44100,
        };

        // Negative/NaN/Zero speed clamps to 1.0 or bounds
        let wsola_neg = Wsola::new(input, -1.5);
        assert_eq!(wsola_neg.playback_speed(), 1.0); // Defaults to 1.0 for <= 0.0

        let input = MockSource {
            samples: vec![0.0 as rodio::Sample; 1000],
            pos: 0,
            channels: 2,
            sample_rate: 44100,
        };
        let wsola_nan = Wsola::new(input, f32::NAN);
        assert_eq!(wsola_nan.playback_speed(), 1.0); // Default to 1.0

        let input = MockSource {
            samples: vec![0.0 as rodio::Sample; 1000],
            pos: 0,
            channels: 2,
            sample_rate: 44100,
        };
        let wsola_with_params = Wsola::with_params(
            input,
            12.0, // speed
            0.5,  // min
            4.0,  // max
            -5.0, // window (invalid)
            0.0,  // search (invalid)
        );
        assert_eq!(wsola_with_params.playback_speed(), 4.0); // Clamped to max
        assert_eq!(wsola_with_params.ola_window_size_ms, 12.0); // default
        assert_eq!(wsola_with_params.wsola_search_interval_ms, 40.0); // default
    }

    #[test]
    fn test_speed_clamping() {
        let input = MockSource {
            samples: vec![0.0 as rodio::Sample; 4000],
            pos: 0,
            channels: 1,
            sample_rate: 16000,
        };

        // Create Wsola with speed 10.0 (max is 8.0 by default)
        let mut wsola = Wsola::new(input, 10.0);
        assert_eq!(wsola.playback_speed(), 8.0); // Clamped to max

        // Verify it runs and produces samples instead of silence or panic
        let mut count = 0;
        while let Some(_) = wsola.next() {
            count += 1;
        }
        assert!(count > 0);
    }

    #[test]
    #[should_panic(expected = "Wsola: input source channel count changed mid-stream")]
    fn test_channel_mismatch_panic() {
        let input = ChangingMockSource {
            samples: vec![0.0 as rodio::Sample; 1000],
            pos: 0,
            channels: 2,
            sample_rate: 16000,
            change_at: 200,
            new_channels: 1,
        };

        let mut wsola = Wsola::new(input, 1.5);
        // This will process samples and eventually panic when it crosses change_at
        while let Some(_) = wsola.next() {}
    }

    #[test]
    fn test_size_hint() {
        let input = MockSource {
            samples: vec![0.0 as rodio::Sample; 2000],
            pos: 0,
            channels: 2,
            sample_rate: 44100,
        };

        let wsola = Wsola::new(input, 2.0);
        let hint = wsola.size_hint();
        // Since speed is 2.0 and input has 2000 samples, hint lower bound should be around 1000.
        assert!(hint.0 >= 990 && hint.0 <= 1010);
    }

    #[test]
    fn test_amortized_compaction() {
        // Create a large input (e.g. 60,000 stereo frames) to trigger compaction multiple times.
        let mut samples = Vec::new();
        for i in 0..60000 {
            let val = (i as f32).sin() * 0.5;
            samples.push(val);
            samples.push(val);
        }

        let input = MockSource {
            samples: samples.clone(),
            pos: 0,
            channels: 2,
            sample_rate: 44100,
        };

        let mut wsola = Wsola::new(input, 1.5);
        let mut output = Vec::new();
        while let Some(sample) = wsola.next() {
            output.push(sample);
        }

        // input has 120,000 samples. Output should be roughly 120,000 / 1.5 = 80,000 samples.
        assert!(output.len() > 70000 && output.len() < 90000);
        // Verify output contains non-silent samples (not corrupted/cleared incorrectly)
        assert!(output.iter().any(|&x| x.abs() > 0.01));
    }
}
