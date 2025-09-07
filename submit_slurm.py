#!/usr/bin/env python3
"""
Speaker Analysis Orchestrator
Processes audio files for multiple speakers through WhisperX transcription and Ollama analysis.
Organizes results based on analysis scores and manages SLURM job submission.
"""

import os
import sys
import subprocess
import time
import shutil
from pathlib import Path
import argparse
import json
from datetime import datetime

# Configuration
AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wmv", ".avi", ".mp4")
SCORE_THRESHOLD = 75

class SpeakerAnalysisOrchestrator:
    def __init__(self, base_dir, hf_token):
        self.base_dir = Path(base_dir)
        self.hf_token = hf_token
        self.job_ids = []
        self.speaker_folders = []
        self.results = {}
        
    def discover_speaker_folders(self):
        """Find all speaker folders containing audio files"""
        print("Discovering speaker folders...")
        self.speaker_folders = []
        
        for item in self.base_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Check if folder contains audio files
                audio_files = [f for f in item.rglob("*") 
                             if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS]
                if audio_files:
                    self.speaker_folders.append(item)
                    print(f"  Found speaker folder: {item.name} ({len(audio_files)} audio files)")
        
        print(f"Total speaker folders found: {len(self.speaker_folders)}")
        return self.speaker_folders
    
    def get_optimal_gpu_config(self, num_jobs):
        """Determine optimal GPU configuration based on job count and availability"""
        try:
            # Check if hyakalloc is available
            result = subprocess.run(['which', 'hyakalloc'], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            
            if result.returncode != 0:
                print("hyakalloc not available, using default config")
                return "gpu-h200", 1, 32
            
            # Get available GPUs using hyakalloc
            result = subprocess.run(['hyakalloc', '-p', 'gpu-h200'], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            
            if result.returncode != 0:
                print("Warning: Could not query GPU status with hyakalloc, using default config")
                return "gpu-h200", 1, 32
            
            # Parse hyakalloc output to find idle GPUs
            idle_gpus = 0
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Idle:' in line and '│' in line:
                    # The format is: "│ Idle: │ 2725 │  197 │"
                    parts = line.split('│')
                    if len(parts) >= 4:
                        try:
                            idle_gpus = int(parts[3].strip())
                            break
                        except ValueError:
                            continue
            
            if idle_gpus == 0:
                print("No idle GPUs available, using default config")
                return "gpu-h200", 1, 32
            
            # Determine strategy based on job count and GPU availability
            if num_jobs <= idle_gpus:
                # More GPUs than jobs - use one GPU per job
                gpus_per_job = 1
                mem_per_job = 32  # GB
                partition = "gpu-h200"
            elif num_jobs <= idle_gpus * 2:
                # 2 jobs per GPU
                gpus_per_job = 1
                mem_per_job = 16  # GB
                partition = "gpu-h200"
            else:
                # Many jobs - use shared GPU
                gpus_per_job = 1
                mem_per_job = 8   # GB
                partition = "gpu-h200"
            
            print(f"GPU Strategy: {num_jobs} jobs, {idle_gpus} idle GPUs available")
            print(f"Config: {gpus_per_job} GPU per job, {mem_per_job}GB memory, partition={partition}")
            
            return partition, gpus_per_job, mem_per_job
            
        except FileNotFoundError:
            print("hyakalloc not found, using default config")
            return "gpu-h200", 1, 32
        except Exception as e:
            print(f"Warning: GPU detection failed ({e}), using default config")
            return "gpu-h200", 1, 32
    
    def create_slurm_job_script(self, speaker_folder, job_type="whisperx", gpu_config=None):
        """Create a SLURM job script for a specific speaker folder"""
        job_name = f"{speaker_folder.name}_{job_type}"
        script_path = self.base_dir / f"{job_name}.slurm"
        
        # Use provided config or default
        if gpu_config:
            partition, gpus_per_job, mem_gb = gpu_config
        else:
            partition, gpus_per_job, mem_gb = "gpu-h200", 1, 32
        
        if job_type == "whisperx":
            script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER@uw.edu
#SBATCH --account=uwit
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus={gpus_per_job}
#SBATCH --mem={mem_gb}G
#SBATCH --time=02:00:00
#SBATCH --export=ALL
#SBATCH --output={self.base_dir}/logs/{job_name}_%j.out
#SBATCH --error={self.base_dir}/logs/{job_name}_%j.err

# Load modules
module load apptainer

# Set up environment
export HF_TOKEN={self.hf_token}
mkdir -p /tmp/$USER/.cache
export XDG_CACHE_HOME=/tmp/$USER/.cache

# Change to base directory first
cd "{self.base_dir}"

# Create output directories
mkdir -p "{speaker_folder.name}/needs_further_attention"
mkdir -p "{speaker_folder.name}/reviewed"

# Run WhisperX transcription using container
apptainer run --nv --bind /gscratch /mmfs1/gscratch/fellows/dawnmai/whisperx_python.sif python3 /mmfs1/gscratch/fellows/dawnmai/transcribe_calls.py "{speaker_folder.name}" --format vtt

# Start Ollama server and run analysis using container
apptainer run --nv --bind /gscratch /mmfs1/gscratch/fellows/dawnmai/ollama_python.sif bash -c "
pip install requests
export CUDA_VISIBLE_DEVICES=0
export OLLAMA_HOST=0.0.0.0:11434
echo 'Starting Ollama server...'
nohup ollama serve > /dev/null 2>&1 &
sleep 20
echo 'Pulling model llama3.2:3b...'
ollama pull llama3.2:3b
echo 'Model pull completed, waiting for model to be ready...'
sleep 40
echo 'Starting analysis...'
python3 /mmfs1/gscratch/fellows/dawnmai/analyze_with_ollama.py '{speaker_folder.name}' --threshold {SCORE_THRESHOLD}
"

echo "Job completed for {speaker_folder.name}"
"""
        else:  # ollama analysis
            script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER@uw.edu
#SBATCH --account=uwit
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --export=ALL
#SBATCH --output={self.base_dir}/logs/{job_name}_%j.out
#SBATCH --error={self.base_dir}/logs/{job_name}_%j.err

# Load modules
module load apptainer

# Start Ollama server and run analysis using container
cd "{self.base_dir}"
apptainer run --nv --bind /gscratch /mmfs1/gscratch/fellows/dawnmai/ollama_python.sif bash -c "ollama serve & sleep 15 && python3 /mmfs1/gscratch/fellows/dawnmai/analyze_with_ollama.py '{speaker_folder}' --threshold {SCORE_THRESHOLD}"

echo "Ollama analysis completed for {speaker_folder.name}"
"""
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        return script_path
    
    def submit_slurm_job(self, script_path):
        """Submit a SLURM job and return job ID"""
        try:
            result = subprocess.run(['sbatch', str(script_path)], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                  universal_newlines=True, check=True)
            job_id = result.stdout.strip().split()[-1]
            print(f"  Submitted job {job_id}: {script_path.name}")
            return job_id
        except subprocess.CalledProcessError as e:
            print(f"  Failed to submit job: {e.stderr}")
            return None
    
    def monitor_jobs(self):
        """Monitor SLURM jobs until all are complete"""
        if not self.job_ids:
            print("No jobs to monitor")
            return
        
        print(f"Monitoring {len(self.job_ids)} SLURM jobs...")
        
        # Get netid for squeue command
        netid = os.environ.get('USER', 'unknown')
        
        while True:
            running_jobs = []
            for job_id in self.job_ids:
                try:
                    result = subprocess.run(['squeue', '-j', job_id, '--noheader'], 
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                    if result.returncode == 0 and result.stdout.strip():
                        running_jobs.append(job_id)
                except subprocess.CalledProcessError:
                    # Job might be finished
                    pass
            
            if not running_jobs:
                print("All jobs completed!")
                break
            
            # Use squeue -u <netid> as confirmation instead of counting jobs
            print(f"  Checking job status with squeue...")
            try:
                result = subprocess.run(['squeue', '-u', netid], 
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                if result.returncode == 0:
                    print(result.stdout)
                else:
                    print(f"  squeue command failed: {result.stderr}")
            except subprocess.CalledProcessError as e:
                print(f"  Error running squeue: {e}")
            
            time.sleep(300)  # Check every 5 minutes
    
    def organize_results(self, speaker_folder):
        """Organize files based on analysis scores"""
        print(f"Organizing results for {speaker_folder.name}...")
        
        # Read analysis results
        results_file = speaker_folder / "analysis_results.json"
        if not results_file.exists():
            print(f"  No analysis results found for {speaker_folder.name}")
            return
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        needs_attention = speaker_folder / "needs_further_attention"
        reviewed = speaker_folder / "reviewed"
        
        needs_attention.mkdir(exist_ok=True)
        reviewed.mkdir(exist_ok=True)
        
        for transcription_file, data in results.items():
            score = data.get('score', 0)
            audio_file = data.get('audio_file', '')
            
            # Use the audio file name from the analysis results
            if audio_file:
                audio_path = speaker_folder / audio_file
            else:
                # Fallback to transcription file name
                audio_path = speaker_folder / transcription_file
            
            # Create call-specific folder name (without extension)
            call_name = audio_path.stem  # Gets filename without extension
            
            if score <= SCORE_THRESHOLD:
                target_dir = needs_attention / call_name
            else:
                target_dir = reviewed / call_name
            
            # Create the call-specific directory
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Move audio file
            if audio_path.exists():
                shutil.move(str(audio_path), str(target_dir / audio_path.name))
            
            # Move associated transcription files
            for ext in ['srt', 'vtt', 'txt', 'json']:
                trans_file = audio_path.with_suffix(f'.{ext}')
                if trans_file.exists():
                    shutil.move(str(trans_file), str(target_dir / trans_file.name))
            
            # Move analysis results JSON to the call-specific folder
            analysis_file = target_dir / "analysis_results.json"
            if results_file.exists() and not analysis_file.exists():
                # Create a call-specific analysis results file
                call_analysis = {transcription_file: data}
                with open(analysis_file, 'w') as f:
                    json.dump(call_analysis, f, indent=2)
        
        print(f"  Organized {len(results)} files for {speaker_folder.name}")
        
        # Clean up the main analysis results file after organization
        if results_file.exists():
            results_file.unlink()
        
        # Clean up the SLURM script file
        slurm_file = self.base_dir / f"{speaker_folder.name}_whisperx.slurm"
        if slurm_file.exists():
            slurm_file.unlink()
            print(f"  Deleted SLURM script: {slurm_file.name}")
    
    
    def run_analysis(self):
        """Main orchestration method"""
        print("Starting Speaker Analysis Orchestrator")
        print(f"Base directory: {self.base_dir}")
        print(f"Score threshold: {SCORE_THRESHOLD}")
        print("-" * 50)
        
        # Step 1: Discover speaker folders
        self.discover_speaker_folders()
        if not self.speaker_folders:
            print("No speaker folders found!")
            return
        
        # Create logs directory
        (self.base_dir / "logs").mkdir(exist_ok=True)
        
        # Step 2: Get optimal GPU configuration and submit WhisperX jobs
        print("\nDetermining optimal GPU configuration...")
        gpu_config = self.get_optimal_gpu_config(len(self.speaker_folders))
        
        print("\nSubmitting WhisperX transcription jobs...")
        for speaker_folder in self.speaker_folders:
            script_path = self.create_slurm_job_script(speaker_folder, "whisperx", gpu_config)
            job_id = self.submit_slurm_job(script_path)
            if job_id:
                self.job_ids.append(job_id)
        
        # Step 3: Monitor jobs
        self.monitor_jobs()
        
        # Step 4: Organize results
        print("\nOrganizing results...")
        for speaker_folder in self.speaker_folders:
            self.organize_results(speaker_folder)
        
        print("\nAnalysis complete!")
        print(f"Processed {len(self.speaker_folders)} speaker folders")

def main():
    global SCORE_THRESHOLD
    
    parser = argparse.ArgumentParser(description="Speaker Analysis Orchestrator")
    parser.add_argument("base_dir", help="Base directory containing speaker folders")
    parser.add_argument("--hf-token", required=True, help="Hugging Face token for WhisperX")
    parser.add_argument("--threshold", type=int, default=SCORE_THRESHOLD, 
                       help=f"Score threshold for organization (default: {SCORE_THRESHOLD})")
    
    args = parser.parse_args()
    
    # Update global threshold
    SCORE_THRESHOLD = args.threshold
    
    orchestrator = SpeakerAnalysisOrchestrator(
        args.base_dir, 
        args.hf_token
    )
    
    orchestrator.run_analysis()

if __name__ == "__main__":
    main()
