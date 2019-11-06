conda create -n speech_env python=3.7
conda activate speech_env
pip install python_speech_features pickle-mixin scipy numpy pandas scikit-learn torch torchvision tqdm
conda deactivate
echo "speech_env" conda environment created
